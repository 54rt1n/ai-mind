# aim-mud-types/client/sync_mixins/turn_request.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Sync MUDTurnRequest-specific operations with CAS support."""

import logging
import uuid
from datetime import datetime
from typing import Optional, Tuple, TYPE_CHECKING

from ...redis_keys import RedisKeys
from ...models.coordination import MUDTurnRequest, TurnRequestStatus, TurnReason
from ...helper import _utc_now, _unix_to_datetime, _datetime_to_unix

if TYPE_CHECKING:
    from ..base import BaseSyncRedisMUDClient


logger = logging.getLogger(__name__)


class SyncTurnRequestMixin:
    """Sync MUDTurnRequest-specific Redis operations.

    Provides CRUD operations for turn request coordination with
    CAS (Compare-And-Swap) support to prevent race conditions.
    """

    def get_turn_request(self: "BaseSyncRedisMUDClient", agent_id: str) -> Optional[MUDTurnRequest]:
        """Fetch turn request for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            MUDTurnRequest object, or None if not found
        """
        key = RedisKeys.agent_turn_request(agent_id)
        return self._get_hash(MUDTurnRequest, key)

    def create_turn_request(
        self: "BaseSyncRedisMUDClient",
        agent_id: str,
        turn_request: MUDTurnRequest
    ) -> bool:
        """Create a new turn request.

        Uses atomic EXISTS check to prevent overwriting existing requests.

        Args:
            agent_id: Agent identifier
            turn_request: Complete MUDTurnRequest object

        Returns:
            True if created, False if turn request already exists
        """
        key = RedisKeys.agent_turn_request(agent_id)
        return self._create_hash(key, turn_request, exists_ok=False)

    def update_turn_request(
        self: "BaseSyncRedisMUDClient",
        agent_id: str,
        turn_request: MUDTurnRequest,
        expected_turn_id: str
    ) -> bool:
        """Update turn request with CAS on turn_id.

        Atomically checks that turn_id matches before updating.
        Prevents race conditions between workers.

        Args:
            agent_id: Agent identifier
            turn_request: Updated MUDTurnRequest object
            expected_turn_id: Expected turn_id value for CAS check

        Returns:
            True if updated, False if CAS check failed
        """
        key = RedisKeys.agent_turn_request(agent_id)
        return self._update_hash(
            key,
            turn_request,
            cas_field="turn_id",
            cas_value=expected_turn_id
        )

    def delete_turn_request(
        self: "BaseSyncRedisMUDClient",
        agent_id: str,
    ) -> bool:
        """Delete turn request for an agent."""
        key = RedisKeys.agent_turn_request(agent_id)
        deleted = self.redis.delete(key)
        return bool(deleted)

    def heartbeat_turn_request(
        self: "BaseSyncRedisMUDClient",
        agent_id: str
    ) -> bool:
        """Update heartbeat timestamp.

        Efficient single-field update for worker heartbeats.

        Args:
            agent_id: Agent identifier

        Returns:
            True if updated
        """
        key = RedisKeys.agent_turn_request(agent_id)
        return self._update_fields(
            key,
            {"heartbeat_at": _utc_now()}
        )

    def _agent_id_from_turn_request_key(self, key: str) -> str:
        """Extract agent_id from agent:{id}:turn_request key."""
        prefix = "agent:"
        suffix = ":turn_request"
        if key.startswith(prefix) and key.endswith(suffix):
            return key[len(prefix):-len(suffix)]
        return key

    def get_all_turn_requests(
        self: "BaseSyncRedisMUDClient",
        *,
        match: str = "agent:*:turn_request",
    ) -> list[tuple[str, MUDTurnRequest]]:
        """Fetch all turn_requests across agents.

        Returns:
            List of (agent_id, MUDTurnRequest) for valid entries.
        """
        keys: list[bytes | str] = []
        cursor = 0
        while True:
            cursor, batch = self.redis.scan(cursor, match=match, count=100)
            keys.extend(batch)
            if cursor == 0:
                break

        if not keys:
            return []

        pipeline = self.redis.pipeline()
        for key in keys:
            pipeline.hgetall(key)
        results = pipeline.execute()

        turn_requests: list[tuple[str, MUDTurnRequest]] = []
        for key, data in zip(keys, results):
            if not data:
                continue

            key_str = key.decode("utf-8") if isinstance(key, bytes) else str(key)
            decoded: dict[str, str] = {}
            for k, v in data.items():
                if isinstance(k, bytes):
                    k = k.decode("utf-8")
                if isinstance(v, bytes):
                    v = v.decode("utf-8")
                decoded[str(k)] = str(v)

            try:
                turn_request = MUDTurnRequest.model_validate(decoded)
            except Exception as e:
                logger.warning(
                    "Failed to validate MUDTurnRequest from key '%s': %s",
                    key_str,
                    e,
                    exc_info=True,
                )
                continue

            agent_id = self._agent_id_from_turn_request_key(key_str)
            turn_requests.append((agent_id, turn_request))

        return turn_requests

    def get_oldest_active_turn(
        self: "BaseSyncRedisMUDClient",
        *,
        heartbeat_stale_threshold: int = 300,
        active_statuses: Optional[set[TurnRequestStatus]] = None,
        exclude_statuses: Optional[set[TurnRequestStatus]] = None,
        match: str = "agent:*:turn_request",
    ) -> Optional[tuple[str, str, datetime]]:
        """Return oldest active turn by assigned_at.

        Returns:
            (agent_id, turn_id, assigned_at) or None if no active turns.
        """
        if active_statuses is None:
            active_statuses = {TurnRequestStatus.ASSIGNED, TurnRequestStatus.IN_PROGRESS}
        if exclude_statuses is None:
            exclude_statuses = {TurnRequestStatus.EXECUTE, TurnRequestStatus.EXECUTING}

        keys: list[bytes | str] = []
        cursor = 0
        while True:
            cursor, batch = self.redis.scan(cursor, match=match, count=100)
            keys.extend(batch)
            if cursor == 0:
                break

        if not keys:
            return None

        pipeline = self.redis.pipeline()
        for key in keys:
            pipeline.hgetall(key)
        results = pipeline.execute()

        oldest_assigned_at: Optional[datetime] = None
        oldest_turn_id: Optional[str] = None
        oldest_agent_id: Optional[str] = None

        active_status_values = {status.value for status in active_statuses}
        exclude_status_values = {status.value for status in exclude_statuses}

        for key, data in zip(keys, results):
            if not data:
                continue

            key_str = key.decode("utf-8") if isinstance(key, bytes) else str(key)
            decoded: dict[str, str] = {}
            for k, v in data.items():
                if isinstance(k, bytes):
                    k = k.decode("utf-8")
                if isinstance(v, bytes):
                    v = v.decode("utf-8")
                decoded[str(k)] = str(v)

            status = decoded.get("status", "")
            if status in exclude_status_values:
                continue
            if status not in active_status_values:
                continue

            heartbeat_at_str = decoded.get("heartbeat_at")
            if heartbeat_at_str:
                try:
                    # Handles both Unix timestamps and ISO format for backwards compatibility
                    heartbeat_at = _unix_to_datetime(heartbeat_at_str)
                    if heartbeat_at:
                        age = (_utc_now() - heartbeat_at).total_seconds()
                        if age > heartbeat_stale_threshold:
                            logger.warning(
                                "Turn guard: Ignoring stale turn %s (heartbeat %.0fs old)",
                                decoded.get("turn_id"),
                                age,
                            )
                            continue
                except (ValueError, AttributeError):
                    pass

            assigned_at_str = decoded.get("assigned_at")
            if not assigned_at_str:
                continue

            try:
                # Handles both Unix timestamps and ISO format for backwards compatibility
                assigned_at = _unix_to_datetime(assigned_at_str)
                if not assigned_at:
                    continue
            except (ValueError, AttributeError):
                continue

            turn_id = decoded.get("turn_id")
            if not turn_id:
                continue

            if oldest_assigned_at is None or assigned_at < oldest_assigned_at:
                oldest_assigned_at = assigned_at
                oldest_turn_id = turn_id
                oldest_agent_id = self._agent_id_from_turn_request_key(key_str)
            elif assigned_at == oldest_assigned_at:
                current_oldest_id = oldest_turn_id or ""
                if turn_id < current_oldest_id:
                    oldest_turn_id = turn_id
                    oldest_agent_id = self._agent_id_from_turn_request_key(key_str)

        if oldest_turn_id is None or oldest_assigned_at is None or oldest_agent_id is None:
            return None

        return oldest_agent_id, oldest_turn_id, oldest_assigned_at

    def initialize_turn_request(
        self: "BaseSyncRedisMUDClient",
        agent_id: str,
        *,
        status: TurnRequestStatus = TurnRequestStatus.READY,
        reason: TurnReason = TurnReason.EVENTS,
        status_reason: Optional[str] = None,
        attempt_count: int = 0,
        deadline_ms: int = 0,
        sequence_id: Optional[int] = None,
        **metadata_kwargs,
    ) -> Tuple[bool, Optional[MUDTurnRequest], str]:
        """Initialize agent turn_request if missing.

        Creates a new turn_request only if one does not already exist.
        Uses atomic create operation to prevent race conditions.

        Args:
            agent_id: Agent identifier
            status: Initial status (default: READY)
            reason: Initial reason (default: EVENTS)
            status_reason: Optional reason text
            attempt_count: Initial attempt count (default: 0)
            deadline_ms: Deadline in milliseconds (default: 0)
            sequence_id: Optional sequence ID (auto-generated if not provided)
            **metadata_kwargs: Additional metadata fields

        Returns:
            Tuple of (created, turn_request, result_message):
            - (True, turn_request, "created") on success
            - (False, existing_turn_request, "exists") if already present
            - (False, None, "invalid") if existing hash cannot be parsed
        """
        # Check if turn_request already exists
        existing = self.get_turn_request(agent_id)
        if existing:
            return False, existing, "exists"

        # Generate sequence_id if not provided
        if sequence_id is None:
            sequence_id = self.redis.incr(RedisKeys.SEQUENCE_COUNTER)

        now = _utc_now()
        turn_request = MUDTurnRequest(
            turn_id=str(uuid.uuid4()),
            sequence_id=sequence_id,
            status=status,
            reason=reason,
            assigned_at=now,
            heartbeat_at=now,
            completed_at=None,
            deadline_ms=str(deadline_ms),
            attempt_count=attempt_count,
            status_reason=status_reason,
            metadata={k: v for k, v in metadata_kwargs.items() if v is not None} or None,
        )

        # Attempt atomic create
        created = self.create_turn_request(agent_id, turn_request)
        if created:
            return True, turn_request, "created"

        # Race: someone else created it - fetch the existing one
        existing = self.get_turn_request(agent_id)
        if existing:
            return False, existing, "exists"
        return False, None, "exists"

    def assign_turn_request(
        self: "BaseSyncRedisMUDClient",
        agent_id: str,
        reason: "str | TurnReason",
        *,
        attempt_count: int = 0,
        deadline_ms: int = 120_000,
        status: Optional[TurnRequestStatus] = None,
        expected_turn_id: Optional[str] = None,
        skip_availability_check: bool = False,
        turn_id_prefix: str = "manual",
        **metadata_kwargs,
    ) -> Tuple[bool, str]:
        """Assign a new turn_request if the agent is available.

        Simple sync interface matching the original helper function signature.
        Does NOT use CAS - directly writes the new turn_request.

        Args:
            agent_id: Agent identifier
            reason: Turn reason (enum or string)
            attempt_count: Attempt count (default: 0)
            deadline_ms: Deadline in milliseconds (default: 120000)
            status: Initial status (auto-determined if None)
            expected_turn_id: Unused (kept for signature compatibility)
            skip_availability_check: Skip availability check (default: False)
            turn_id_prefix: Prefix for generated turn_id (default: "manual")
            **metadata_kwargs: Additional metadata fields

        Returns:
            (True, turn_id) on success
            (False, reason) on failure
        """
        import json

        # Check pause state
        pause_key = RedisKeys.agent_pause(agent_id)
        paused = self.redis.get(pause_key)
        if paused in (b"1", "1"):
            return False, "paused"

        # Get current turn_request for availability check
        key = RedisKeys.agent_turn_request(agent_id)
        current_raw = self.redis.hgetall(key) or {}
        if not current_raw:
            return False, "offline"

        # Decode and validate current turn_request
        decoded: dict[str, str] = {}
        for k, v in current_raw.items():
            if isinstance(k, bytes):
                k = k.decode("utf-8")
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            decoded[str(k)] = str(v)

        try:
            current = MUDTurnRequest.model_validate(decoded)
        except Exception:
            return False, "invalid"

        if not skip_availability_check and not current.is_available_for_assignment():
            return False, f"busy:{current.status.value}"

        # Determine reason and status
        reason_enum = reason if isinstance(reason, TurnReason) else TurnReason(reason)
        if status is None:
            status = TurnRequestStatus.EXECUTE if reason_enum.is_immediate_command() else TurnRequestStatus.ASSIGNED

        # Generate turn_id with prefix
        now = _utc_now()
        turn_id = f"{turn_id_prefix}_{int(now.timestamp() * 1000)}"
        sequence_id = self.redis.incr(RedisKeys.SEQUENCE_COUNTER)

        # Build payload
        now_ts = str(_datetime_to_unix(now))
        payload = {
            "turn_id": turn_id,
            "status": status.value,
            "reason": reason_enum.value,
            "assigned_at": now_ts,
            "heartbeat_at": now_ts,
            "deadline_ms": str(deadline_ms),
            "sequence_id": str(sequence_id),
            "attempt_count": str(attempt_count),
        }

        metadata = {k: v for k, v in metadata_kwargs.items() if v is not None}
        if metadata:
            payload["metadata"] = json.dumps(metadata)

        self.redis.hset(key, mapping=payload)

        return True, turn_id

    def assign_turn_request_with_cas(
        self: "BaseSyncRedisMUDClient",
        agent_id: str,
        reason: "str | TurnReason",
        *,
        attempt_count: int = 0,
        deadline_ms: int = 0,
        status: Optional[TurnRequestStatus] = None,
        expected_turn_id: Optional[str] = None,
        skip_availability_check: bool = False,
        **metadata_kwargs,
    ) -> Tuple[bool, Optional[MUDTurnRequest], str]:
        """Assign a new turn_request with CAS.

        Checks availability and assigns a new turn with atomic update.
        Uses CAS (Compare-And-Swap) for race-free updates.

        Args:
            agent_id: Agent identifier
            reason: Turn reason (enum or string)
            attempt_count: Attempt count (default: 0)
            deadline_ms: Deadline in milliseconds (default: 0)
            status: Initial status (auto-determined if None)
            expected_turn_id: Expected turn_id for CAS (fetched if None)
            skip_availability_check: Skip availability check (default: False)
            **metadata_kwargs: Additional metadata fields

        Returns:
            Tuple of (success, turn_request, result_message):
            - (True, turn_request, "ok") on success
            - (False, None, reason) on failure
        """
        # Check pause state
        pause_key = RedisKeys.agent_pause(agent_id)
        paused = self.redis.get(pause_key)
        if paused in (b"1", "1"):
            return False, None, "paused"

        # Get current turn_request for availability check and CAS
        current = None
        if not skip_availability_check or expected_turn_id is None:
            current = self.get_turn_request(agent_id)
            if not current:
                return False, None, "offline"
            if not skip_availability_check and not current.is_available_for_assignment():
                return False, None, f"busy:{current.status.value}"
            if expected_turn_id is None:
                expected_turn_id = current.turn_id

        if expected_turn_id is None:
            return False, None, "missing_expected_turn_id"

        # Determine reason and status
        reason_enum = reason if isinstance(reason, TurnReason) else TurnReason(reason)
        if status is None:
            status = TurnRequestStatus.EXECUTE if reason_enum.is_immediate_command() else TurnRequestStatus.ASSIGNED

        # Create new turn_request
        sequence_id = self.redis.incr(RedisKeys.SEQUENCE_COUNTER)
        now = _utc_now()
        turn_request = MUDTurnRequest(
            turn_id=str(uuid.uuid4()),
            sequence_id=sequence_id,
            status=status,
            reason=reason_enum,
            assigned_at=now,
            heartbeat_at=now,
            completed_at=None,
            deadline_ms=str(deadline_ms),
            attempt_count=attempt_count,
            metadata={k: v for k, v in metadata_kwargs.items() if v is not None} or None,
        )

        # Atomic update with CAS
        success = self.update_turn_request(agent_id, turn_request, expected_turn_id=expected_turn_id)
        if not success:
            return False, None, "cas_failed"

        return True, turn_request, "ok"

    def transition_turn_request_and_update(
        self: "BaseSyncRedisMUDClient",
        agent_id: str,
        turn_request: MUDTurnRequest,
        expected_turn_id: str,
        *,
        status: Optional[TurnRequestStatus] = None,
        message: Optional[str] = None,
        status_reason: Optional[str] = None,
        attempt_count: Optional[int] = None,
        next_attempt_at: Optional[str] = None,
        turn_id: Optional[str] = None,
        new_turn_id: bool = False,
        set_completed: bool = False,
        update_heartbeat: bool = False,
    ) -> bool:
        """Apply a transition and persist with CAS.

        Modifies the turn_request in-place with the specified changes,
        then atomically updates Redis with CAS on turn_id.

        Args:
            agent_id: Agent identifier
            turn_request: MUDTurnRequest to modify and update
            expected_turn_id: Expected turn_id for CAS check
            status: New status (optional)
            message: New message (optional)
            status_reason: New status reason (optional)
            attempt_count: New attempt count (optional)
            next_attempt_at: Next attempt timestamp (optional)
            turn_id: New turn_id (optional)
            new_turn_id: Generate new turn_id (default: False)
            set_completed: Set completed_at timestamp (default: False)
            update_heartbeat: Update heartbeat_at timestamp (default: False)

        Returns:
            True if updated, False if CAS check failed
        """
        # Apply transition to turn_request in-place
        if new_turn_id:
            turn_request.turn_id = str(uuid.uuid4())
        if turn_id is not None:
            turn_request.turn_id = turn_id
        if status is not None:
            turn_request.status = status
        if message is not None:
            turn_request.message = message
        if status_reason is not None:
            turn_request.status_reason = status_reason
        if attempt_count is not None:
            turn_request.attempt_count = attempt_count
        if next_attempt_at is not None:
            turn_request.next_attempt_at = next_attempt_at

        now = _utc_now()
        if set_completed:
            turn_request.completed_at = now
            if update_heartbeat:
                turn_request.heartbeat_at = now
        elif update_heartbeat:
            turn_request.heartbeat_at = now

        return self.update_turn_request(agent_id, turn_request, expected_turn_id=expected_turn_id)

    def atomic_heartbeat_update(
        self: "BaseSyncRedisMUDClient",
        agent_id: str,
    ) -> int:
        """Atomically update heartbeat with validation.

        Uses Lua script to atomically:
        1. Check key exists
        2. Verify required fields (status, turn_id) are present
        3. Update heartbeat_at timestamp

        This prevents TOCTOU race conditions where the key could be deleted
        between EXISTS check and HSET write, which would create a partial hash.

        Args:
            agent_id: Agent identifier

        Returns:
            1: Success - heartbeat updated
            0: Key doesn't exist (normal during shutdown)
            -1: Key corrupted (missing required fields)
        """
        key = RedisKeys.agent_turn_request(agent_id)
        lua_script = """
            local key = KEYS[1]
            local heartbeat_at = ARGV[1]

            if redis.call('EXISTS', key) == 0 then
                return 0
            end

            local status = redis.call('HGET', key, 'status')
            local turn_id = redis.call('HGET', key, 'turn_id')

            if not status or not turn_id then
                return -1
            end

            redis.call('HSET', key, 'heartbeat_at', heartbeat_at)

            return 1
        """

        result = self.redis.eval(
            lua_script,
            1,
            key,
            str(_datetime_to_unix(_utc_now())),
        )
        return int(result)
