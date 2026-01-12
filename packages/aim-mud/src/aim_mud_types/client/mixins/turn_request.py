# aim-mud-types/client/mixins/turn_request.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""MUDTurnRequest-specific operations with CAS support."""

import logging
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from ...redis_keys import RedisKeys
from ...coordination import MUDTurnRequest, TurnRequestStatus
from ...helper import _utc_now

if TYPE_CHECKING:
    from .. import BaseRedisMUDClient


logger = logging.getLogger(__name__)


class TurnRequestMixin:
    """MUDTurnRequest-specific Redis operations.

    Provides CRUD operations for turn request coordination with
    CAS (Compare-And-Swap) support to prevent race conditions.
    """

    async def get_turn_request(self: "BaseRedisMUDClient", agent_id: str) -> Optional[MUDTurnRequest]:
        """Fetch turn request for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            MUDTurnRequest object, or None if not found
        """
        key = RedisKeys.agent_turn_request(agent_id)
        return await self._get_hash(MUDTurnRequest, key)

    async def create_turn_request(
        self: "BaseRedisMUDClient",
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
        return await self._create_hash(key, turn_request, exists_ok=False)

    async def update_turn_request(
        self: "BaseRedisMUDClient",
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
        return await self._update_hash(
            key,
            turn_request,
            cas_field="turn_id",
            cas_value=expected_turn_id
        )

    async def delete_turn_request(
        self: "BaseRedisMUDClient",
        agent_id: str,
    ) -> bool:
        """Delete turn request for an agent."""
        key = RedisKeys.agent_turn_request(agent_id)
        deleted = await self.redis.delete(key)
        return bool(deleted)

    async def heartbeat_turn_request(
        self: "BaseRedisMUDClient",
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
        return await self._update_fields(
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

    async def get_all_turn_requests(
        self: "BaseRedisMUDClient",
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
            cursor, batch = await self.redis.scan(cursor, match=match, count=100)
            keys.extend(batch)
            if cursor == 0:
                break

        if not keys:
            return []

        pipeline = self.redis.pipeline()
        for key in keys:
            pipeline.hgetall(key)
        results = await pipeline.execute()

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

    async def get_oldest_active_turn(
        self: "BaseRedisMUDClient",
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
            cursor, batch = await self.redis.scan(cursor, match=match, count=100)
            keys.extend(batch)
            if cursor == 0:
                break

        if not keys:
            return None

        pipeline = self.redis.pipeline()
        for key in keys:
            pipeline.hgetall(key)
        results = await pipeline.execute()

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
                    heartbeat_at = datetime.fromisoformat(
                        heartbeat_at_str.replace("Z", "+00:00")
                    )
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
                assigned_at = datetime.fromisoformat(
                    assigned_at_str.replace("Z", "+00:00")
                )
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
