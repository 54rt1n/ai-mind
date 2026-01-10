# andimud_worker/datastore/adapter.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Stateless Redis data access adapter for MUD agent worker.

Pure data access layer with no business logic or state.
All methods take explicit parameters and return data.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, TYPE_CHECKING, Union

from ...config import MUDConfig
from aim_mud_types import RedisKeys, MUDTurnRequest, TurnRequestStatus
from aim_mud_types.helper import _utc_now

if TYPE_CHECKING:
    from ...worker import MUDAgentWorker

logger = logging.getLogger(__name__)


class TurnRequestMixin:
    """Mixin for turn request data access methods."""

    def _turn_request_key(self: "MUDAgentWorker") -> str:
        """Return the Redis key for this agent's turn request."""
        return RedisKeys.agent_turn_request(self.config.agent_id)

    async def _get_turn_request(self: "MUDAgentWorker") -> Optional[MUDTurnRequest]:
        """Fetch the current turn request hash from Redis (private method)."""
        return await self.get_turn_request()

    async def get_turn_request(self: "MUDAgentWorker") -> Optional[MUDTurnRequest]:
        """Fetch the current turn request hash from Redis."""
        return await MUDTurnRequest.from_redis(self.redis, self.config.agent_id)

    async def _set_turn_request_state(
        self: "MUDAgentWorker",
        turn_id: str,
        status: Union[str, TurnRequestStatus],
        message: Optional[str] = None,
        extra_fields: Optional[dict] = None,
        expected_turn_id: Optional[str] = None,
    ) -> bool:
        """Set turn request status with CAS pattern (private method)."""
        return await self.set_turn_request_state(turn_id, status, message, extra_fields, expected_turn_id)

    async def set_turn_request_state(
        self: "MUDAgentWorker",
        turn_id: str,
        status: Union[str, TurnRequestStatus],
        message: Optional[str] = None,
        extra_fields: Optional[dict] = None,
        expected_turn_id: Optional[str] = None,
    ) -> bool:
        """Set turn request status with CAS (Compare-And-Swap) pattern.

        Uses Lua script for atomic update. If expected_turn_id is provided,
        only updates if current turn_id matches (preventing race conditions).

        Args:
            turn_id: Turn ID to set
            status: New status (TurnRequestStatus enum or string)
            message: Optional status message
            extra_fields: Optional additional fields to set
            expected_turn_id: If provided, only update if current turn_id matches (CAS)

        Returns:
            True if update succeeded, False if CAS check failed
        """

        key = RedisKeys.agent_turn_request(self.config.agent_id)

        # Lua script for atomic CAS
        lua_script = """
            local key = KEYS[1]
            local expected_turn_id = ARGV[1]
            local new_turn_id = ARGV[2]

            -- If CAS check requested, verify current turn_id matches
            if expected_turn_id ~= "" then
                local current = redis.call('HGET', key, 'turn_id')
                if current ~= expected_turn_id then
                    return 0  -- CAS failed
                end
            end

            -- Update fields (passed as key-value pairs starting at ARGV[3])
            for i = 3, #ARGV, 2 do
                redis.call('HSET', key, ARGV[i], ARGV[i+1])
            end

            return 1  -- Success
        """

        # Build field updates (convert enum to string value for Redis)
        status_str = status.value if isinstance(status, TurnRequestStatus) else status
        fields = [
            "turn_id", turn_id,
            "status", status_str,
            "heartbeat_at", _utc_now().isoformat()
        ]
        if message:
            fields.extend(["message", message])
        if extra_fields:
            for k, v in extra_fields.items():
                fields.extend([k, str(v)])

        # Execute CAS update
        result = await self.redis.eval(
            lua_script,
            1,  # number of keys
            key,
            expected_turn_id or "",  # Empty string = no CAS check
            turn_id,
            *fields
        )

        if result == 1:
            # Success - conditionally update TTL (0 = no TTL)
            if self.config.turn_request_ttl_seconds > 0:
                await self.redis.expire(key, self.config.turn_request_ttl_seconds)
            return True
        else:
            logger.warning(f"CAS failed: expected turn_id {expected_turn_id}, update aborted")
            return False

    async def atomic_heartbeat_update(
        self: "MUDAgentWorker",
        update_ttl: bool = True,
    ) -> int:
        """Atomically update heartbeat timestamp with validation.

        Uses Lua script to atomically:
        1. Check key exists
        2. Verify required fields (status, turn_id) are present
        3. Update heartbeat_at timestamp
        4. Optionally refresh TTL

        This prevents TOCTOU race conditions where the key could be deleted
        between EXISTS check and HSET write, which would create a partial hash.

        Args:
            update_ttl: If True, refresh TTL after updating heartbeat

        Returns:
            1: Success - heartbeat updated
            0: Key doesn't exist (normal during shutdown)
            -1: Key corrupted (missing required fields)
        """
        key = RedisKeys.agent_turn_request(self.config.agent_id)

        # Lua script for atomic heartbeat update with validation
        lua_script = """
            local key = KEYS[1]
            local heartbeat_at = ARGV[1]
            local ttl = ARGV[2]

            -- Check key exists
            if redis.call('EXISTS', key) == 0 then
                return 0
            end

            -- Verify required fields to detect corruption
            local status = redis.call('HGET', key, 'status')
            local turn_id = redis.call('HGET', key, 'turn_id')

            if not status or not turn_id then
                return -1  -- Corrupted hash
            end

            -- Update heartbeat
            redis.call('HSET', key, 'heartbeat_at', heartbeat_at)

            -- Optional TTL refresh
            if ttl ~= "" then
                redis.call('EXPIRE', key, tonumber(ttl))
            end

            return 1
        """

        # Build arguments
        ttl_arg = str(self.config.turn_request_ttl_seconds) if update_ttl and self.config.turn_request_ttl_seconds > 0 else ""

        # Execute atomic update
        result = await self.redis.eval(
            lua_script,
            1,  # number of keys
            key,
            _utc_now().isoformat(),
            ttl_arg
        )

        return result

    async def _heartbeat_turn_request(self: "MUDAgentWorker", stop_event) -> None:
        """Refresh the turn request TTL while processing a turn.

        Args:
            stop_event: asyncio.Event to signal when to stop heartbeating
        """
        try:
            while not stop_event.is_set():
                await asyncio.sleep(self.config.turn_request_heartbeat_seconds)
                if stop_event.is_set():
                    break

                # Check running flag before updating
                if not self.running:
                    logger.debug("Worker shutting down, stopping heartbeat")
                    return

                # Atomic heartbeat update with validation
                result = await self.atomic_heartbeat_update(update_ttl=True)

                if result == 0:
                    # Key doesn't exist (turn completed or deleted during shutdown)
                    logger.debug("Turn completed or key missing, stopping heartbeat")
                    return
                elif result == -1:
                    # Corrupted hash - should not happen during turn processing
                    logger.error("Heartbeat detected corrupted turn_request hash, stopping")
                    return
                # result == 1: success, continue heartbeating

        except asyncio.CancelledError:
            return

    async def _should_process_turn(
        self: "MUDAgentWorker",
        turn_request: MUDTurnRequest
    ) -> bool:
        """Check if this worker has the oldest active turn."""
        HEARTBEAT_STALE_THRESHOLD = 300  # 5 minutes

        try:
            # Scan for all agent turn_request keys
            agent_keys = []
            cursor = 0
            while True:
                cursor, keys = await self.redis.scan(
                    cursor, match="agent:*:turn_request", count=100
                )
                agent_keys.extend(keys)
                if cursor == 0:
                    break

            if not agent_keys:
                return True

            # Fetch all turn requests
            pipeline = self.redis.pipeline()
            for key in agent_keys:
                pipeline.hgetall(key)
            results = await pipeline.execute()

            # Find oldest ASSIGNED or IN_PROGRESS turn
            oldest_assigned_at = None
            oldest_turn_id = None

            for key, data in zip(agent_keys, results):
                if not data:
                    continue

                # Decode bytes
                decoded = {}
                for k, v in data.items():
                    if isinstance(k, bytes):
                        k = k.decode("utf-8")
                    if isinstance(v, bytes):
                        v = v.decode("utf-8")
                    decoded[str(k)] = str(v)

                # Skip immediate commands - they don't participate in turn guard
                status = decoded.get("status", "")
                if status in [TurnRequestStatus.EXECUTE.value, TurnRequestStatus.EXECUTING.value]:
                    logger.debug(
                        f"Turn guard: Skipping immediate command {decoded.get('turn_id')} "
                        f"(status={status})"
                    )
                    continue

                # Skip non-active turns
                if status not in [TurnRequestStatus.ASSIGNED.value,
                                 TurnRequestStatus.IN_PROGRESS.value]:
                    continue

                # Skip stale heartbeats
                heartbeat_at_str = decoded.get("heartbeat_at")
                if heartbeat_at_str:
                    try:
                        heartbeat_at = datetime.fromisoformat(
                            heartbeat_at_str.replace("Z", "+00:00")
                        )
                        age = (_utc_now() - heartbeat_at).total_seconds()
                        if age > HEARTBEAT_STALE_THRESHOLD:
                            logger.warning(
                                f"Turn guard: Ignoring stale turn "
                                f"{decoded.get('turn_id')} (heartbeat {age:.0f}s old)"
                            )
                            continue
                    except (ValueError, AttributeError):
                        pass

                # Parse assigned_at
                assigned_at_str = decoded.get("assigned_at")
                if not assigned_at_str:
                    continue

                try:
                    assigned_at = datetime.fromisoformat(
                        assigned_at_str.replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    continue

                # Track oldest
                if oldest_assigned_at is None or assigned_at < oldest_assigned_at:
                    oldest_assigned_at = assigned_at
                    oldest_turn_id = decoded.get("turn_id")
                elif assigned_at == oldest_assigned_at:
                    # Tie-breaker: lexicographic turn_id
                    current_oldest_id = oldest_turn_id or ""
                    candidate_id = decoded.get("turn_id", "")
                    if candidate_id < current_oldest_id:
                        oldest_turn_id = candidate_id

            # If no active turns, proceed
            if oldest_turn_id is None:
                return True

            # Check if we're oldest
            is_oldest = (turn_request.turn_id == oldest_turn_id)

            if not is_oldest:
                logger.info(
                    f"Turn guard: Turn {turn_request.turn_id} assigned at "
                    f"{turn_request.assigned_at}, waiting for older turn "
                    f"{oldest_turn_id} (assigned at {oldest_assigned_at})"
                )

            return is_oldest

        except Exception as e:
            logger.error(f"Turn guard: Error checking turn order: {e}", exc_info=True)
            logger.warning("Turn guard: Proceeding despite error to avoid deadlock")
            return True
