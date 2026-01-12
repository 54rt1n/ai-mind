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
from aim_mud_types import RedisKeys, MUDTurnRequest, TurnRequestStatus, TurnReason
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
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)
        return await client.get_turn_request(self.config.agent_id)

    async def create_turn_request(
        self: "MUDAgentWorker",
        turn_request: MUDTurnRequest
    ) -> bool:
        """Create a new turn_request from a MUDTurnRequest object.

        Args:
            turn_request: Complete MUDTurnRequest object with all required fields

        Returns:
            True if created, False if turn_request already exists
        """
        key = RedisKeys.agent_turn_request(self.config.agent_id)

        # Lua script to create only if key doesn't exist
        lua_script = """
            local key = KEYS[1]

            -- Fail if key already exists
            if redis.call('EXISTS', key) == 1 then
                return 0
            end

            -- Create with all fields (passed as key-value pairs)
            for i = 1, #ARGV, 2 do
                redis.call('HSET', key, ARGV[i], ARGV[i+1])
            end

            return 1
        """

        # Convert MUDTurnRequest to field list
        import json
        fields = []
        for field_name, field_value in turn_request.model_dump().items():
            if field_value is not None:
                # Convert to string for Redis
                if isinstance(field_value, datetime):
                    value_str = field_value.isoformat()
                elif isinstance(field_value, (TurnRequestStatus, TurnReason)):
                    value_str = field_value.value
                elif isinstance(field_value, dict):
                    value_str = json.dumps(field_value)
                else:
                    value_str = str(field_value)
                fields.extend([field_name, value_str])

        result = await self.redis.eval(lua_script, 1, key, *fields)

        return result == 1

    async def update_turn_request(
        self: "MUDAgentWorker",
        turn_request: MUDTurnRequest,
        expected_turn_id: str
    ) -> bool:
        """Update an existing turn_request with CAS.

        Args:
            turn_request: Complete MUDTurnRequest object with updated fields
            expected_turn_id: The turn_id that must match for update to succeed (CAS)

        Returns:
            True if updated, False if CAS failed
        """
        key = RedisKeys.agent_turn_request(self.config.agent_id)

        # Lua script for atomic CAS update
        lua_script = """
            local key = KEYS[1]
            local expected_turn_id = ARGV[1]

            -- Verify turn_id matches (CAS check)
            local current = redis.call('HGET', key, 'turn_id')
            if current ~= expected_turn_id then
                return 0  -- CAS failed
            end

            -- Update fields (passed as key-value pairs starting at ARGV[2])
            for i = 2, #ARGV, 2 do
                redis.call('HSET', key, ARGV[i], ARGV[i+1])
            end

            return 1  -- Success
        """

        # Convert MUDTurnRequest to field list
        import json
        fields = []
        for field_name, field_value in turn_request.model_dump().items():
            if field_value is not None:
                # Convert to string for Redis
                if isinstance(field_value, datetime):
                    value_str = field_value.isoformat()
                elif isinstance(field_value, (TurnRequestStatus, TurnReason)):
                    value_str = field_value.value
                elif isinstance(field_value, dict):
                    value_str = json.dumps(field_value)
                else:
                    value_str = str(field_value)
                fields.extend([field_name, value_str])

        # Execute CAS update
        result = await self.redis.eval(
            lua_script,
            1,  # number of keys
            key,
            expected_turn_id,
            *fields
        )

        if result == 1:
            return True
        else:
            logger.warning(f"CAS failed: expected turn_id {expected_turn_id}, update aborted")
            return False

    async def atomic_heartbeat_update(
        self: "MUDAgentWorker",
    ) -> int:
        """Atomically update heartbeat timestamp with validation.

        Uses Lua script to atomically:
        1. Check key exists
        2. Verify required fields (status, turn_id) are present
        3. Update heartbeat_at timestamp

        This prevents TOCTOU race conditions where the key could be deleted
        between EXISTS check and HSET write, which would create a partial hash.

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

            return 1
        """

        # Execute atomic update
        result = await self.redis.eval(
            lua_script,
            1,  # number of keys
            key,
            _utc_now().isoformat()
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
                result = await self.atomic_heartbeat_update()

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
