# andimud_worker/datastore/adapter.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Stateless Redis data access adapter for MUD agent worker.

Pure data access layer with no business logic or state.
All methods take explicit parameters and return data.
"""

import asyncio
import logging
import uuid
from typing import Optional, TYPE_CHECKING

from aim_mud_types import RedisKeys, MUDTurnRequest, TurnRequestStatus
from aim_mud_types.helper import _utc_now

from ...exceptions import TurnPreemptedException

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

    async def peek_turn_id(self: "MUDAgentWorker") -> Optional[str]:
        """Fetch the current turn_id without loading full turn_request."""
        key = RedisKeys.agent_turn_request(self.config.agent_id)
        raw = await self.redis.hget(key, "turn_id")
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return str(raw)

    async def ensure_turn_id_current(
        self: "MUDAgentWorker",
        expected_turn_id: str,
    ) -> None:
        """Raise if the current turn_id no longer matches expected."""
        current = await self.peek_turn_id()
        if current != expected_turn_id:
            raise TurnPreemptedException(
                f"Turn preempted (expected {expected_turn_id}, current {current})"
            )

    async def claim_idle_turn(
        self: "MUDAgentWorker",
        turn_request: MUDTurnRequest,
    ) -> str:
        """Claim idle turn by rotating turn_id with CAS."""
        old_turn_id = turn_request.turn_id
        new_turn_id = str(uuid.uuid4())
        turn_request.turn_id = new_turn_id
        turn_request.heartbeat_at = _utc_now()
        success = await self.update_turn_request(turn_request, expected_turn_id=old_turn_id)
        if not success:
            turn_request.turn_id = old_turn_id
            raise TurnPreemptedException(
                f"Turn preempted during claim (expected {old_turn_id})"
            )
        return new_turn_id

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
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)
        return await client.create_turn_request(self.config.agent_id, turn_request)

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
        from aim_mud_types.turn_request_helpers import update_turn_request_async

        success = await update_turn_request_async(
            self.redis,
            self.config.agent_id,
            turn_request,
            expected_turn_id=expected_turn_id,
        )
        if not success:
            logger.warning(f"CAS failed: expected turn_id {expected_turn_id}, update aborted")
        return success

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
        from aim_mud_types.turn_request_helpers import atomic_heartbeat_update_async

        return await atomic_heartbeat_update_async(
            self.redis,
            self.config.agent_id,
        )

    async def _heartbeat_turn_request(self: "MUDAgentWorker", stop_event) -> None:
        """Refresh the turn request heartbeat while processing a turn.

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
            from aim_mud_types.client import RedisMUDClient
            client = RedisMUDClient(self.redis)
            oldest = await client.get_oldest_active_turn(
                heartbeat_stale_threshold=HEARTBEAT_STALE_THRESHOLD,
            )
            if oldest is None:
                return True

            _, oldest_turn_id, oldest_assigned_at = oldest

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
