# andimud_worker/datastore/adapter.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Stateless Redis data access adapter for MUD agent worker.

Pure data access layer with no business logic or state.
All methods take explicit parameters and return data.
"""

import asyncio
import logging
from typing import Optional, TYPE_CHECKING

from ...config import MUDConfig
from aim_mud_types import RedisKeys
from aim_mud_types.helper import _utc_now

if TYPE_CHECKING:
    from ...worker import MUDAgentWorker

logger = logging.getLogger(__name__)


class TurnRequestMixin:
    """Mixin for turn request data access methods."""

    def _turn_request_key(self: "MUDAgentWorker") -> str:
        """Return the Redis key for this agent's turn request."""
        return RedisKeys.agent_turn_request(self.config.agent_id)

    async def _get_turn_request(self: "MUDAgentWorker") -> dict[str, str]:
        """Fetch the current turn request hash from Redis (private method)."""
        return await self.get_turn_request()

    async def get_turn_request(self: "MUDAgentWorker") -> dict[str, str]:
        """Fetch the current turn request hash from Redis."""
        key = RedisKeys.agent_turn_request(self.config.agent_id)
        data = await self.redis.hgetall(key)

        if not data:
            return {}

        # Decode bytes to strings
        result: dict[str, str] = {}
        for k, v in data.items():
            if isinstance(k, bytes):
                k = k.decode("utf-8")
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            result[str(k)] = str(v)

        return result

    async def _set_turn_request_state(
        self: "MUDAgentWorker",
        turn_id: str,
        status: str,
        message: Optional[str] = None,
        extra_fields: Optional[dict] = None,
        expected_turn_id: Optional[str] = None,
    ) -> bool:
        """Set turn request status with CAS pattern (private method)."""
        return await self.set_turn_request_state(turn_id, status, message, extra_fields, expected_turn_id)

    async def set_turn_request_state(
        self: "MUDAgentWorker",
        turn_id: str,
        status: str,
        message: Optional[str] = None,
        extra_fields: Optional[dict] = None,
        expected_turn_id: Optional[str] = None,
    ) -> bool:
        """Set turn request status with CAS (Compare-And-Swap) pattern.

        Uses Lua script for atomic update. If expected_turn_id is provided,
        only updates if current turn_id matches (preventing race conditions).

        Args:
            turn_id: Turn ID to set
            status: New status ("assigned", "in_progress", "done", "fail")
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

        # Build field updates
        fields = [
            "turn_id", turn_id,
            "status", status,
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
                # Only set TTL if configured (0 = no TTL)
                if self.config.turn_request_ttl_seconds > 0:
                    await self.redis.expire(
                        self._turn_request_key(),
                        self.config.turn_request_ttl_seconds,
                    )
                await self.redis.hset(
                    self._turn_request_key(),
                    mapping={"heartbeat_at": _utc_now().isoformat()},
                )
        except asyncio.CancelledError:
            return
