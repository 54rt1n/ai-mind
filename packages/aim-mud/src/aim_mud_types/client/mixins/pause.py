# aim-mud-types/client/mixins/pause.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Pause flag operations."""

from typing import TYPE_CHECKING

from ...redis_keys import RedisKeys

if TYPE_CHECKING:
    from .. import BaseRedisMUDClient


def _is_pause_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value == "1"


class PauseMixin:
    """Pause flag Redis operations."""

    async def is_paused(self: "BaseRedisMUDClient", pause_key: str) -> bool:
        """Return True if pause flag is set for key."""
        value = await self.redis.get(pause_key)
        return _is_pause_value(value)

    async def is_agent_paused(self: "BaseRedisMUDClient", agent_id: str) -> bool:
        """Return True if agent pause flag is set."""
        return await self.is_paused(RedisKeys.agent_pause(agent_id))

    async def is_mediator_paused(self: "BaseRedisMUDClient") -> bool:
        """Return True if mediator pause flag is set."""
        return await self.is_paused(RedisKeys.MEDIATOR_PAUSE)
