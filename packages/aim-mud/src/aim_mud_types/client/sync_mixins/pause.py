# aim-mud-types/client/sync_mixins/pause.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Sync pause flag operations."""

from typing import TYPE_CHECKING

from ...redis_keys import RedisKeys

if TYPE_CHECKING:
    from ..base import BaseSyncRedisMUDClient


def _is_pause_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value == "1"


class SyncPauseMixin:
    """Sync pause flag Redis operations."""

    def is_paused(self: "BaseSyncRedisMUDClient", pause_key: str) -> bool:
        """Return True if pause flag is set for key."""
        value = self.redis.get(pause_key)
        return _is_pause_value(value)

    def is_agent_paused(self: "BaseSyncRedisMUDClient", agent_id: str) -> bool:
        """Return True if agent pause flag is set."""
        return self.is_paused(RedisKeys.agent_pause(agent_id))

    def is_mediator_paused(self: "BaseSyncRedisMUDClient") -> bool:
        """Return True if mediator pause flag is set."""
        return self.is_paused(RedisKeys.MEDIATOR_PAUSE)
