# aim-mud-types/client/sync_mixins/idle.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Sync idle active flag operations."""

from typing import TYPE_CHECKING

from ...redis_keys import RedisKeys

if TYPE_CHECKING:
    from ..base import BaseSyncRedisMUDClient


def _is_idle_active_value(value) -> bool:
    """Parse idle active flag from Redis value.

    Accepts truthy string values: "1", "true", "yes", "on" (case-insensitive).

    Args:
        value: Raw Redis value (bytes, str, or None)

    Returns:
        True if value represents an active idle flag
    """
    if value is None:
        return False
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class SyncIdleMixin:
    """Sync idle active flag Redis operations.

    The idle active flag controls whether agents can take actions
    during idle turns (when awake but no events pending).

    When set to True, idle turns can result in actions.
    When set to False, idle turns only generate thoughts.
    """

    def is_idle_active(self: "BaseSyncRedisMUDClient", agent_id: str) -> bool:
        """Check if idle active flag is set for agent.

        Args:
            agent_id: Agent identifier

        Returns:
            True if idle active flag is set, False otherwise
        """
        key = RedisKeys.agent_idle_active(agent_id)
        value = self.redis.get(key)
        return _is_idle_active_value(value)

    def set_idle_active(
        self: "BaseSyncRedisMUDClient",
        agent_id: str,
        enabled: bool
    ) -> None:
        """Set idle active flag for agent.

        Args:
            agent_id: Agent identifier
            enabled: True to enable idle actions, False to disable
        """
        key = RedisKeys.agent_idle_active(agent_id)
        self.redis.set(key, "true" if enabled else "false")
