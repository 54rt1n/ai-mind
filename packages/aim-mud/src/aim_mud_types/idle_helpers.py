# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Helpers for idle active flag access."""

from __future__ import annotations

from .redis_keys import RedisKeys


def _is_idle_active_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def is_idle_active(redis_client, agent_id: str) -> bool:
    """Return True if idle active flag is set (sync)."""
    key = RedisKeys.agent_idle_active(agent_id)
    value = redis_client.get(key)
    return _is_idle_active_value(value)


async def is_idle_active_async(redis_client, agent_id: str) -> bool:
    """Return True if idle active flag is set (async)."""
    key = RedisKeys.agent_idle_active(agent_id)
    value = await redis_client.get(key)
    return _is_idle_active_value(value)


def set_idle_active(redis_client, agent_id: str, enabled: bool) -> None:
    """Set idle active flag (sync)."""
    key = RedisKeys.agent_idle_active(agent_id)
    redis_client.set(key, "true" if enabled else "false")


async def set_idle_active_async(redis_client, agent_id: str, enabled: bool) -> None:
    """Set idle active flag (async)."""
    key = RedisKeys.agent_idle_active(agent_id)
    await redis_client.set(key, "true" if enabled else "false")
