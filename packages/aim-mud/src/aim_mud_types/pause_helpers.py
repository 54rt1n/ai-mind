# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Helpers for pause flag access."""

from __future__ import annotations

from .redis_keys import RedisKeys


def _is_pause_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value == "1"


def is_paused(redis_client, pause_key: str) -> bool:
    """Return True if pause flag is set (sync)."""
    value = redis_client.get(pause_key)
    return _is_pause_value(value)


async def is_paused_async(redis_client, pause_key: str) -> bool:
    """Return True if pause flag is set (async)."""
    value = await redis_client.get(pause_key)
    return _is_pause_value(value)


def is_agent_paused(redis_client, agent_id: str) -> bool:
    """Return True if agent pause flag is set (sync)."""
    return is_paused(redis_client, RedisKeys.agent_pause(agent_id))


async def is_agent_paused_async(redis_client, agent_id: str) -> bool:
    """Return True if agent pause flag is set (async)."""
    return await is_paused_async(redis_client, RedisKeys.agent_pause(agent_id))


def is_mediator_paused(redis_client) -> bool:
    """Return True if mediator pause flag is set (sync)."""
    return is_paused(redis_client, RedisKeys.MEDIATOR_PAUSE)


async def is_mediator_paused_async(redis_client) -> bool:
    """Return True if mediator pause flag is set (async)."""
    return await is_paused_async(redis_client, RedisKeys.MEDIATOR_PAUSE)
