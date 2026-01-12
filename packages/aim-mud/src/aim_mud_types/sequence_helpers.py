# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Helpers for sequence counter access."""

from __future__ import annotations

from .redis_keys import RedisKeys


def next_sequence_id(redis_client) -> int:
    """Return next sequence id (sync)."""
    return int(redis_client.incr(RedisKeys.SEQUENCE_COUNTER))


async def next_sequence_id_async(redis_client) -> int:
    """Return next sequence id (async)."""
    return int(await redis_client.incr(RedisKeys.SEQUENCE_COUNTER))
