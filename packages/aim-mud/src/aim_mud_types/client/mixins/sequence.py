# aim-mud-types/client/mixins/sequence.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Sequence counter operations."""

from typing import TYPE_CHECKING

from ...redis_keys import RedisKeys

if TYPE_CHECKING:
    from .. import BaseRedisMUDClient


class SequenceMixin:
    """Sequence counter Redis operations."""

    async def next_sequence_id(self: "BaseRedisMUDClient") -> int:
        """Return next sequence id."""
        value = await self.redis.incr(RedisKeys.SEQUENCE_COUNTER)
        return int(value)
