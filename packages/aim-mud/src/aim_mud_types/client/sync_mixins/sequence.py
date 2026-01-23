# aim-mud-types/client/sync_mixins/sequence.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Sync sequence counter operations."""

from typing import TYPE_CHECKING

from ...redis_keys import RedisKeys

if TYPE_CHECKING:
    from ..base import BaseSyncRedisMUDClient


class SyncSequenceMixin:
    """Sync sequence counter Redis operations."""

    def next_sequence_id(self: "BaseSyncRedisMUDClient") -> int:
        """Return next sequence id."""
        value = self.redis.incr(RedisKeys.SEQUENCE_COUNTER)
        return int(value)
