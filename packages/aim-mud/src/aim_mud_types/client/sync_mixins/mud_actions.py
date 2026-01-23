# aim-mud-types/client/sync_mixins/mud_actions.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Sync helpers for mud:actions stream and processed hash."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from ...redis_keys import RedisKeys

if TYPE_CHECKING:
    from ..base import BaseSyncRedisMUDClient


class SyncMudActionsStreamMixin:
    """Sync Redis helpers for mud:actions stream and processed hash."""

    def _decode_stream_id(self, msg_id: str | bytes) -> str:
        if isinstance(msg_id, bytes):
            return msg_id.decode("utf-8")
        return str(msg_id)

    def _mud_actions_stream_key(self, stream_key: Optional[str] = None) -> str:
        return stream_key or RedisKeys.MUD_ACTIONS

    def read_mud_actions(
        self: "BaseSyncRedisMUDClient",
        last_id: str,
        *,
        block_ms: int,
        count: int = 100,
        stream_key: Optional[str] = None,
    ) -> list[tuple[str, dict]]:
        """Read entries from mud:actions using XREAD."""
        key = self._mud_actions_stream_key(stream_key)
        result = self.redis.xread(
            {key: last_id},
            block=block_ms,
            count=count,
        )
        if not result:
            return []
        for _stream_name, messages in result:
            return [(self._decode_stream_id(msg_id), data) for msg_id, data in messages]
        return []

    def append_mud_action(
        self: "BaseSyncRedisMUDClient",
        payload: dict,
        *,
        maxlen: Optional[int] = None,
        approximate: bool = True,
        stream_key: Optional[str] = None,
    ) -> str:
        """Append an entry to mud:actions."""
        key = self._mud_actions_stream_key(stream_key)
        if maxlen is None:
            msg_id = self.redis.xadd(key, payload)
        else:
            msg_id = self.redis.xadd(
                key,
                payload,
                maxlen=maxlen,
                approximate=approximate,
            )
        return self._decode_stream_id(msg_id)

    def range_mud_actions_reverse(
        self: "BaseSyncRedisMUDClient",
        *,
        count: int = 100,
        max_id: str = "+",
        min_id: str = "-",
        stream_key: Optional[str] = None,
    ) -> list[tuple[str, dict]]:
        """Read a reverse range of entries from mud:actions."""
        key = self._mud_actions_stream_key(stream_key)
        result = self.redis.xrevrange(
            key,
            max=max_id,
            min=min_id,
            count=count,
        )
        return [(self._decode_stream_id(msg_id), data) for msg_id, data in result]

    def get_mud_actions_length(
        self: "BaseSyncRedisMUDClient",
        *,
        stream_key: Optional[str] = None,
    ) -> int:
        """Return mud:actions stream length."""
        key = self._mud_actions_stream_key(stream_key)
        return self.redis.xlen(key)

    def trim_mud_actions_maxlen(
        self: "BaseSyncRedisMUDClient",
        *,
        maxlen: int,
        approximate: bool = True,
        stream_key: Optional[str] = None,
    ) -> int:
        """Trim mud:actions by maxlen."""
        key = self._mud_actions_stream_key(stream_key)
        return self.redis.xtrim(
            key,
            maxlen=maxlen,
            approximate=approximate,
        )

    def is_mud_action_processed(
        self: "BaseSyncRedisMUDClient",
        msg_id: str,
    ) -> bool:
        """Check if a mud action id exists in processed hash."""
        return self.redis.hexists(RedisKeys.ACTIONS_PROCESSED, msg_id)

    def mark_mud_action_processed(
        self: "BaseSyncRedisMUDClient",
        msg_id: str,
        value: str = "1",
    ) -> None:
        """Mark a mud action id as processed."""
        self.redis.hset(RedisKeys.ACTIONS_PROCESSED, msg_id, value)

    def get_mud_action_processed_ids(
        self: "BaseSyncRedisMUDClient",
    ) -> list[str]:
        """Return processed mud action ids."""
        keys = self.redis.hkeys(RedisKeys.ACTIONS_PROCESSED)
        ids: list[str] = []
        for key in keys:
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            ids.append(str(key))
        return ids

    def get_max_processed_mud_action_id(
        self: "BaseSyncRedisMUDClient",
    ) -> Optional[str]:
        """Return maximum processed mud action id."""
        ids = self.get_mud_action_processed_ids()
        return max(ids) if ids else None

    def trim_processed_mud_action_ids(
        self: "BaseSyncRedisMUDClient",
        keep_count: int,
    ) -> int:
        """Trim processed mud action hash to keep most recent N."""
        ids = self.get_mud_action_processed_ids()
        if len(ids) <= keep_count:
            return 0
        ids.sort()
        to_remove = ids[:-keep_count]
        if not to_remove:
            return 0
        self.redis.hdel(RedisKeys.ACTIONS_PROCESSED, *to_remove)
        return len(to_remove)
