# aim-mud-types/client/mixins/mud_events.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Async helpers for mud:events stream and processed hash."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from ...helper import _utc_now
from ...redis_keys import RedisKeys

if TYPE_CHECKING:
    from ..base import BaseAsyncRedisMUDClient


class MudEventsStreamMixin:
    """Redis helpers for mud:events and its processed hash."""

    def _decode_stream_id(self, msg_id: str | bytes) -> str:
        if isinstance(msg_id, bytes):
            return msg_id.decode("utf-8")
        return str(msg_id)

    def _mud_events_stream_key(self, stream_key: Optional[str] = None) -> str:
        return stream_key or RedisKeys.MUD_EVENTS

    async def get_mud_events_last_id(
        self: "BaseAsyncRedisMUDClient",
        *,
        stream_key: Optional[str] = None,
    ) -> Optional[str]:
        """Return last-generated id for mud:events, or None if empty."""
        key = self._mud_events_stream_key(stream_key)
        info = await self.redis.xinfo_stream(key)
        last_id = info.get("last-generated-id") or info.get(b"last-generated-id")
        if isinstance(last_id, bytes):
            last_id = last_id.decode("utf-8")
        if not last_id or last_id in ("0", "0-0"):
            return None
        return str(last_id)

    async def read_mud_events(
        self: "BaseAsyncRedisMUDClient",
        last_id: str,
        *,
        block_ms: int,
        count: int = 100,
        stream_key: Optional[str] = None,
    ) -> list[tuple[str, dict]]:
        """Read entries from mud:events using XREAD."""
        key = self._mud_events_stream_key(stream_key)
        result = await self.redis.xread(
            {key: last_id},
            block=block_ms,
            count=count,
        )
        if not result:
            return []
        for _stream_name, messages in result:
            return [(self._decode_stream_id(msg_id), data) for msg_id, data in messages]
        return []

    async def append_mud_event(
        self: "BaseAsyncRedisMUDClient",
        payload: dict,
        *,
        maxlen: Optional[int] = None,
        approximate: bool = True,
        stream_key: Optional[str] = None,
    ) -> str:
        """Append an entry to mud:events."""
        key = self._mud_events_stream_key(stream_key)
        if maxlen is None:
            msg_id = await self.redis.xadd(key, payload)
        else:
            msg_id = await self.redis.xadd(
                key,
                payload,
                maxlen=maxlen,
                approximate=approximate,
            )
        return self._decode_stream_id(msg_id)

    async def trim_mud_events_minid(
        self: "BaseAsyncRedisMUDClient",
        *,
        min_id: str,
        approximate: bool = True,
        stream_key: Optional[str] = None,
    ) -> int:
        """Trim mud:events by min id."""
        key = self._mud_events_stream_key(stream_key)
        return await self.redis.xtrim(
            key,
            minid=min_id,
            approximate=approximate,
        )

    async def is_mud_event_processed(
        self: "BaseAsyncRedisMUDClient",
        msg_id: str,
    ) -> bool:
        """Check if a mud event id exists in processed hash."""
        return await self.redis.hexists(RedisKeys.EVENTS_PROCESSED, msg_id)

    async def mark_mud_event_processed(
        self: "BaseAsyncRedisMUDClient",
        msg_id: str,
        agents: list[str],
    ) -> None:
        """Mark a mud event as processed with timestamp + agent list."""
        timestamp = _utc_now().isoformat()
        agent_list = ",".join(agents) if agents else ""
        value = f"{timestamp}|{agent_list}"
        await self.redis.hset(RedisKeys.EVENTS_PROCESSED, msg_id, value)

    async def get_mud_event_processed_ids(
        self: "BaseAsyncRedisMUDClient",
    ) -> list[str]:
        """Return processed mud event ids."""
        keys = await self.redis.hkeys(RedisKeys.EVENTS_PROCESSED)
        ids: list[str] = []
        for key in keys:
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            ids.append(str(key))
        return ids

    async def get_min_processed_mud_event_id(
        self: "BaseAsyncRedisMUDClient",
    ) -> Optional[str]:
        """Return minimum processed mud event id."""
        ids = await self.get_mud_event_processed_ids()
        return min(ids) if ids else None

    async def get_max_processed_mud_event_id(
        self: "BaseAsyncRedisMUDClient",
    ) -> Optional[str]:
        """Return maximum processed mud event id."""
        ids = await self.get_mud_event_processed_ids()
        return max(ids) if ids else None

    async def trim_processed_mud_event_ids(
        self: "BaseAsyncRedisMUDClient",
        keep_count: int,
    ) -> int:
        """Trim processed mud event hash to keep most recent N."""
        ids = await self.get_mud_event_processed_ids()
        if len(ids) <= keep_count:
            return 0
        ids.sort()
        to_remove = ids[:-keep_count]
        if not to_remove:
            return 0
        await self.redis.hdel(RedisKeys.EVENTS_PROCESSED, *to_remove)
        return len(to_remove)
