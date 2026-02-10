# aim-mud-types/client/sync_mixins/mud_events.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Sync helpers for mud:events stream and processed hash."""

from __future__ import annotations

import json
from typing import Any, Optional, TYPE_CHECKING

from ...helper import _utc_now
from ...models import ActorType, EventType, MUDEvent
from ...redis_keys import RedisKeys

if TYPE_CHECKING:
    from ..base import BaseSyncRedisMUDClient


class SyncMudEventsStreamMixin:
    """Sync Redis helpers for mud:events and its processed hash."""

    def _decode_stream_id(self, msg_id: str | bytes) -> str:
        if isinstance(msg_id, bytes):
            return msg_id.decode("utf-8")
        return str(msg_id)

    def _mud_events_stream_key(self, stream_key: Optional[str] = None) -> str:
        return stream_key or RedisKeys.MUD_EVENTS

    def get_mud_events_last_id(
        self: "BaseSyncRedisMUDClient",
        *,
        stream_key: Optional[str] = None,
    ) -> Optional[str]:
        """Return last-generated id for mud:events, or None if empty."""
        key = self._mud_events_stream_key(stream_key)
        info = self.redis.xinfo_stream(key)
        last_id = info.get("last-generated-id") or info.get(b"last-generated-id")
        if isinstance(last_id, bytes):
            last_id = last_id.decode("utf-8")
        if not last_id or last_id in ("0", "0-0"):
            return None
        return str(last_id)

    def read_mud_events(
        self: "BaseSyncRedisMUDClient",
        last_id: str,
        *,
        block_ms: int,
        count: int = 100,
        stream_key: Optional[str] = None,
    ) -> list[tuple[str, dict]]:
        """Read entries from mud:events using XREAD."""
        key = self._mud_events_stream_key(stream_key)
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

    def append_mud_event(
        self: "BaseSyncRedisMUDClient",
        payload: dict,
        *,
        maxlen: Optional[int] = None,
        approximate: bool = True,
        stream_key: Optional[str] = None,
    ) -> str:
        """Append an entry to mud:events."""
        key = self._mud_events_stream_key(stream_key)
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

    def publish_mud_event(
        self: "BaseSyncRedisMUDClient",
        *,
        event_type: EventType | str,
        actor: str,
        actor_id: str = "",
        actor_type: ActorType | str = ActorType.PLAYER,
        room_id: str = "",
        room_name: str = "",
        content: str = "",
        target: Optional[str] = None,
        target_id: str = "",
        metadata: Optional[dict[str, Any]] = None,
        timestamp: Optional[Any] = None,
        sequence_id: Optional[int] = None,
        action_id: Optional[str] = None,
        maxlen: Optional[int] = None,
        approximate: bool = True,
        stream_key: Optional[str] = None,
    ) -> str:
        """Build a MUDEvent and append it to mud:events.

        If metadata includes an "action_id", it is promoted to the top-level
        event field for correlation. An explicit action_id parameter overrides
        any metadata value.
        """
        meta = dict(metadata) if isinstance(metadata, dict) else {}
        meta_action_id = meta.pop("action_id", None)
        if action_id is None:
            action_id = meta_action_id

        event = MUDEvent(
            event_type=event_type,
            actor=actor,
            actor_id=actor_id,
            action_id=action_id,
            actor_type=actor_type,
            room_id=room_id or "",
            room_name=room_name or "",
            content=content or "",
            target=target,
            target_id=target_id or "",
            timestamp=timestamp if timestamp is not None else _utc_now(),
            metadata=meta,
            sequence_id=sequence_id,
        )

        payload = event.to_redis_dict()
        return self.append_mud_event(
            {"data": json.dumps(payload)},
            maxlen=maxlen,
            approximate=approximate,
            stream_key=stream_key,
        )

    def trim_mud_events_minid(
        self: "BaseSyncRedisMUDClient",
        *,
        min_id: str,
        approximate: bool = True,
        stream_key: Optional[str] = None,
    ) -> int:
        """Trim mud:events by min id."""
        key = self._mud_events_stream_key(stream_key)
        return self.redis.xtrim(
            key,
            minid=min_id,
            approximate=approximate,
        )

    def is_mud_event_processed(
        self: "BaseSyncRedisMUDClient",
        msg_id: str,
    ) -> bool:
        """Check if a mud event id exists in processed hash."""
        return self.redis.hexists(RedisKeys.EVENTS_PROCESSED, msg_id)

    def mark_mud_event_processed(
        self: "BaseSyncRedisMUDClient",
        msg_id: str,
        agents: list[str],
    ) -> None:
        """Mark a mud event as processed with timestamp + agent list."""
        timestamp = _utc_now().isoformat()
        agent_list = ",".join(agents) if agents else ""
        value = f"{timestamp}|{agent_list}"
        self.redis.hset(RedisKeys.EVENTS_PROCESSED, msg_id, value)

    def get_mud_event_processed_ids(
        self: "BaseSyncRedisMUDClient",
    ) -> list[str]:
        """Return processed mud event ids."""
        keys = self.redis.hkeys(RedisKeys.EVENTS_PROCESSED)
        ids: list[str] = []
        for key in keys:
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            ids.append(str(key))
        return ids

    def get_min_processed_mud_event_id(
        self: "BaseSyncRedisMUDClient",
    ) -> Optional[str]:
        """Return minimum processed mud event id."""
        ids = self.get_mud_event_processed_ids()
        return min(ids) if ids else None

    def get_max_processed_mud_event_id(
        self: "BaseSyncRedisMUDClient",
    ) -> Optional[str]:
        """Return maximum processed mud event id."""
        ids = self.get_mud_event_processed_ids()
        return max(ids) if ids else None

    def trim_processed_mud_event_ids(
        self: "BaseSyncRedisMUDClient",
        keep_count: int,
    ) -> int:
        """Trim processed mud event hash to keep most recent N."""
        ids = self.get_mud_event_processed_ids()
        if len(ids) <= keep_count:
            return 0
        ids.sort()
        to_remove = ids[:-keep_count]
        if not to_remove:
            return 0
        self.redis.hdel(RedisKeys.EVENTS_PROCESSED, *to_remove)
        return len(to_remove)
