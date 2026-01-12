# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Sync helpers for mud:events stream and processed hash."""

from __future__ import annotations

from typing import Optional

from .helper import _utc_now
from .redis_keys import RedisKeys


def _decode_stream_id(msg_id: str | bytes) -> str:
    if isinstance(msg_id, bytes):
        return msg_id.decode("utf-8")
    return str(msg_id)


def _mud_events_stream_key(stream_key: Optional[str] = None) -> str:
    return stream_key or RedisKeys.MUD_EVENTS


def get_mud_events_last_id(redis_client, *, stream_key: Optional[str] = None) -> Optional[str]:
    """Return last-generated id for mud:events, or None if empty."""
    key = _mud_events_stream_key(stream_key)
    info = redis_client.xinfo_stream(key)
    last_id = info.get("last-generated-id") or info.get(b"last-generated-id")
    if isinstance(last_id, bytes):
        last_id = last_id.decode("utf-8")
    if not last_id or last_id in ("0", "0-0"):
        return None
    return str(last_id)


def read_mud_events(
    redis_client,
    last_id: str,
    *,
    block_ms: int,
    count: int = 100,
    stream_key: Optional[str] = None,
) -> list[tuple[str, dict]]:
    """Read entries from mud:events using XREAD (sync)."""
    key = _mud_events_stream_key(stream_key)
    result = redis_client.xread(
        {key: last_id},
        block=block_ms,
        count=count,
    )
    if not result:
        return []
    for _stream_name, messages in result:
        return [(_decode_stream_id(msg_id), data) for msg_id, data in messages]
    return []


def append_mud_event(
    redis_client,
    payload: dict,
    *,
    maxlen: Optional[int] = None,
    approximate: bool = True,
    stream_key: Optional[str] = None,
) -> str:
    """Append an entry to mud:events (sync)."""
    key = _mud_events_stream_key(stream_key)
    if maxlen is None:
        msg_id = redis_client.xadd(key, payload)
    else:
        msg_id = redis_client.xadd(
            key,
            payload,
            maxlen=maxlen,
            approximate=approximate,
        )
    return _decode_stream_id(msg_id)


def trim_mud_events_minid(
    redis_client,
    *,
    min_id: str,
    approximate: bool = True,
    stream_key: Optional[str] = None,
) -> int:
    """Trim mud:events by min id (sync)."""
    key = _mud_events_stream_key(stream_key)
    return redis_client.xtrim(
        key,
        minid=min_id,
        approximate=approximate,
    )


def is_mud_event_processed(redis_client, msg_id: str) -> bool:
    """Check if a mud event id exists in processed hash (sync)."""
    return bool(redis_client.hexists(RedisKeys.EVENTS_PROCESSED, msg_id))


def mark_mud_event_processed(redis_client, msg_id: str, agents: list[str]) -> None:
    """Mark a mud event as processed with timestamp + agent list (sync)."""
    timestamp = _utc_now().isoformat()
    agent_list = ",".join(agents) if agents else ""
    value = f"{timestamp}|{agent_list}"
    redis_client.hset(RedisKeys.EVENTS_PROCESSED, msg_id, value)


def get_mud_event_processed_ids(redis_client) -> list[str]:
    """Return processed mud event ids (sync)."""
    keys = redis_client.hkeys(RedisKeys.EVENTS_PROCESSED)
    ids: list[str] = []
    for key in keys:
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        ids.append(str(key))
    return ids


def get_min_processed_mud_event_id(redis_client) -> Optional[str]:
    """Return minimum processed mud event id (sync)."""
    ids = get_mud_event_processed_ids(redis_client)
    return min(ids) if ids else None


def get_max_processed_mud_event_id(redis_client) -> Optional[str]:
    """Return maximum processed mud event id (sync)."""
    ids = get_mud_event_processed_ids(redis_client)
    return max(ids) if ids else None


def trim_processed_mud_event_ids(redis_client, keep_count: int) -> int:
    """Trim processed mud event hash to keep most recent N (sync)."""
    ids = get_mud_event_processed_ids(redis_client)
    if len(ids) <= keep_count:
        return 0
    ids.sort()
    to_remove = ids[:-keep_count]
    if not to_remove:
        return 0
    redis_client.hdel(RedisKeys.EVENTS_PROCESSED, *to_remove)
    return len(to_remove)
