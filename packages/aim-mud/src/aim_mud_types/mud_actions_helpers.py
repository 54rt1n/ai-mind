# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Sync helpers for mud:actions stream and processed hash."""

from __future__ import annotations

from typing import Optional

from .redis_keys import RedisKeys


def _decode_stream_id(msg_id: str | bytes) -> str:
    if isinstance(msg_id, bytes):
        return msg_id.decode("utf-8")
    return str(msg_id)


def _mud_actions_stream_key(stream_key: Optional[str] = None) -> str:
    return stream_key or RedisKeys.MUD_ACTIONS


def read_mud_actions(
    redis_client,
    last_id: str,
    *,
    block_ms: int,
    count: int = 100,
    stream_key: Optional[str] = None,
) -> list[tuple[str, dict]]:
    """Read entries from mud:actions using XREAD (sync)."""
    key = _mud_actions_stream_key(stream_key)
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


def append_mud_action(
    redis_client,
    payload: dict,
    *,
    maxlen: Optional[int] = None,
    approximate: bool = True,
    stream_key: Optional[str] = None,
) -> str:
    """Append an entry to mud:actions (sync)."""
    key = _mud_actions_stream_key(stream_key)
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


def range_mud_actions_reverse(
    redis_client,
    *,
    count: int = 100,
    max_id: str = "+",
    min_id: str = "-",
    stream_key: Optional[str] = None,
) -> list[tuple[str, dict]]:
    """Read a reverse range of entries from mud:actions (sync)."""
    key = _mud_actions_stream_key(stream_key)
    result = redis_client.xrevrange(
        key,
        max=max_id,
        min=min_id,
        count=count,
    )
    return [(_decode_stream_id(msg_id), data) for msg_id, data in result]


def get_mud_actions_length(
    redis_client,
    *,
    stream_key: Optional[str] = None,
) -> int:
    """Return mud:actions stream length (sync)."""
    key = _mud_actions_stream_key(stream_key)
    return redis_client.xlen(key)


def trim_mud_actions_maxlen(
    redis_client,
    *,
    maxlen: int,
    approximate: bool = True,
    stream_key: Optional[str] = None,
) -> int:
    """Trim mud:actions by maxlen (sync)."""
    key = _mud_actions_stream_key(stream_key)
    return redis_client.xtrim(
        key,
        maxlen=maxlen,
        approximate=approximate,
    )


def is_mud_action_processed(redis_client, msg_id: str) -> bool:
    """Check if a mud action id exists in processed hash (sync)."""
    return bool(redis_client.hexists(RedisKeys.ACTIONS_PROCESSED, msg_id))


def mark_mud_action_processed(redis_client, msg_id: str, value: str = "1") -> None:
    """Mark a mud action id as processed (sync)."""
    redis_client.hset(RedisKeys.ACTIONS_PROCESSED, msg_id, value)


def get_mud_action_processed_ids(redis_client) -> list[str]:
    """Return processed mud action ids (sync)."""
    keys = redis_client.hkeys(RedisKeys.ACTIONS_PROCESSED)
    ids: list[str] = []
    for key in keys:
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        ids.append(str(key))
    return ids


def get_max_processed_mud_action_id(redis_client) -> Optional[str]:
    """Return maximum processed mud action id (sync)."""
    ids = get_mud_action_processed_ids(redis_client)
    return max(ids) if ids else None


def trim_processed_mud_action_ids(redis_client, keep_count: int) -> int:
    """Trim processed mud action hash to keep most recent N (sync)."""
    ids = get_mud_action_processed_ids(redis_client)
    if len(ids) <= keep_count:
        return 0
    ids.sort()
    to_remove = ids[:-keep_count]
    if not to_remove:
        return 0
    redis_client.hdel(RedisKeys.ACTIONS_PROCESSED, *to_remove)
    return len(to_remove)
