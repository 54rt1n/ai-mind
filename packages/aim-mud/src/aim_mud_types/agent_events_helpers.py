# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Sync helpers for agent:{id}:events streams."""

from __future__ import annotations

from typing import Optional

from .redis_keys import RedisKeys


def _decode_stream_id(msg_id: str | bytes) -> str:
    if isinstance(msg_id, bytes):
        return msg_id.decode("utf-8")
    return str(msg_id)


def _agent_events_stream_key(agent_id: str, stream_key: Optional[str] = None) -> str:
    return stream_key or RedisKeys.agent_events(agent_id)


def get_agent_events_last_id(
    redis_client,
    agent_id: str,
    *,
    stream_key: Optional[str] = None,
) -> Optional[str]:
    """Return last-generated id for agent events stream, or None if empty."""
    key = _agent_events_stream_key(agent_id, stream_key)
    info = redis_client.xinfo_stream(key)
    last_id = info.get("last-generated-id") or info.get(b"last-generated-id")
    if isinstance(last_id, bytes):
        last_id = last_id.decode("utf-8")
    if not last_id or last_id in ("0", "0-0"):
        return None
    return str(last_id)


def range_agent_events(
    redis_client,
    agent_id: str,
    min_id: str,
    max_id: str,
    *,
    count: int = 100,
    stream_key: Optional[str] = None,
) -> list[tuple[str, dict]]:
    """Read a range of entries from agent events stream using XRANGE (sync)."""
    key = _agent_events_stream_key(agent_id, stream_key)
    result = redis_client.xrange(
        key,
        min=min_id,
        max=max_id,
        count=count,
    )
    return [(_decode_stream_id(msg_id), data) for msg_id, data in result]


def range_agent_events_reverse(
    redis_client,
    agent_id: str,
    *,
    count: int = 100,
    max_id: str = "+",
    min_id: str = "-",
    stream_key: Optional[str] = None,
) -> list[tuple[str, dict]]:
    """Read a reverse range of entries from agent events stream (sync)."""
    key = _agent_events_stream_key(agent_id, stream_key)
    result = redis_client.xrevrange(
        key,
        max=max_id,
        min=min_id,
        count=count,
    )
    return [(_decode_stream_id(msg_id), data) for msg_id, data in result]


def append_agent_event(
    redis_client,
    agent_id: str,
    payload: dict,
    *,
    maxlen: Optional[int] = None,
    approximate: bool = True,
    stream_key: Optional[str] = None,
) -> str:
    """Append an entry to agent events stream (sync)."""
    key = _agent_events_stream_key(agent_id, stream_key)
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


def get_agent_events_length(
    redis_client,
    agent_id: str,
    *,
    stream_key: Optional[str] = None,
) -> int:
    """Return agent events stream length (sync)."""
    key = _agent_events_stream_key(agent_id, stream_key)
    return redis_client.xlen(key)
