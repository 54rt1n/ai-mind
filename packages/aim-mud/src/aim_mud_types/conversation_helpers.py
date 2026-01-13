# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Sync helpers for agent conversation lists."""

from __future__ import annotations

from typing import Optional

from .redis_keys import RedisKeys


def _decode_list_entry(raw) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def get_conversation_entries(
    redis_client,
    agent_id: str,
    start: int = 0,
    end: int = -1,
) -> list[str]:
    """Fetch conversation entries as JSON strings (sync)."""
    key = RedisKeys.agent_conversation(agent_id)
    raw_entries = redis_client.lrange(key, start, end) or []
    return [_decode_list_entry(raw) for raw in raw_entries]


def get_conversation_entry(
    redis_client,
    agent_id: str,
    index: int,
) -> Optional[str]:
    """Fetch a single conversation entry by index (sync)."""
    key = RedisKeys.agent_conversation(agent_id)
    raw = redis_client.lindex(key, index)
    if raw is None:
        return None
    return _decode_list_entry(raw)


def append_conversation_entry(
    redis_client,
    agent_id: str,
    entry_json: str,
) -> int:
    """Append a conversation entry JSON to the list (sync)."""
    key = RedisKeys.agent_conversation(agent_id)
    return redis_client.rpush(key, entry_json)


def set_conversation_entry(
    redis_client,
    agent_id: str,
    index: int,
    entry_json: str,
) -> bool:
    """Set a conversation entry at index (sync)."""
    key = RedisKeys.agent_conversation(agent_id)
    redis_client.lset(key, index, entry_json)
    return True


def pop_conversation_entry(
    redis_client,
    agent_id: str,
) -> Optional[str]:
    """Pop the oldest conversation entry (left) (sync)."""
    key = RedisKeys.agent_conversation(agent_id)
    raw = redis_client.lpop(key)
    if raw is None:
        return None
    return _decode_list_entry(raw)


def pop_last_conversation_entry(
    redis_client,
    agent_id: str,
) -> Optional[str]:
    """Pop the newest conversation entry (right) (sync)."""
    key = RedisKeys.agent_conversation(agent_id)
    raw = redis_client.rpop(key)
    if raw is None:
        return None
    return _decode_list_entry(raw)


def get_conversation_length(
    redis_client,
    agent_id: str,
) -> int:
    """Return conversation list length (sync)."""
    key = RedisKeys.agent_conversation(agent_id)
    return redis_client.llen(key)


def delete_conversation(
    redis_client,
    agent_id: str,
) -> int:
    """Delete conversation list key (sync)."""
    key = RedisKeys.agent_conversation(agent_id)
    return redis_client.delete(key)


def replace_conversation_entries(
    redis_client,
    agent_id: str,
    entries_json: list[str],
) -> None:
    """Replace entire conversation list (sync)."""
    key = RedisKeys.agent_conversation(agent_id)
    pipe = redis_client.pipeline()
    pipe.delete(key)
    for entry_json in entries_json:
        pipe.rpush(key, entry_json)
    pipe.execute()
