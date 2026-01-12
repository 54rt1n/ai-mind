# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Sync helpers for agent profile access."""

from __future__ import annotations

from typing import Optional

from .redis_keys import RedisKeys
from .helper import _utc_now


def _decode_hash(raw: dict) -> dict[str, str]:
    decoded: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, bytes):
            k = k.decode("utf-8")
        if isinstance(v, bytes):
            v = v.decode("utf-8")
        decoded[str(k)] = str(v)
    return decoded


def get_agent_profile_raw(redis_client, agent_id: str) -> dict[str, str]:
    """Fetch agent profile hash and decode to string values."""
    key = RedisKeys.agent_profile(agent_id)
    data = redis_client.hgetall(key) or {}
    return _decode_hash(data)


def update_agent_profile_fields(
    redis_client,
    agent_id: str,
    touch_updated_at: bool = False,
    **fields,
) -> bool:
    """Partial update of agent profile fields (sync)."""
    key = RedisKeys.agent_profile(agent_id)
    if touch_updated_at and "updated_at" not in fields:
        fields["updated_at"] = _utc_now().isoformat()
    payload = {k: v for k, v in fields.items() if v is not None}
    if not payload:
        return False
    redis_client.hset(key, mapping=payload)
    return True


def set_agent_profile_fields(redis_client, agent_id: str, fields: dict[str, str]) -> bool:
    """Set agent profile fields from a mapping (sync)."""
    key = RedisKeys.agent_profile(agent_id)
    if not fields:
        return False
    redis_client.hset(key, mapping=fields)
    return True
