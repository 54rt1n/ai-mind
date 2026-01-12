# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Sync helpers for room profile access."""

from __future__ import annotations

from .redis_keys import RedisKeys


def _decode_hash(raw: dict) -> dict[str, str]:
    decoded: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, bytes):
            k = k.decode("utf-8")
        if isinstance(v, bytes):
            v = v.decode("utf-8")
        decoded[str(k)] = str(v)
    return decoded


def get_room_profile_raw(redis_client, room_id: str) -> dict[str, str]:
    """Fetch room profile hash and decode to string values."""
    key = RedisKeys.room_profile(room_id)
    data = redis_client.hgetall(key) or {}
    return _decode_hash(data)


def update_room_profile_fields(redis_client, room_id: str, **fields) -> bool:
    """Partial update of room profile fields (sync)."""
    key = RedisKeys.room_profile(room_id)
    payload = {k: v for k, v in fields.items() if v is not None}
    if not payload:
        return False
    redis_client.hset(key, mapping=payload)
    return True


def set_room_profile_fields(redis_client, room_id: str, fields: dict[str, str]) -> bool:
    """Set room profile fields from a mapping (sync)."""
    key = RedisKeys.room_profile(room_id)
    if not fields:
        return False
    redis_client.hset(key, mapping=fields)
    return True
