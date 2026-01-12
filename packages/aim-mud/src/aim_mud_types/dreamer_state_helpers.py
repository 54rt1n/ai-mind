# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Sync helpers for dreamer state access."""

from __future__ import annotations

import logging
from typing import Optional

from .coordination import DreamerState
from .redis_keys import RedisKeys

logger = logging.getLogger(__name__)


def _decode_hash(raw: dict) -> dict[str, str]:
    decoded: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, bytes):
            k = k.decode("utf-8")
        if isinstance(v, bytes):
            v = v.decode("utf-8")
        decoded[str(k)] = str(v)
    return decoded


def _serialize_value(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def get_dreamer_state(redis_client, agent_id: str) -> Optional[DreamerState]:
    """Fetch dreamer state for an agent (sync)."""
    key = RedisKeys.agent_dreamer(agent_id)
    data = redis_client.hgetall(key)
    if not data:
        return None
    decoded = _decode_hash(data)
    try:
        return DreamerState.model_validate(decoded)
    except Exception as exc:
        logger.warning(
            "Failed to validate DreamerState from Redis key '%s': %s",
            key,
            exc,
            exc_info=True,
        )
        return None


def update_dreamer_state_fields(redis_client, agent_id: str, **fields) -> bool:
    """Partial update of dreamer state fields (sync)."""
    key = RedisKeys.agent_dreamer(agent_id)
    payload: dict[str, str] = {}
    for field_name, field_value in fields.items():
        serialized = _serialize_value(field_value)
        if serialized is not None:
            payload[field_name] = serialized
    if not payload:
        return False
    redis_client.hset(key, mapping=payload)
    return True
