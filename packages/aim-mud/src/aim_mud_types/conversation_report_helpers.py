# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Helpers for conversation report cache access."""

from __future__ import annotations

import json
import logging
from typing import Optional

from .redis_keys import RedisKeys

logger = logging.getLogger(__name__)


def _decode_report(raw) -> Optional[dict]:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if raw == "":
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to decode conversation report JSON: %s", exc, exc_info=True)
        return None
    if not isinstance(data, dict):
        return None
    return data


def get_conversation_report(redis_client, agent_id: str) -> Optional[dict]:
    """Fetch conversation report dict (sync)."""
    key = RedisKeys.agent_conversation_report(agent_id)
    raw = redis_client.get(key)
    return _decode_report(raw)


def set_conversation_report(redis_client, agent_id: str, report: dict) -> bool:
    """Store conversation report dict (sync)."""
    key = RedisKeys.agent_conversation_report(agent_id)
    payload = json.dumps(report or {})
    redis_client.set(key, payload)
    return True


async def get_conversation_report_async(redis_client, agent_id: str) -> Optional[dict]:
    """Fetch conversation report dict (async)."""
    key = RedisKeys.agent_conversation_report(agent_id)
    raw = await redis_client.get(key)
    return _decode_report(raw)


async def set_conversation_report_async(redis_client, agent_id: str, report: dict) -> bool:
    """Store conversation report dict (async)."""
    key = RedisKeys.agent_conversation_report(agent_id)
    payload = json.dumps(report or {})
    await redis_client.set(key, payload)
    return True
