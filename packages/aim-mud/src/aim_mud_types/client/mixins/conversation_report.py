# aim-mud-types/client/mixins/conversation_report.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Conversation report cache operations."""

import json
import logging
from typing import Optional, TYPE_CHECKING

from ...redis_keys import RedisKeys

if TYPE_CHECKING:
    from .. import BaseRedisMUDClient


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


class ConversationReportMixin:
    """Conversation report cache Redis operations."""

    async def get_conversation_report(
        self: "BaseRedisMUDClient",
        agent_id: str,
    ) -> Optional[dict]:
        """Fetch conversation report dict for agent."""
        key = RedisKeys.agent_conversation_report(agent_id)
        raw = await self.redis.get(key)
        return _decode_report(raw)

    async def set_conversation_report(
        self: "BaseRedisMUDClient",
        agent_id: str,
        report: dict,
    ) -> bool:
        """Store conversation report dict for agent."""
        key = RedisKeys.agent_conversation_report(agent_id)
        payload = json.dumps(report or {})
        await self.redis.set(key, payload)
        return True
