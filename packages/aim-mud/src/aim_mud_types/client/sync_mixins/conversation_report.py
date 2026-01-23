# aim-mud-types/client/sync_mixins/conversation_report.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Sync conversation report cache operations."""

import json
import logging
from typing import Optional, TYPE_CHECKING

from ...redis_keys import RedisKeys

if TYPE_CHECKING:
    from ..base import BaseSyncRedisMUDClient


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


class SyncConversationReportMixin:
    """Sync conversation report cache Redis operations."""

    def get_conversation_report(
        self: "BaseSyncRedisMUDClient",
        agent_id: str,
    ) -> Optional[dict]:
        """Fetch conversation report dict for agent."""
        key = RedisKeys.agent_conversation_report(agent_id)
        raw = self.redis.get(key)
        return _decode_report(raw)

    def set_conversation_report(
        self: "BaseSyncRedisMUDClient",
        agent_id: str,
        report: dict,
    ) -> bool:
        """Store conversation report dict for agent."""
        key = RedisKeys.agent_conversation_report(agent_id)
        payload = json.dumps(report or {})
        self.redis.set(key, payload)
        return True
