# aim-mud-types/client/sync_mixins/thought.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Sync thought injection operations (read-only).

Thoughts are injected by external commands (MUD commands, dreamer, etc.)
and consumed by workers during turn processing. This mixin provides
read-only access to thought data.
"""

import json
from typing import TYPE_CHECKING, Optional

from ...redis_keys import RedisKeys

if TYPE_CHECKING:
    from ..base import BaseSyncRedisMUDClient


def _decode_thought(raw: bytes | str | None) -> Optional[dict]:
    """Decode thought JSON from Redis value.

    Args:
        raw: Raw Redis value (bytes, str, or None)

    Returns:
        Dict with {content, source, timestamp} or None if invalid/missing
    """
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


class SyncThoughtMixin:
    """Sync thought injection Redis operations (read-only).

    Thoughts are external content injected into an agent's processing.
    They can come from various sources:
    - "manual": Injected by MUD commands (@think)
    - "dreamer": Generated during dream processing
    - "system": System-generated prompts

    Thought data structure:
        {
            "content": str,      # The thought text
            "source": str,       # Origin of the thought
            "timestamp": int,    # Unix timestamp when set
        }
    """

    def has_active_thought(
        self: "BaseSyncRedisMUDClient",
        agent_id: str
    ) -> bool:
        """Check if agent has an active thought in Redis.

        Args:
            agent_id: Agent identifier

        Returns:
            True if agent has an active thought, False otherwise
        """
        thought_key = RedisKeys.agent_thought(agent_id)
        return self.redis.exists(thought_key) == 1

    def get_thought(
        self: "BaseSyncRedisMUDClient",
        agent_id: str
    ) -> Optional[dict]:
        """Get agent's thought data.

        Args:
            agent_id: Agent identifier

        Returns:
            Dict with {content, source, timestamp} or None if no thought
        """
        thought_key = RedisKeys.agent_thought(agent_id)
        raw = self.redis.get(thought_key)
        return _decode_thought(raw)
