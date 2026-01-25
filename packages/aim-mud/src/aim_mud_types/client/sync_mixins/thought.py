# aim-mud-types/client/sync_mixins/thought.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Sync thought state operations.

This mixin provides CRUD operations for ThoughtState, which tracks:
- Reasoning content
- Creation timestamp
- Actions since generation (for throttle)

Thoughts can come from various sources:
- "reasoning": Generated during thinking turn
- "manual": Injected by MUD commands (@thought)
- "dreamer": Generated during dream processing
"""

from typing import TYPE_CHECKING, Optional

from ...redis_keys import RedisKeys
from ...models.coordination import ThoughtState

if TYPE_CHECKING:
    from ..base import BaseSyncRedisMUDClient


class SyncThoughtMixin:
    """Sync thought state Redis operations.

    ThoughtState is stored as a Redis hash at agent:{id}:thought
    """

    def get_thought_state(
        self: "BaseSyncRedisMUDClient",
        agent_id: str
    ) -> Optional[ThoughtState]:
        """Fetch thought state as Pydantic model.

        Args:
            agent_id: Agent identifier

        Returns:
            ThoughtState object, or None if no thought exists or invalid format
        """
        key = RedisKeys.agent_thought(agent_id)
        return self._get_hash(ThoughtState, key)

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
        key = RedisKeys.agent_thought(agent_id)
        return self.redis.exists(key) == 1

    def get_thought(
        self: "BaseSyncRedisMUDClient",
        agent_id: str
    ) -> Optional[dict]:
        """Get agent's thought data as raw dict.

        Legacy method - prefer get_thought_state().

        Args:
            agent_id: Agent identifier

        Returns:
            Dict with thought fields or None
        """
        thought = self.get_thought_state(agent_id)
        if thought is None:
            return None
        return thought.model_dump()

    def save_thought_state(
        self: "BaseSyncRedisMUDClient",
        thought: ThoughtState,
        ttl_seconds: int = 7200
    ) -> bool:
        """Save thought state to Redis with TTL.

        Args:
            thought: ThoughtState to save
            ttl_seconds: Time-to-live in seconds (default: 2 hours)

        Returns:
            True if saved successfully
        """
        key = RedisKeys.agent_thought(thought.agent_id)
        created = self._create_hash(key, thought, exists_ok=True)
        if created and ttl_seconds > 0:
            self.redis.expire(key, ttl_seconds)
        return created

    def delete_thought_state(
        self: "BaseSyncRedisMUDClient",
        agent_id: str
    ) -> bool:
        """Delete thought state from Redis.

        Args:
            agent_id: Agent identifier

        Returns:
            True if deleted (key existed)
        """
        key = RedisKeys.agent_thought(agent_id)
        return self.redis.delete(key) > 0
