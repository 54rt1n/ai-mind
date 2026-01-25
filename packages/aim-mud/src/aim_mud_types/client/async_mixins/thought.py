# aim-mud-types/client/async_mixins/thought.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Thought state operations for throttle-based reasoning.

This mixin provides CRUD operations for ThoughtState, which tracks:
- Reasoning content
- Creation timestamp
- Actions since generation (for throttle)

Thoughts can come from various sources:
- "reasoning": Generated during thinking turn
- "manual": Injected by MUD commands (@think)
- "dreamer": Generated during dream processing
"""

from typing import TYPE_CHECKING, Optional

from ...redis_keys import RedisKeys
from ...models.coordination import ThoughtState

if TYPE_CHECKING:
    from ..base import BaseAsyncRedisMUDClient


class ThoughtMixin:
    """Thought state Redis operations.

    ThoughtState is stored as a Redis hash at agent:{id}:thought
    """

    async def get_thought_state(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str
    ) -> Optional[ThoughtState]:
        """Fetch thought state as Pydantic model.

        Args:
            agent_id: Agent identifier

        Returns:
            ThoughtState object, or None if no thought exists or invalid format
        """
        key = RedisKeys.agent_thought(agent_id)
        return await self._get_hash(ThoughtState, key)

    async def save_thought_state(
        self: "BaseAsyncRedisMUDClient",
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
        created = await self._create_hash(key, thought, exists_ok=True)
        if created and ttl_seconds > 0:
            await self.redis.expire(key, ttl_seconds)
        return created

    async def increment_thought_action_counter(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str
    ) -> int:
        """Atomically increment actions_since_generation counter.

        Uses Redis HINCRBY for atomic operation. Safe to call even if
        thought doesn't exist (creates the field with value 1).

        Args:
            agent_id: Agent identifier

        Returns:
            New counter value
        """
        key = RedisKeys.agent_thought(agent_id)
        return await self.redis.hincrby(key, "actions_since_generation", 1)

    async def delete_thought_state(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str
    ) -> bool:
        """Delete thought state from Redis.

        Args:
            agent_id: Agent identifier

        Returns:
            True if deleted
        """
        key = RedisKeys.agent_thought(agent_id)
        return await self.redis.delete(key) > 0

    async def should_generate_thought(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str
    ) -> bool:
        """Check if agent should generate a new thought.

        Returns True if:
        - No thought exists, OR
        - Existing thought meets throttle conditions (5min OR 5 actions)

        Args:
            agent_id: Agent identifier

        Returns:
            True if new thought should be generated
        """
        thought = await self.get_thought_state(agent_id)
        if thought is None:
            return True
        return thought.should_regenerate()

    # Legacy compatibility methods

    async def has_active_thought(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str
    ) -> bool:
        """Check if agent has an active thought in Redis.

        Legacy method - prefer get_thought_state().

        Args:
            agent_id: Agent identifier

        Returns:
            True if agent has an active thought
        """
        key = RedisKeys.agent_thought(agent_id)
        return await self.redis.exists(key) == 1

    async def get_thought(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str
    ) -> Optional[dict]:
        """Get agent's thought data as raw dict.

        Legacy method - prefer get_thought_state().

        Args:
            agent_id: Agent identifier

        Returns:
            Dict with thought fields or None
        """
        thought = await self.get_thought_state(agent_id)
        if thought is None:
            return None
        return thought.model_dump()
