# aim-mud-types/client/mixins/dreamer_state.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""DreamerState-specific operations."""

from datetime import datetime
from typing import Optional, TYPE_CHECKING

from ...redis_keys import RedisKeys
from ...models.coordination import DreamerState, DreamingState, DreamStatus
from ...helper import _utc_now

if TYPE_CHECKING:
    from ..base import BaseAsyncRedisMUDClient


class DreamerStateMixin:
    """DreamerState-specific Redis operations.

    Provides CRUD operations for automatic dreaming configuration.
    """

    async def get_dreamer_state(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str
    ) -> Optional[DreamerState]:
        """Fetch dreamer state for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            DreamerState object, or None if not found
        """
        key = RedisKeys.agent_dreamer(agent_id)
        return await self._get_hash(DreamerState, key)

    async def update_dreamer_state_fields(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str,
        **fields
    ) -> bool:
        """Partial update of dreamer state fields.

        Args:
            agent_id: Agent identifier
            **fields: Field names and values to update

        Returns:
            True if updated

        Example:
            await client.update_dreamer_state_fields(
                "andi",
                enabled=True,
                idle_threshold_seconds=1800
            )
        """
        key = RedisKeys.agent_dreamer(agent_id)
        return await self._update_fields(key, fields)

    async def has_running_dream(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str
    ) -> bool:
        """Check if agent has a dream in PENDING or RUNNING status.

        Args:
            agent_id: The agent ID to check

        Returns:
            True if dream is PENDING or RUNNING, False otherwise
        """
        key = RedisKeys.agent_dreaming_state(agent_id)
        state = await self._get_hash(DreamingState, key)

        if state is None:
            return False

        return state.status in [DreamStatus.PENDING, DreamStatus.RUNNING]

    async def get_dreaming_state(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str
    ) -> Optional[DreamingState]:
        """Fetch active dreaming state for an agent."""
        key = RedisKeys.agent_dreaming_state(agent_id)
        return await self._get_hash(DreamingState, key)

    async def abort_running_dream(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str,
        reason: str = "Aborted by user request",
    ) -> bool:
        """Abort active dream (PENDING/RUNNING), archive it, and clear live state.

        Returns:
            True if an active dream was aborted, False otherwise.
        """
        state = await self.get_dreaming_state(agent_id)
        if state is None:
            return False

        if state.status not in (DreamStatus.PENDING, DreamStatus.RUNNING):
            return False

        now = _utc_now()
        state.status = DreamStatus.ABORTED
        state.updated_at = now
        state.completed_at = now

        metadata = state.metadata or {}
        metadata["abort_reason"] = reason
        metadata["aborted_at"] = int(datetime.timestamp(now))
        state.metadata = metadata

        # Archive final state before deleting active hash.
        history_key = RedisKeys.agent_dreaming_history(agent_id)
        await self.redis.lpush(history_key, state.model_dump_json())
        await self.redis.ltrim(history_key, 0, 99)

        key = RedisKeys.agent_dreaming_state(agent_id)
        await self.redis.delete(key)
        return True
