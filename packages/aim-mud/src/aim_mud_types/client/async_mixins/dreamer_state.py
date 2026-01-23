# aim-mud-types/client/mixins/dreamer_state.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""DreamerState-specific operations."""

from typing import Optional, TYPE_CHECKING

from ...redis_keys import RedisKeys
from ...models.coordination import DreamerState

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
