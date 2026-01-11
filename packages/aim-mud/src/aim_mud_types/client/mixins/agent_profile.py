# aim-mud-types/client/mixins/agent_profile.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""AgentProfile-specific operations."""

from typing import Optional, TYPE_CHECKING

from ...redis_keys import RedisKeys
from ...profile import AgentProfile

if TYPE_CHECKING:
    from .. import BaseRedisMUDClient


class AgentProfileMixin:
    """AgentProfile-specific Redis operations.

    Provides CRUD operations for agent profile persistence.
    Note: Uses `agent:{id}` key format (not `agent:{id}:profile`).
    """

    async def get_agent_profile(
        self: "BaseRedisMUDClient",
        agent_id: str
    ) -> Optional[AgentProfile]:
        """Fetch agent profile.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentProfile object, or None if not found
        """
        key = RedisKeys.agent_profile(agent_id)
        return await self._get_hash(AgentProfile, key)

    async def create_agent_profile(
        self: "BaseRedisMUDClient",
        profile: AgentProfile
    ) -> bool:
        """Create or update agent profile.

        Args:
            profile: Complete AgentProfile object

        Returns:
            True (always succeeds, overwrites if exists)
        """
        key = RedisKeys.agent_profile(profile.agent_id)
        return await self._create_hash(key, profile, exists_ok=True)

    async def update_agent_profile_fields(
        self: "BaseRedisMUDClient",
        agent_id: str,
        **fields
    ) -> bool:
        """Partial update of agent profile fields.

        Args:
            agent_id: Agent identifier
            **fields: Field names and values to update

        Returns:
            True if updated

        Example:
            await client.update_agent_profile_fields(
                "andi",
                conversation_id="conv2",
                last_event_id="12345-0"
            )
        """
        key = RedisKeys.agent_profile(agent_id)
        return await self._update_fields(key, fields)

    async def get_agent_is_sleeping(
        self: "BaseRedisMUDClient",
        agent_id: str
    ) -> bool:
        """Check if agent is sleeping.

        Efficient single-field read for sleep status.

        Args:
            agent_id: Agent identifier

        Returns:
            True if sleeping, False otherwise
        """
        key = RedisKeys.agent_profile(agent_id)
        result = await self._get_field(key, "is_sleeping")
        return result is not None and result.lower() == "true"
