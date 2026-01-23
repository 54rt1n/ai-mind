# aim-mud-types/client/sync_mixins/agent_profile.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Sync AgentProfile-specific operations."""

from typing import Optional, TYPE_CHECKING

from ...redis_keys import RedisKeys
from ...helper import _utc_now
from ...models.profile import AgentProfile

if TYPE_CHECKING:
    from ..base import BaseSyncRedisMUDClient


class SyncAgentProfileMixin:
    """Sync AgentProfile-specific Redis operations.

    Provides CRUD operations for agent profile persistence.
    Note: Uses `agent:{id}` key format (not `agent:{id}:profile`).
    """

    def get_agent_profile(
        self: "BaseSyncRedisMUDClient",
        agent_id: str
    ) -> Optional[AgentProfile]:
        """Fetch agent profile.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentProfile object, or None if not found
        """
        key = RedisKeys.agent_profile(agent_id)
        return self._get_hash(AgentProfile, key)

    def get_agent_profile_raw(
        self: "BaseSyncRedisMUDClient",
        agent_id: str
    ) -> dict[str, str]:
        """Fetch agent profile hash and decode to string values."""
        key = RedisKeys.agent_profile(agent_id)
        data = self.redis.hgetall(key)
        decoded: dict[str, str] = {}
        for k, v in (data or {}).items():
            if isinstance(k, bytes):
                k = k.decode("utf-8")
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            decoded[str(k)] = str(v)
        return decoded

    def create_agent_profile(
        self: "BaseSyncRedisMUDClient",
        profile: AgentProfile
    ) -> bool:
        """Create or update agent profile.

        Args:
            profile: Complete AgentProfile object

        Returns:
            True (always succeeds, overwrites if exists)
        """
        key = RedisKeys.agent_profile(profile.agent_id)
        return self._create_hash(key, profile, exists_ok=True)

    def update_agent_profile_fields(
        self: "BaseSyncRedisMUDClient",
        agent_id: str,
        touch_updated_at: bool = False,
        **fields,
    ) -> bool:
        """Partial update of agent profile fields.

        Args:
            agent_id: Agent identifier
            **fields: Field names and values to update

        Returns:
            True if updated

        Example:
            client.update_agent_profile_fields(
                "andi",
                conversation_id="conv2",
                last_event_id="12345-0"
            )
        """
        if touch_updated_at and "updated_at" not in fields:
            fields["updated_at"] = _utc_now()
        key = RedisKeys.agent_profile(agent_id)
        return self._update_fields(key, fields)

    def get_agent_is_sleeping(
        self: "BaseSyncRedisMUDClient",
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
        result = self._get_field(key, "is_sleeping")
        return result is not None and result.lower() == "true"

    def set_agent_profile_fields(
        self: "BaseSyncRedisMUDClient",
        agent_id: str,
        fields: dict[str, str],
    ) -> bool:
        """Set agent profile fields from a mapping.

        Args:
            agent_id: Agent identifier
            fields: Dictionary of field names and string values

        Returns:
            True if fields were set, False if fields dict was empty
        """
        if not fields:
            return False
        key = RedisKeys.agent_profile(agent_id)
        self.redis.hset(key, mapping=fields)
        return True
