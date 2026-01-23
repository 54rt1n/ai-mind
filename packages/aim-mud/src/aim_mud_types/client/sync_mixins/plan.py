# aim-mud-types/client/sync_mixins/plan.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Sync Plan-specific operations."""

from typing import Optional, TYPE_CHECKING

from ...redis_keys import RedisKeys
from ...models.plan import AgentPlan

if TYPE_CHECKING:
    from ..base import BaseSyncRedisMUDClient


class SyncPlanMixin:
    """Sync Plan-specific Redis operations.

    Provides CRUD operations for agent execution plans.
    """

    def get_plan(
        self: "BaseSyncRedisMUDClient",
        agent_id: str
    ) -> Optional[AgentPlan]:
        """Fetch current plan for an agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            AgentPlan if exists, None otherwise.
        """
        key = RedisKeys.agent_plan(agent_id)
        return self._get_hash(AgentPlan, key)

    def create_plan(
        self: "BaseSyncRedisMUDClient",
        plan: AgentPlan
    ) -> bool:
        """Create or overwrite a plan.

        Args:
            plan: The plan to store.

        Returns:
            True if successful.
        """
        key = RedisKeys.agent_plan(plan.agent_id)
        return self._create_hash(key, plan, exists_ok=True)

    def update_plan_fields(
        self: "BaseSyncRedisMUDClient",
        agent_id: str,
        **fields
    ) -> bool:
        """Partial update of plan fields.

        Args:
            agent_id: Agent identifier.
            **fields: Fields to update (key=value).

        Returns:
            True if updated.

        Example:
            client.update_plan_fields(
                "andi",
                status=PlanStatus.ACTIVE,
                current_task_id=1
            )
        """
        key = RedisKeys.agent_plan(agent_id)
        return self._update_fields(key, fields)

    def delete_plan(
        self: "BaseSyncRedisMUDClient",
        agent_id: str
    ) -> bool:
        """Delete agent's current plan.

        Args:
            agent_id: Agent identifier.

        Returns:
            True if deleted, False if didn't exist.
        """
        key = RedisKeys.agent_plan(agent_id)
        result = self.redis.delete(key)
        return result > 0

    def is_planner_enabled(
        self: "BaseSyncRedisMUDClient",
        agent_id: str
    ) -> bool:
        """Check if planner is enabled for agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            True if enabled, False if disabled or not set.
        """
        key = RedisKeys.agent_planner_enabled(agent_id)
        value = self.redis.get(key)
        if value is None:
            return False
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return value.lower() == "true"

    def set_planner_enabled(
        self: "BaseSyncRedisMUDClient",
        agent_id: str,
        enabled: bool
    ) -> None:
        """Set planner enabled state for agent.

        Args:
            agent_id: Agent identifier.
            enabled: True to enable, False to disable.
        """
        key = RedisKeys.agent_planner_enabled(agent_id)
        self.redis.set(key, "true" if enabled else "false")
