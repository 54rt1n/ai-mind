# andimud_worker/mixins/planner.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Planner mixin for MUD worker.

Adds plan-following capabilities to MUDAgentWorker:
- check_active_plan(): Check if agent has an active plan ready for execution
- set_active_plan(): Set the active plan for context injection
- clear_active_plan(): Clear plan context after turn completes
- get_active_plan(): Get the currently active plan
"""

from typing import TYPE_CHECKING, Optional
import logging

if TYPE_CHECKING:
    from aim_mud_types.plan import AgentPlan
    from ..worker import MUDAgentWorker

logger = logging.getLogger(__name__)


class PlannerMixin:
    """Mixin adding planner capabilities to MUDAgentWorker.

    Adds plan checking and context injection to MUDAgentWorker.

    Expected attributes from MUDAgentWorker:
    - self.redis: Async Redis client
    - self.config: MUDConfig (has agent_id)
    - self._decision_strategy: Optional strategy with _active_plan attribute
    """

    _active_plan: Optional["AgentPlan"] = None

    async def check_active_plan(self: "MUDAgentWorker") -> Optional["AgentPlan"]:
        """Check if agent has an active plan that's ready for execution.

        Returns:
            AgentPlan if planner is enabled and plan is ACTIVE, None otherwise.
        """
        from aim_mud_types.client import RedisMUDClient
        from aim_mud_types.plan import PlanStatus

        client = RedisMUDClient(self.redis)

        # Check if planner is enabled
        if not await client.is_planner_enabled(self.config.agent_id):
            return None

        # Get active plan
        plan = await client.get_plan(self.config.agent_id)
        if not plan:
            return None

        # Only return if ACTIVE status (not BLOCKED, COMPLETED, etc.)
        if plan.status != PlanStatus.ACTIVE:
            return None

        return plan

    def set_active_plan(self: "MUDAgentWorker", plan: "AgentPlan") -> None:
        """Set the active plan for context injection.

        This plan will be included in consciousness block and user footer.
        Also sets plan on the decision strategy for memory integration.

        Args:
            plan: The active plan to inject into context.
        """
        self._active_plan = plan

        # Also set on decision strategy for consciousness block
        if hasattr(self, "_decision_strategy") and self._decision_strategy:
            self._decision_strategy._active_plan = plan

    def clear_active_plan(self: "MUDAgentWorker") -> None:
        """Clear the active plan context after turn completes."""
        self._active_plan = None

        if hasattr(self, "_decision_strategy") and self._decision_strategy:
            self._decision_strategy._active_plan = None

    def get_active_plan(self: "MUDAgentWorker") -> Optional["AgentPlan"]:
        """Get the currently active plan.

        Returns:
            The active plan, or None if no plan is active.
        """
        return self._active_plan
