# andimud_mediator/mixins/planner.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Planner mixin for the mediator service."""

import logging
from aim_mud_types import MUDEvent, EventType, TurnRequestStatus, TurnReason, normalize_agent_id
from aim_mud_types.client import RedisMUDClient
from aim_mud_types.plan import PlanStatus

from ..patterns import PLANNER_PATTERN, PLAN_PATTERN, UPDATE_PATTERN

logger = logging.getLogger(__name__)


class PlannerMixin:
    """Planner mixin for the mediator service.

    Provides command handlers for:
    - @planner <agent> on/off - toggle planner enabled
    - @plan <agent> = <objective> - create a new plan
    - @update <agent> = <guidance> - update plan with guidance
    """

    async def _handle_planner_toggle(
        self,
        agent_id: str,
        enabled: bool,
    ) -> bool:
        """Handle @planner command by toggling planner enabled state.

        Args:
            agent_id: Target agent ID.
            enabled: True to enable, False to disable.

        Returns:
            True if state was updated successfully.
        """
        # Check agent is registered
        if self.registered_agents and agent_id not in self.registered_agents:
            logger.warning(f"@planner: Agent '{agent_id}' not registered")
            return False

        client = RedisMUDClient(self.redis)
        await client.set_planner_enabled(agent_id, enabled)
        logger.info(f"@planner: Set {agent_id} planner enabled={enabled}")
        return True

    async def _handle_plan_command(
        self,
        agent_id: str,
        objective: str,
    ) -> bool:
        """Handle @plan command by assigning a planner turn.

        Creates a turn_request with reason="dream" and metadata indicating
        planner pipeline with the objective.

        Args:
            agent_id: Target agent ID.
            objective: The plan objective (high-level goal).

        Returns:
            True if turn was assigned, False if agent unavailable.
        """
        # Check agent is registered
        if self.registered_agents and agent_id not in self.registered_agents:
            logger.warning(f"@plan: Agent '{agent_id}' not registered")
            return False

        # Get current turn request state
        current = await self._get_turn_request(agent_id)
        if not current:
            logger.warning(f"@plan: Agent '{agent_id}' offline (no turn_request)")
            return False

        status = current.status

        # Validate status
        if status is None:
            logger.error(f"@plan: Agent '{agent_id}' has corrupted turn_request")
            return False

        # Block if agent is busy
        if status == TurnRequestStatus.CRASHED:
            logger.warning(f"@plan: Agent '{agent_id}' is crashed")
            return False

        if status in (TurnRequestStatus.ASSIGNED, TurnRequestStatus.IN_PROGRESS,
                      TurnRequestStatus.ABORT_REQUESTED, TurnRequestStatus.EXECUTING,
                      TurnRequestStatus.RETRY):
            logger.warning(f"@plan: Agent '{agent_id}' is busy (status={status.value})")
            return False

        # Build metadata with planner pipeline indicator
        metadata = {
            "pipeline": "planner",
            "objective": objective,
        }

        from aim_mud_types.turn_request_helpers import assign_turn_request_async
        success, turn_request, result = await assign_turn_request_async(
            self.redis,
            agent_id,
            TurnReason.DREAM,
            attempt_count=0,
            deadline_ms=1800000,
            status=TurnRequestStatus.ASSIGNED,
            expected_turn_id=current.turn_id,
            skip_availability_check=True,
            **metadata,
        )

        if success and turn_request:
            logger.info(
                f"@plan: Assigned planner turn to {agent_id} "
                f"(sequence_id={turn_request.sequence_id}, objective={objective[:50]}...)"
            )
            return True
        else:
            logger.debug(f"@plan: Assign failed for {agent_id}: {result}")
            return False

    async def _handle_update_command(
        self,
        agent_id: str,
        guidance: str,
    ) -> bool:
        """Handle @update command by assigning an update turn with guidance.

        The guidance will be injected into the agent's next turn to help them
        update their current task status (completed/blocked/skipped).

        Args:
            agent_id: Target agent ID.
            guidance: Guidance text for the update.

        Returns:
            True if guidance was set, False if agent unavailable.
        """
        # Check agent is registered
        if self.registered_agents and agent_id not in self.registered_agents:
            logger.warning(f"@update: Agent '{agent_id}' not registered")
            return False

        # Check if agent has an active plan
        client = RedisMUDClient(self.redis)

        plan = await client.get_plan(agent_id)
        if not plan:
            logger.warning(f"@update: Agent '{agent_id}' has no plan")
            return False

        if plan.status != PlanStatus.ACTIVE and plan.status != PlanStatus.BLOCKED:
            logger.warning(f"@update: Agent '{agent_id}' plan is not active (status={plan.status.value})")
            return False

        # Get current turn request
        current = await self._get_turn_request(agent_id)
        if not current:
            logger.warning(f"@update: Agent '{agent_id}' offline")
            return False

        status = current.status
        if status is None:
            logger.error(f"@update: Agent '{agent_id}' has corrupted turn_request")
            return False

        # Block if busy
        if status in (TurnRequestStatus.ASSIGNED, TurnRequestStatus.IN_PROGRESS,
                      TurnRequestStatus.ABORT_REQUESTED, TurnRequestStatus.EXECUTING,
                      TurnRequestStatus.RETRY, TurnRequestStatus.CRASHED):
            logger.warning(f"@update: Agent '{agent_id}' is busy (status={status.value})")
            return False

        # Create an idle turn with plan guidance
        # The idle command will detect the active plan and inject context
        metadata = {
            "plan_guidance": guidance,
        }

        from aim_mud_types.turn_request_helpers import assign_turn_request_async
        success, turn_request, result = await assign_turn_request_async(
            self.redis,
            agent_id,
            TurnReason.IDLE,
            attempt_count=0,
            deadline_ms=300000,
            status=TurnRequestStatus.ASSIGNED,
            expected_turn_id=current.turn_id,
            skip_availability_check=True,
            **metadata,
        )

        if success and turn_request:
            logger.info(f"@update: Assigned update turn to {agent_id} with guidance")
            return True
        else:
            logger.debug(f"@update: Assign failed for {agent_id}: {result}")
            return False

    async def _try_handle_planner_command(self, event: MUDEvent) -> bool:
        """Check if event is a planner command and handle it.

        Args:
            event: The MUDEvent to check.

        Returns:
            True if event was a planner command (handled), False otherwise.
        """
        # Only check SYSTEM events
        if event.event_type != EventType.SYSTEM:
            return False

        content = event.content.strip() if event.content else ""
        if not content:
            return False

        # Try @planner command
        match = PLANNER_PATTERN.match(content)
        if match:
            agent_id = normalize_agent_id(match.group(1))
            enabled = match.group(2).lower() == "on"
            await self._handle_planner_toggle(agent_id, enabled)
            return True

        # Try @plan command
        match = PLAN_PATTERN.match(content)
        if match:
            agent_id = normalize_agent_id(match.group(1))
            objective = match.group(2).strip()
            await self._handle_plan_command(agent_id, objective)
            return True

        # Try @update command
        match = UPDATE_PATTERN.match(content)
        if match:
            agent_id = normalize_agent_id(match.group(1))
            guidance = match.group(2).strip()
            await self._handle_update_command(agent_id, guidance)
            return True

        return False
