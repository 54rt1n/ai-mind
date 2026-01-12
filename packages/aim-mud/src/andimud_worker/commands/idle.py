# andimud_worker/commands/idle.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Idle command - check active plan and process idle turn."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import TurnRequestStatus
from .base import Command
from .result import CommandResult

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class IdleCommand(Command):
    """Idle turn - check active plan, then fall through to idle processing.

    Priority order:
    1. If agent has an ACTIVE plan -> set up plan context for turn
    2. Else -> fall through to idle turn processing
    """

    @property
    def name(self) -> str:
        return "idle"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Process idle turn, checking for active plan.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: Contains turn_id

        Returns:
            CommandResult with complete=False (falls through to process_turn)
        """
        turn_id = kwargs.get("turn_id", "unknown")

        # Priority 1: Use active plan already loaded by worker
        plan = worker.get_active_plan()
        if plan:
            logger.info(f"[{turn_id}] Active plan detected: {plan.summary}")

            # set_active_plan handles decision-strategy context and tool refresh

            # Build plan guidance for user footer
            plan_guidance = ""
            if hasattr(worker, "_decision_strategy") and worker._decision_strategy:
                plan_guidance = worker._decision_strategy.get_plan_guidance()

            # Build message with current task info
            message = "Plan active"
            if plan.current_task_id < len(plan.tasks):
                message = f"Plan active: {plan.tasks[plan.current_task_id].summary}"

            # Fall through to process_turn with plan context
            # The decision strategy will inject plan into consciousness block
            # and get_plan_guidance() will be called for user footer
            return CommandResult(
                complete=False,
                flush_drain=True,
                saved_event_id=None,
                status=TurnRequestStatus.DONE,
                message=message,
                plan_guidance=plan_guidance,
            )

        # Fall through to process_turn with empty events
        # flush_drain=True tells worker to clear pending_events
        return CommandResult(
            complete=False,
            flush_drain=True,
            saved_event_id=None,
            status=TurnRequestStatus.DONE,
            message="Idle turn ready",
        )
