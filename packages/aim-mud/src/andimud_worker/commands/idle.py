# andimud_worker/commands/idle.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Idle command - check active plan, auto-dream triggers, and process idle turn."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import TurnRequestStatus
from .base import Command
from .result import CommandResult

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class IdleCommand(Command):
    """Idle turn - check active plan, then auto-dream triggers.

    Priority order:
    1. If agent has an ACTIVE plan -> set up plan context for turn
    2. Else if auto-dream triggers met -> run dream pipeline
    3. Else -> fall through to idle turn processing
    """

    @property
    def name(self) -> str:
        return "idle"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Process idle turn, checking for plan or auto-dream triggers.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: Contains turn_id

        Returns:
            CommandResult with complete=False (falls through to process_turn)
        """
        turn_id = kwargs.get("turn_id", "unknown")

        # Priority 1: Check for active plan
        plan = await worker.check_active_plan()
        if plan:
            logger.info(f"[{turn_id}] Active plan detected: {plan.summary}")

            # Set plan on worker for context injection
            worker.set_active_plan(plan)

            # Set Redis context on decision strategy for plan tool execution
            # This must happen BEFORE init_tools() is called (which happens in start())
            # but also before any plan tool execution in this turn
            if hasattr(worker, "_decision_strategy") and worker._decision_strategy:
                worker._decision_strategy.set_context(worker.redis, worker.config.agent_id)

            # Add plan tools to tool helper if available
            if hasattr(worker, "_tool_helper") and worker._tool_helper:
                worker._tool_helper.add_plan_tools(plan, worker.chat_config.tools_path)

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

        # Priority 2: Check auto-dream triggers during idle
        dream_request = await worker.check_auto_dream_triggers()
        if dream_request:
            logger.info(f"Auto-dream triggered: {dream_request.scenario}")
            result = await worker.process_dream_turn(
                scenario=dream_request.scenario,
                query=dream_request.query,
                guidance=dream_request.guidance,
                triggered_by="auto",
            )
            if result.success:
                logger.info(
                    f"Auto-dream completed: {result.pipeline_id} "
                    f"in {result.duration_seconds:.1f}s"
                )
                # Update conversation report
                await worker._update_conversation_report()
            else:
                logger.warning(f"Auto-dream failed: {result.error}")

        # Fall through to process_turn with empty events
        # flush_drain=True tells worker to clear pending_events
        return CommandResult(
            complete=False,
            flush_drain=True,
            saved_event_id=None,
            status=TurnRequestStatus.DONE,
            message="Idle turn ready",
        )
