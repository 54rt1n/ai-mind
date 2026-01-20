# andimud_worker/commands/idle.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Idle command - check active plan, dreams, and process idle turn."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import TurnRequestStatus
from aim_mud_types.coordination import DreamStatus
from .base import Command
from .result import CommandResult

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class IdleCommand(Command):
    """Idle turn - check plan, dreams, auto-analysis, then fall through.

    Priority order:
    1. Active plan -> set up plan context, fall through to process_turn
    2. PENDING dream -> initialize to RUNNING, execute first step
    3. RUNNING dream -> execute next step
    4. No dream -> check for unanalyzed conversations (auto-analysis)
    5. Regular idle -> fall through to process_turn
    """

    @property
    def name(self) -> str:
        return "idle"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Process idle turn with 5-priority logic.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: Contains turn_id

        Returns:
            CommandResult - complete=True if dream handled turn,
            complete=False to fall through to process_turn
        """
        turn_id = kwargs.get("turn_id", "unknown")

        # Priority 1: Use active plan already loaded by worker
        plan = worker.get_active_plan()
        if plan:
            logger.info(f"[{turn_id}] Priority 1: Active plan detected: {plan.summary}")

            # Build plan guidance for user footer
            plan_guidance = ""
            if hasattr(worker, "_decision_strategy") and worker._decision_strategy:
                plan_guidance = worker._decision_strategy.get_plan_guidance()

            # Build message with current task info
            message = "Plan active"
            if plan.current_task_id < len(plan.tasks):
                message = f"Plan active: {plan.tasks[plan.current_task_id].summary}"

            # Fall through to process_turn with plan context
            return CommandResult(
                complete=False,
                flush_drain=True,
                saved_event_id=None,
                status=TurnRequestStatus.DONE,
                message=message,
                plan_guidance=plan_guidance,
            )

        # Load dreaming state
        dreaming_state = await worker.load_dreaming_state(worker.config.agent_id)

        # Priority 2: PENDING dreams (manual commands waiting for initialization)
        if dreaming_state and dreaming_state.status == DreamStatus.PENDING:
            logger.info(
                f"[{turn_id}] Priority 2: Initializing PENDING dream "
                f"(scenario={dreaming_state.scenario_name})"
            )
            # Initialize PENDING -> RUNNING
            initialized_state = await worker.initialize_pending_dream(dreaming_state)
            # Execute first step immediately
            is_complete = await worker.execute_scenario_step(initialized_state.pipeline_id)
            return CommandResult(
                complete=True,
                flush_drain=True,
                saved_event_id=None,
                status=TurnRequestStatus.DONE,
                message=f"Dream step executed (complete={is_complete})",
            )

        # Priority 3: RUNNING dreams (continue step-by-step execution)
        if dreaming_state and dreaming_state.status == DreamStatus.RUNNING:
            # Check for stale dream (missing framework/state from before code upgrade)
            if not dreaming_state.framework or not dreaming_state.state:
                logger.warning(
                    f"[{turn_id}] Priority 3: Aborting stale dream {dreaming_state.pipeline_id} "
                    f"(missing framework/state fields)"
                )
                dreaming_state.status = DreamStatus.FAILED
                await worker.save_dreaming_state(dreaming_state)
                await worker.archive_dreaming_state(dreaming_state)
                await worker.delete_dreaming_state(worker.config.agent_id)
                return CommandResult(
                    complete=True,
                    flush_drain=True,
                    saved_event_id=None,
                    status=TurnRequestStatus.DONE,
                    message="Stale dream aborted (missing framework/state)",
                )

            logger.debug(
                f"[{turn_id}] Priority 3: Executing dream step"
            )
            is_complete = await worker.execute_scenario_step(dreaming_state.pipeline_id)
            return CommandResult(
                complete=True,
                flush_drain=True,
                saved_event_id=None,
                status=TurnRequestStatus.DONE,
                message=f"Dream step executed (complete={is_complete})",
            )

        # Priority 4: Auto-analysis check (no active dream)
        if not dreaming_state:
            logger.debug(f"[{turn_id}] Priority 4: Checking if should auto-analyze")
            dream_decision = await worker._decide_dream_action()

            if dream_decision:
                logger.info(
                    f"[{turn_id}] Auto-analysis: initiating {dream_decision.scenario} "
                    f"on {dream_decision.conversation_id}"
                )
                new_state = await worker.initialize_auto_dream(dream_decision)
                is_complete = await worker.execute_scenario_step(new_state.pipeline_id)
                return CommandResult(
                    complete=True,
                    flush_drain=True,
                    saved_event_id=None,
                    status=TurnRequestStatus.DONE,
                    message=f"Dream step executed (complete={is_complete})",
                )

        # Priority 5: Regular idle - fall through to process_turn
        logger.debug(f"[{turn_id}] Priority 5: Regular idle turn")
        return CommandResult(
            complete=False,
            flush_drain=True,
            saved_event_id=None,
            status=TurnRequestStatus.DONE,
            message="Idle turn ready",
        )
