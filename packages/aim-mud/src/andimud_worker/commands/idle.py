# andimud_worker/commands/idle.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Idle command - check active plan, dreams, and process idle turn."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDTurnRequest, MUDEvent
from aim_mud_types.client import RedisMUDClient
from aim_mud_types.decision import DecisionType
from aim_mud_types import TurnRequestStatus
from aim_mud_types.coordination import DreamStatus
from .base import Command
from .result import CommandResult
from ..turns.processor.decision import DecisionProcessor
from ..turns.processor.speaking import SpeakingProcessor
from ..turns.processor.thinking import ThinkingTurnProcessor

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
        turn_request = MUDTurnRequest.model_validate(kwargs)
        events = worker.pending_events
        is_sleeping = await worker._check_agent_is_sleeping()

        if is_sleeping:
            return await self._sleep_turn(worker, turn_id, events, turn_request)
        else:
            return await self._awake_turn(worker, turn_id, events, turn_request)


    async def _awake_turn(self, worker: "MUDAgentWorker", turn_id: str, events: list[MUDEvent], turn_request: MUDTurnRequest) -> CommandResult:
        """Process awake turn."""
        # Priority 1: Use active plan already loaded by worker
        plan_guidance = worker.get_plan_guidance()

        # Setup turn context
        await worker._setup_turn_context(events)
        # Priority 2: If we don't have a current thought, we need to generate one
        if not worker._decision_strategy.thought_content:
            thinking_processor = ThinkingTurnProcessor(worker)
            if plan_guidance:
                thinking_processor.user_guidance = plan_guidance
            await thinking_processor.execute(turn_request, events)
            return CommandResult(
                complete=True,
                flush_drain=False,
                saved_event_id=None,
                status=TurnRequestStatus.DONE,
                message="Thought generated",
            )


        plan = worker.get_active_plan()
        if plan:
            # Build message with current task info
            message = "Plan active"
            if plan.current_task_id < len(plan.tasks):
                message = f"Plan active: {plan.tasks[plan.current_task_id].summary}"

        # Priority 3: Agent awake, but no active plan
        return CommandResult(
            complete=True,
            flush_drain=False,
            saved_event_id=None,
            status=TurnRequestStatus.DONE,
            message="Agent awake",
        )

    async def _sleep_turn(self, worker: "MUDAgentWorker", turn_id: str, events: list[MUDEvent], turn_request: MUDTurnRequest) -> CommandResult:
        """Process sleep turn."""
        # Load dreaming state
        dreaming_state = await worker.load_dreaming_state(worker.config.agent_id)

        # Priority 1: PENDING dreams (manual commands waiting for initialization)
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
                flush_drain=False,
                saved_event_id=None,
                status=TurnRequestStatus.DONE,
                message=f"Dream step executed (complete={is_complete})",
            )

        # Priority 2: RUNNING dreams (continue step-by-step execution)
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
                    flush_drain=False,
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
                flush_drain=False,
                saved_event_id=None,
                status=TurnRequestStatus.DONE,
                message=f"Dream step executed (complete={is_complete})",
            )

        # Priority 3: Auto-analysis check (no active dream)
        if not dreaming_state:
            logger.debug(f"[{turn_id}] Priority 3: Checking if should auto-analyze")
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
                    flush_drain=False,
                    saved_event_id=None,
                    status=TurnRequestStatus.DONE,
                    message=f"Dream step executed (complete={is_complete})",
                )

        logger.info(f"[{turn_id}] Priority 4: Agent sleeping, skipping phased turn")
        return CommandResult(
            complete=True,
            flush_drain=False,
            saved_event_id=None,
            status=TurnRequestStatus.DONE,
            message="Agent sleeping",
        )
