# andimud_worker/commands/idle.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Idle command - check active plan, dreams, and process idle turn."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDTurnRequest, MUDEvent
from aim_mud_types.client import RedisMUDClient
from aim_mud_types.models.decision import DecisionType
from aim_mud_types import TurnRequestStatus
from aim_mud_types.models.coordination import DreamStatus
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
        # Get turn_request directly from kwargs
        turn_request = kwargs.get("turn_request")
        if not turn_request:
            # Fallback for backward compatibility
            turn_request = MUDTurnRequest.model_validate(kwargs)

        turn_id = turn_request.turn_id or "unknown"
        events = kwargs.get("events", [])
        is_sleeping = await worker._check_agent_is_sleeping()

        if is_sleeping:
            return await self._sleep_turn(worker, turn_id, events, turn_request)
        else:
            return await self._awake_turn(worker, turn_id, events, turn_request)


    async def _awake_turn(self, worker: "MUDAgentWorker", turn_id: str, events: list[MUDEvent], turn_request: MUDTurnRequest) -> CommandResult:
        """Process awake turn with throttle-based thinking.

        New flow:
        1. Check if thought should be regenerated (throttle check)
        2. If yes: generate thought, then CONTINUE to action phase (dual turn)
        3. If no: use existing thought for action phase
        4. Take action
        5. Increment action counter (for throttle tracking)
        """
        await worker.ensure_turn_id_current(turn_id)
        # Priority 1: Use active plan already loaded by worker
        plan_guidance = worker.get_plan_guidance()

        # Setup turn context
        await worker._setup_turn_context(events)

        # Phase 1: Throttle-based thought generation
        should_think = await worker._should_generate_new_thought()
        dual_turn = False

        if should_think:
            # Claim the turn for reasoning generation
            turn_id = await worker.claim_idle_turn(turn_request)
            dual_turn = True

            # Clear any stale entries; thinking doesn't depend on "new entries" gating
            worker._current_turn_entries = []
            logger.info(f"[{turn_id}] Throttle passed - generating new thought")

            thinking_processor = ThinkingTurnProcessor(worker)
            if plan_guidance:
                thinking_processor.user_guidance = plan_guidance

            # Generate thought
            await thinking_processor.execute(turn_request, events)
            logger.info(f"[{turn_id}] New thought generated")

            # Reload the thought we just generated
            await worker._load_thought_content()

        # Phase 2: Check plan status for logging
        plan = worker.get_active_plan()
        if plan:
            # Build message with current task info
            message = "Plan active"
            if plan.current_task_id < len(plan.tasks):
                message = f"Plan active: {plan.tasks[plan.current_task_id].summary}"

        # Phase 3: Check if idle_active allows action
        await worker.ensure_turn_id_current(turn_id)

        is_active = await worker._is_idle_active()

        if not is_active:
            return CommandResult(
                complete=True,
                status=TurnRequestStatus.DONE,
                message="Agent awake" + (" (thought generated)" if dual_turn else ""),
                turn_id=turn_id,
            )

        # Phase 4: Take action
        decision = await worker.take_turn(turn_id, events, turn_request, user_guidance=plan_guidance)
        if decision is None:
            error_detail = worker._last_turn_error or "unknown error"
            logger.error("Event turn %s produced no decision: %s", turn_id, error_detail)
            return CommandResult(
                complete=True,
                status=TurnRequestStatus.FAIL,
                message=f"Idle Turn failed: {error_detail}",
                turn_id=turn_id,
            )

        # Clear thought from strategies (but NOT from Redis - throttle needs it)
        await worker._clear_thought_content()

        # Get emitted action_ids from worker (set by _emit_actions during take_turn)
        # All decision types emit actions (including WAIT and CONFUSED with non-published emotes)
        action_ids = worker._last_emitted_action_ids
        expects_echo = worker._last_emitted_expects_echo
        return CommandResult(
            complete=True,
            status=TurnRequestStatus.DONE,
            message=f"Idle Turn processed: {decision.decision_type.name}" + (" (dual)" if dual_turn else ""),
            turn_id=turn_id,
            emitted_action_ids=action_ids,
            expects_echo=expects_echo,
        )

    async def _sleep_turn(self, worker: "MUDAgentWorker", turn_id: str, events: list[MUDEvent], turn_request: MUDTurnRequest) -> CommandResult:
        """Process sleep turn."""
        await worker.ensure_turn_id_current(turn_id)
        # Load dreaming state
        dreaming_state = await worker.load_dreaming_state(worker.config.agent_id)

        # Priority 1: PENDING dreams (manual commands waiting for initialization)
        if dreaming_state and dreaming_state.status == DreamStatus.PENDING:
            claimed_turn_id = await worker.claim_idle_turn(turn_request)
            turn_id = claimed_turn_id
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
                status=TurnRequestStatus.DONE,
                message=f"Dream step executed (complete={is_complete})",
                turn_id=claimed_turn_id,
            )

        # Priority 2: RUNNING dreams (continue step-by-step execution)
        if dreaming_state and dreaming_state.status == DreamStatus.RUNNING:
            claimed_turn_id = await worker.claim_idle_turn(turn_request)
            turn_id = claimed_turn_id
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
                    status=TurnRequestStatus.DONE,
                    message="Stale dream aborted (missing framework/state)",
                    turn_id=claimed_turn_id,
                )

            logger.debug(
                f"[{turn_id}] Priority 3: Executing dream step"
            )
            is_complete = await worker.execute_scenario_step(dreaming_state.pipeline_id)
            return CommandResult(
                complete=True,
                status=TurnRequestStatus.DONE,
                message=f"Dream step executed (complete={is_complete})",
                turn_id=claimed_turn_id,
            )

        # Priority 3: Auto-analysis check (no active dream)
        if not dreaming_state:
            logger.debug(f"[{turn_id}] Priority 3: Checking if should auto-analyze")
            dream_decision = await worker._decide_dream_action()

            if dream_decision:
                claimed_turn_id = await worker.claim_idle_turn(turn_request)
                turn_id = claimed_turn_id
                logger.info(
                    f"[{turn_id}] Auto-analysis: initiating {dream_decision.scenario} "
                    f"on {dream_decision.conversation_id}"
                )
                new_state = await worker.initialize_auto_dream(dream_decision)
                is_complete = await worker.execute_scenario_step(new_state.pipeline_id)
                return CommandResult(
                    complete=True,
                    status=TurnRequestStatus.DONE,
                    message=f"Dream step executed (complete={is_complete})",
                    turn_id=claimed_turn_id,
                )

        logger.info(f"[{turn_id}] Priority 4: Agent sleeping, skipping phased turn")
        await worker.ensure_turn_id_current(turn_id)
        return CommandResult(
            complete=True,
            status=TurnRequestStatus.DONE,
            message="Agent sleeping",
        )
