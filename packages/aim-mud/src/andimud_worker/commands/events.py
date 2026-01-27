# andimud_worker/commands/events.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Events command - regular event-driven turn processing."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDTurnRequest, TurnRequestStatus, EventType
from aim_mud_types.client import RedisMUDClient
from aim_mud_types.models.decision import DecisionType
from .base import Command
from .result import CommandResult

from ..turns.processor.decision import DecisionProcessor
from ..turns.processor.speaking import SpeakingProcessor
from ..turns.processor.thinking import ThinkingTurnProcessor


if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class EventsCommand(Command):
    """Regular event-driven turn processing.

    Uses new decision processor architecture:
    1. Setup turn context
    2. Run DecisionProcessor for Phase 1
    3. Route based on DecisionType to appropriate processor
    """

    @property
    def name(self) -> str:
        return "events"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Process regular event-driven turn.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: Contains turn_id, reason, turn_request

        Returns:
            CommandResult with complete=True
        """
        # Get turn_request directly from kwargs
        turn_request = kwargs.get("turn_request")
        if not turn_request:
            # Fallback for backward compatibility
            turn_request = MUDTurnRequest.model_validate(kwargs)

        turn_id = turn_request.turn_id or "unknown"
        reason = turn_request.reason or "events"

        # Events passed via kwargs from main loop
        events = kwargs.get("events", [])

        logger.info(
            "Processing assigned turn %s (%s) with %d events",
            turn_id,
            reason,
            len(events),
        )

        # Check if sleeping before processing events
        is_sleeping = await worker._check_agent_is_sleeping()

        if is_sleeping:
            logger.info(f"[{turn_id}] Agent sleeping, skipping event processing")
            return CommandResult(
                complete=True,
                status=TurnRequestStatus.DONE,
                message="Agent sleeping",
            )

        # Check if ALL events are non-reactive (awareness-only events)
        # These events are drained to conversation but don't trigger a turn
        # This prevents ping-pong where agents react to each other's idle emotes
        # and ensures terminal/code output doesn't cascade to other agents
        NON_REACTIVE_TYPES = (
            EventType.NON_REACTIVE,    # Idle emotes, context-building actions
            EventType.NON_PUBLISHED,   # Not routed to streams at all
            EventType.TERMINAL,        # Terminal output (targeted to caller)
            EventType.CODE_FILE,       # Code file output (targeted to caller)
            EventType.CODE_ACTION,     # Code action output (targeted to caller)
        )
        reactive_events = [e for e in events if e.event_type not in NON_REACTIVE_TYPES]
        if events and not reactive_events:
            logger.info(
                f"[{turn_id}] Only non-reactive events ({len(events)}), "
                "pushing to conversation without taking turn"
            )
            # Push non-reactive events to conversation for context
            await worker._push_events_to_conversation(events)
            return CommandResult(
                complete=True,
                status=TurnRequestStatus.DONE,
                message=f"Non-reactive events only ({len(events)})",
            )

        decision = await worker.take_turn(turn_id, events, turn_request)

        if decision is None:
            error_detail = worker._last_turn_error or "unknown error"
            logger.error("Event turn %s produced no decision: %s", turn_id, error_detail)
            return CommandResult(
                complete=True,
                status=TurnRequestStatus.FAIL,
                message=f"Turn failed: {error_detail}",
            )

        # Get emitted action_ids from worker (set by _emit_actions during take_turn)
        action_ids = worker._last_emitted_action_ids if decision.decision_type not in (DecisionType.WAIT, DecisionType.CONFUSED) else []
        return CommandResult(
            complete=True,
            status=TurnRequestStatus.DONE,
            message=f"Turn processed: {decision.decision_type.name}",
            emitted_action_ids=action_ids,
        )
