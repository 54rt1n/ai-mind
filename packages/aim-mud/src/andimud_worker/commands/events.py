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
        turn_id = kwargs.get("turn_id", "unknown")
        reason = kwargs.get("reason", "events")
        turn_request = MUDTurnRequest.model_validate(kwargs)

        # Worker has already drained events into worker.pending_events
        events = worker.pending_events

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
                flush_drain=True,
                saved_event_id=None,
                status=TurnRequestStatus.DONE,
                message="Agent sleeping",
            )

        # Check if ALL events are NON_REACTIVE (awareness-only events)
        # These events are drained/consumed but don't trigger a turn
        # This prevents ping-pong where agents react to each other's idle emotes
        reactive_events = [e for e in events if e.event_type not in (EventType.NON_REACTIVE, EventType.NON_PUBLISHED)]
        if events and not reactive_events:
            logger.info(
                f"[{turn_id}] Only NON_REACTIVE events ({len(events)}), "
                "draining without taking turn"
            )
            return CommandResult(
                complete=True,
                flush_drain=True,  # Consume the events
                saved_event_id=None,
                status=TurnRequestStatus.DONE,
                message=f"Non-reactive events only ({len(events)})",
            )

        decision = await worker.take_turn(turn_id, events, turn_request)

        if decision is None:
            error_detail = worker._last_turn_error or "unknown error"
            logger.error("Event turn %s produced no decision: %s", turn_id, error_detail)
            return CommandResult(
                complete=True,
                flush_drain=False,
                saved_event_id=None,
                status=TurnRequestStatus.FAIL,
                message=f"Turn failed: {error_detail}",
            )

        return CommandResult(
            complete=True,
            flush_drain=decision.should_flush,
            saved_event_id=None,
            status=TurnRequestStatus.DONE,
            message=f"Turn processed: {decision.decision_type.name}",
        )
