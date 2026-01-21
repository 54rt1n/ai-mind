# andimud_worker/commands/events.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Events command - regular event-driven turn processing."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import TurnRequestStatus
from aim_mud_types.decision import DecisionType
from .base import Command
from .result import CommandResult
from .helpers import setup_turn_context

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
        from ..turns.processor.decision import DecisionProcessor
        from ..turns.processor.speaking import SpeakingProcessor
        from ..turns.processor.thinking import ThinkingTurnProcessor

        turn_id = kwargs.get("turn_id", "unknown")
        reason = kwargs.get("reason", "events")
        turn_request = kwargs.get("turn_request")

        # Worker has already drained events into worker.pending_events
        events = worker.pending_events

        logger.info(
            "Processing assigned turn %s (%s) with %d events",
            turn_id,
            reason,
            len(events),
        )

        # Check if sleeping before processing events
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(worker.redis)
        is_sleeping = await client.get_agent_is_sleeping(worker.config.agent_id)

        if is_sleeping:
            logger.info(f"[{turn_id}] Agent sleeping, skipping event processing")
            return CommandResult(
                complete=True,
                flush_drain=True,
                saved_event_id=None,
                status=TurnRequestStatus.DONE,
                message="Agent sleeping",
            )

        # Setup turn context ONCE
        await setup_turn_context(worker, events)

        # Run DecisionProcessor for Phase 1
        decision_processor = DecisionProcessor(worker)
        await decision_processor.execute(turn_request, events)

        # Route based on decision type
        decision = worker._last_decision

        if decision.decision_type == DecisionType.SPEAK:
            speaking_processor = SpeakingProcessor(worker)
            await speaking_processor.execute(turn_request, events)
        elif decision.decision_type == DecisionType.THINK:
            thinking_processor = ThinkingTurnProcessor(worker)
            await thinking_processor.execute(turn_request, events)
        else:
            # Direct action (move, take, drop, give, emote, wait, etc.)
            await worker._emit_decision_action(decision)

        # Clear decision
        worker._last_decision = None

        return CommandResult(
            complete=True,
            flush_drain=False,
            saved_event_id=None,
            status=TurnRequestStatus.DONE,
            message=f"Turn processed: {decision.decision_type.name}"
        )
