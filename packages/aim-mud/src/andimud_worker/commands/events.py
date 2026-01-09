# andimud_worker/commands/events.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Events command - regular event-driven turn processing."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import TurnRequestStatus
from .base import Command
from .result import CommandResult

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class EventsCommand(Command):
    """Regular event-driven turn processing.

    Extracted from worker.py lines 424-438 (default case)
    """

    @property
    def name(self) -> str:
        return "events"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Process regular event-driven turn.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: Contains turn_id, reason

        Returns:
            CommandResult with complete=False (falls through to process_turn)
        """
        turn_id = kwargs.get("turn_id", "unknown")
        reason = kwargs.get("reason", "events")

        # Worker has already drained events into worker.pending_events
        events = worker.pending_events

        logger.info(
            "Processing assigned turn %s (%s) with %d events",
            turn_id,
            reason,
            len(events),
        )

        # Fall through to process_turn with drained events
        # flush_drain=False keeps events in buffer for process_turn
        return CommandResult(
            complete=False,
            flush_drain=False,
            saved_event_id=None,
            status=TurnRequestStatus.DONE,
            message="Events ready for processing"
        )
