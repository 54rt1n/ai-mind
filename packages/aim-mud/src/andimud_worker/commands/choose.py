# andimud_worker/commands/choose.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Choose command - @choose turn with user guidance."""

import logging
from typing import TYPE_CHECKING

from .base import Command
from .result import CommandResult

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class ChooseCommand(Command):
    """@choose console command - guided turn with user input.

    Extracted from worker.py lines 344-358
    """

    @property
    def name(self) -> str:
        return "choose"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Process @choose turn with user guidance.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: Contains turn_id, guidance

        Returns:
            CommandResult with complete=True, flush_drain=True
        """
        turn_id = kwargs.get("turn_id", "unknown")
        guidance = kwargs.get("guidance", "")

        # Worker has already drained events into worker.pending_events
        events = worker.pending_events

        logger.info(
            "Processing @choose turn %s with %d events",
            turn_id,
            len(events),
        )
        # Use standard phased turn with user guidance
        await worker.process_turn(events, user_guidance=guidance)

        return CommandResult(
            complete=True,
            flush_drain=True,
            saved_event_id=None,
            status="done",
            message="Choose turn processed"
        )
