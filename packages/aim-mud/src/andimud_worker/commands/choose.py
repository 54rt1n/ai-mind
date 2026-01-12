# andimud_worker/commands/choose.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Choose command - @choose turn with user guidance."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import TurnRequestStatus
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
            **kwargs: Contains turn_id, metadata, sequence_id, attempt_count, etc.

        Returns:
            CommandResult with complete=False (falls through to process_turn)
        """
        turn_id = kwargs.get("turn_id", "unknown")

        logger.info(
            "Processing @choose turn %s with %d events",
            turn_id,
            len(worker.pending_events),
        )

        return CommandResult(
            complete=False,
            flush_drain=False,
            saved_event_id=None,
            status=TurnRequestStatus.DONE,
            message="Choose turn ready"
        )
