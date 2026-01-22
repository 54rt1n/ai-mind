# andimud_worker/commands/choose.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Choose command - @choose turn with user guidance."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDTurnRequest, TurnRequestStatus
from aim_mud_types.decision import DecisionType
from .base import Command
from .result import CommandResult
from ..turns.processor.decision import DecisionProcessor
from ..turns.processor.speaking import SpeakingProcessor
from ..turns.processor.thinking import ThinkingTurnProcessor

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class ChooseCommand(Command):
    """@choose console command - guided turn with user input.

    Uses new decision processor architecture with user guidance injection.
    """

    @property
    def name(self) -> str:
        return "choose"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Process @choose turn with user guidance.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: Contains turn_id, metadata, turn_request

        Returns:
            CommandResult with complete=True
        """

        turn_id = kwargs.get("turn_id", "unknown")

        # Construct MUDTurnRequest from kwargs - Pydantic parses JSON metadata
        turn_request = MUDTurnRequest.model_validate(kwargs)

        # Extract guidance from validated metadata (now a dict)
        guidance = ""
        if turn_request.metadata:
            guidance = (turn_request.metadata.get("guidance", "") or "").strip()

        # Worker has already drained events into worker.pending_events
        events = worker.pending_events

        logger.info(
            "Processing @choose turn %s with %d events, guidance=%s",
            turn_id,
            len(events),
            "yes" if guidance else "no"
        )

        decision = await worker.take_turn(turn_id, events, turn_request, guidance)

        return CommandResult(
            complete=True,
            flush_drain=decision.should_flush if decision else False,
            saved_event_id=None,
            status=TurnRequestStatus.DONE,
            message=f"@choose turn processed: {decision.decision_type.name}"
        )
