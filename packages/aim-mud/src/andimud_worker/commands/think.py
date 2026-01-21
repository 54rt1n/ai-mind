# andimud_worker/commands/think.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Think command - @think turn that generates structured reasoning."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDTurnRequest, TurnRequestStatus
from .base import Command
from .result import CommandResult
from ..turns.processor.thinking import ThinkingTurnProcessor


if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class ThinkCommand(Command):
    """@think console command - generate structured reasoning.

    Uses new processor architecture:
    1. Setup turn context
    2. Run ThinkingTurnProcessor directly
    3. Processor handles reasoning generation and emote emission
    """

    @property
    def name(self) -> str:
        return "think"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Process @think turn.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: Contains turn_id, metadata (with optional guidance), turn_request

        Returns:
            CommandResult with complete=True
        """
        turn_id = kwargs.get("turn_id", "unknown")

        # Use Pydantic to parse metadata JSON
        turn_request = MUDTurnRequest.model_validate(kwargs)
        metadata = turn_request.metadata or {}
        guidance = metadata.get("guidance", "") if metadata else ""

        # @think has no new events, uses pending events for context
        events = worker.pending_events

        logger.info(
            "[%s] Processing @think turn with %d events, guidance=%s",
            turn_id,
            len(events),
            "yes" if guidance else "no",
        )

        # Setup turn context
        await worker._setup_turn_context(events)

        # Run ThinkingTurnProcessor directly
        processor = ThinkingTurnProcessor(worker)
        if guidance:
            processor.user_guidance = guidance
        await processor.execute(turn_request, events)

        # Build message
        message = "Think turn processed - reasoning generated"
        if guidance:
            message = f"Think turn with guidance processed: {guidance[:50]}..." if len(guidance) > 50 else f"Think turn with guidance processed: {guidance}"

        return CommandResult(
            complete=True,
            flush_drain=False,
            saved_event_id=None,
            status=TurnRequestStatus.DONE,
            message=message,
        )
