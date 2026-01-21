# andimud_worker/commands/agent.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Agent command - @agent turn with action spec."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDTurnRequest, TurnRequestStatus
from .base import Command
from .result import CommandResult

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class AgentCommand(Command):
    """@agent console command - process guided action turn.

    Extracted from worker.py lines 329-343
    """

    @property
    def name(self) -> str:
        return "agent"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Process @agent turn with guidance.

        Args:
            worker: MUDAgentWorker instance
        **kwargs: Contains turn_id, metadata, sequence_id, attempt_count, etc.

        Returns:
            CommandResult with complete=True, flush_drain=True
        """
        turn_id = kwargs.get("turn_id", "unknown")

        # Construct MUDTurnRequest from kwargs - Pydantic parses JSON metadata
        turn_request = MUDTurnRequest.model_validate(kwargs)

        # Extract guidance and tool from validated metadata (now a dict)
        guidance = ""
        required_tool = ""
        if turn_request.metadata:
            guidance = (turn_request.metadata.get("guidance", "") or "").strip()
            required_tool = (turn_request.metadata.get("tool", "") or "").strip()

        # Worker has already drained events into worker.pending_events
        events = worker.pending_events

        logger.info(
            "Processing @agent turn %s with tool=%s, %d events",
            turn_id,
            required_tool or "(any)",
            len(events),
        )
        await worker.process_agent_turn(turn_request, events, guidance, required_tool)

        # Agent turns are memory palace actions, outside MUD world
        # Events are environmental context only, don't flush drain
        return CommandResult(
            complete=True,
            flush_drain=False,
            saved_event_id=None,
            status=TurnRequestStatus.DONE,
            message="Agent turn processed"
        )
