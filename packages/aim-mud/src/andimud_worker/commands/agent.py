# andimud_worker/commands/agent.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Agent command - @agent turn with action spec."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDTurnRequest, TurnRequestStatus
from .base import Command
from .result import CommandResult
from ..turns.processor.agent import AgentTurnProcessor

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class AgentCommand(Command):
    """@agent console command - process guided action turn.

    Uses new processor architecture:
    1. Setup turn context
    2. Run AgentTurnProcessor directly with guidance and tool
    3. Processor handles tool execution and response
    """

    @property
    def name(self) -> str:
        return "agent"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Process @agent turn with guidance.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: Contains turn_id, metadata, turn_request

        Returns:
            CommandResult with complete=True, flush_drain=False
        """

        # Get turn_request directly from kwargs
        turn_request = kwargs.get("turn_request")
        if not turn_request:
            # Fallback for backward compatibility
            turn_request = MUDTurnRequest.model_validate(kwargs)

        turn_id = turn_request.turn_id or "unknown"

        # Extract guidance and tool from validated metadata (now a dict)
        guidance = ""
        required_tool = ""
        if turn_request.metadata:
            guidance = (turn_request.metadata.get("guidance", "") or "").strip()
            required_tool = (turn_request.metadata.get("tool", "") or "").strip()

        # Events passed via kwargs from main loop
        events = kwargs.get("events", [])

        logger.info(
            "Processing @agent turn %s with tool=%s, %d events",
            turn_id,
            required_tool or "(any)",
            len(events),
        )

        # Setup turn context
        await worker._setup_turn_context(events)

        # Capture new conversation entries for embedding-based reranking
        new_entries = await worker.get_new_conversation_entries()
        new_entries = worker.collapse_consecutive_entries(new_entries)
        worker._current_turn_entries = new_entries
        if new_entries:
            logger.info(f"[{turn_id}] Captured {len(new_entries)} entries for @agent turn")

        # Run AgentTurnProcessor directly
        processor = AgentTurnProcessor.from_config(worker, worker.chat_config, worker.config)
        processor.user_guidance = guidance
        processor.required_tool = required_tool
        await processor.execute(turn_request, events)

        # Get emitted action_ids from worker (set by _emit_actions during processor execution)
        action_ids = worker._last_emitted_action_ids

        # Agent turns are memory palace actions, outside MUD world
        # Events are environmental context only, don't flush drain
        return CommandResult(
            complete=True,
            status=TurnRequestStatus.DONE,
            message=f"Agent turn processed: {required_tool or 'any tool'}",
            emitted_action_ids=action_ids,
        )
