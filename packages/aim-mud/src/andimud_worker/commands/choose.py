# andimud_worker/commands/choose.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Choose command - @choose turn with user guidance."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDTurnRequest, TurnRequestStatus
from aim_mud_types.models.decision import DecisionType
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

        # Events passed via kwargs from main loop
        events = kwargs.get("events", [])

        logger.info(
            "Processing @choose turn %s with %d events, guidance=%s",
            turn_id,
            len(events),
            "yes" if guidance else "no"
        )

        decision = await worker.take_turn(turn_id, events, turn_request, guidance)

        if decision is None:
            error_detail = worker._last_turn_error or "unknown error"
            logger.error("Choose turn %s produced no decision: %s", turn_id, error_detail)
            return CommandResult(
                complete=True,
                status=TurnRequestStatus.FAIL,
                message=f"@choose turn failed: {error_detail}",
            )

        # Get emitted action_ids from worker (set by _emit_actions during take_turn)
        action_ids = worker._last_emitted_action_ids if decision.decision_type not in (DecisionType.WAIT, DecisionType.CONFUSED) else []
        return CommandResult(
            complete=True,
            status=TurnRequestStatus.DONE,
            message=f"@choose turn processed: {decision.decision_type.name}",
            emitted_action_ids=action_ids,
        )
