# andimud_worker/commands/retry.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Retry command - retry a failed turn after exponential backoff."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import TurnRequestStatus, MUDTurnRequest
from aim_mud_types.models.decision import DecisionType
from .base import Command
from .result import CommandResult
from ..turns.processor.decision import DecisionProcessor
from ..turns.processor.speaking import SpeakingProcessor
from ..turns.processor.thinking import ThinkingTurnProcessor

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class RetryCommand(Command):
    """Retry a failed turn after exponential backoff.

    The mediator assigns this when a failed turn's next_attempt_at
    timestamp has been reached. The worker re-processes the turn
    through the normal pipeline using decision processor architecture.
    """

    @property
    def name(self) -> str:
        return "retry"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Process retry turn.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: Contains turn_id, attempt_count, status_reason, turn_request

        Returns:
            CommandResult with complete=True
        """
        # Get turn_request directly from kwargs
        turn_request = kwargs.get("turn_request")
        if not turn_request:
            # Fallback for backward compatibility
            turn_request = MUDTurnRequest.model_validate(kwargs)

        turn_id = turn_request.turn_id or "unknown"
        attempt_count = turn_request.attempt_count or 1
        status_reason = turn_request.status_reason or ""

        logger.info(
            "Retrying turn %s (attempt %s) - previous failure: %s",
            turn_id,
            attempt_count,
            status_reason,
        )

        # Check if sleeping before processing retry
        is_sleeping = await worker._check_agent_is_sleeping()

        if is_sleeping:
            logger.info(f"[{turn_id}] Agent sleeping, skipping retry")
            return CommandResult(
                complete=True,
                status=TurnRequestStatus.DONE,
                message="Agent sleeping",
            )

        # Events passed via kwargs from main loop
        events = kwargs.get("events", [])

        decision = await worker.take_turn(turn_id, events, turn_request)
        decision_type = getattr(decision, "decision_type", None)

        if not decision_type:
            error_detail = worker._last_turn_error or "decision_type missing"
            logger.error(
                "Retry failed for turn %s (attempt %s): %s",
                turn_id,
                attempt_count,
                error_detail,
            )
            return CommandResult(
                complete=True,
                status=TurnRequestStatus.FAIL,
                message=f"Retry failed (attempt {attempt_count}): {error_detail}",
            )

        # Get emitted action_ids from worker (set by _emit_actions during take_turn)
        # All decision types emit actions (including WAIT and CONFUSED with non-published emotes)
        action_ids = worker._last_emitted_action_ids
        expects_echo = worker._last_emitted_expects_echo
        return CommandResult(
            complete=True,
            status=TurnRequestStatus.DONE,
            message=f"Retry successful (attempt {attempt_count}): {decision_type.name}",
            emitted_action_ids=action_ids,
            expects_echo=expects_echo,
        )
