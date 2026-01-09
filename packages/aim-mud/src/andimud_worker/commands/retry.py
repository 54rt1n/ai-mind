# andimud_worker/commands/retry.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Retry command - retry a failed turn after exponential backoff."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import TurnRequestStatus
from .base import Command
from .result import CommandResult

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class RetryCommand(Command):
    """Retry a failed turn after exponential backoff.

    The mediator assigns this when a failed turn's next_attempt_at
    timestamp has been reached. The worker re-processes the turn
    through the normal pipeline.
    """

    @property
    def name(self) -> str:
        return "retry"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Process retry turn.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: Contains turn_id, attempt_count, status_reason, etc.

        Returns:
            CommandResult with complete=False (falls through to process_turn)
        """
        turn_id = kwargs.get("turn_id", "unknown")
        attempt_count = kwargs.get("attempt_count", "1")
        status_reason = kwargs.get("status_reason", "")

        logger.info(
            "Retrying turn %s (attempt %s) - previous failure: %s",
            turn_id,
            attempt_count,
            status_reason,
        )

        # Fall through to normal turn processing
        # The worker's exception handler manages attempt_count and backoff
        return CommandResult(
            complete=False,
            flush_drain=False,
            saved_event_id=None,
            status=TurnRequestStatus.IN_PROGRESS,
            message=f"Retry attempt {attempt_count}"
        )
