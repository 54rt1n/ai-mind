# andimud_worker/commands/idle.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Idle command - check auto-dream triggers and process idle turn."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import TurnRequestStatus
from .base import Command
from .result import CommandResult

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class IdleCommand(Command):
    """Idle turn - check auto-dream triggers.

    Extracted from worker.py lines 399-423
    """

    @property
    def name(self) -> str:
        return "idle"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Process idle turn, checking for auto-dream triggers.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: Contains turn_id

        Returns:
            CommandResult with complete=False (falls through to process_turn)
        """
        turn_id = kwargs.get("turn_id", "unknown")

        # Check auto-dream triggers during idle
        dream_request = await worker.check_auto_dream_triggers()
        if dream_request:
            logger.info(f"Auto-dream triggered: {dream_request.scenario}")
            result = await worker.process_dream_turn(
                scenario=dream_request.scenario,
                query=dream_request.query,
                guidance=dream_request.guidance,
                triggered_by="auto",
            )
            if result.success:
                logger.info(
                    f"Auto-dream completed: {result.pipeline_id} "
                    f"in {result.duration_seconds:.1f}s"
                )
                # Update conversation report
                await worker._update_conversation_report()
            else:
                logger.warning(f"Auto-dream failed: {result.error}")

        # Fall through to process_turn with empty events
        # flush_drain=True tells worker to clear pending_events
        return CommandResult(
            complete=False,
            flush_drain=True,
            saved_event_id=None,
            status=TurnRequestStatus.DONE,
            message="Idle turn ready"
        )
