# andimud_worker/commands/dream.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Dream command - run dreamer pipeline."""

import logging
from typing import TYPE_CHECKING

from .base import Command
from .result import CommandResult

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class DreamCommand(Command):
    """@dream console command - run a dreamer pipeline.

    Extracted from worker.py lines 359-398
    """

    @property
    def name(self) -> str:
        return "dream"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Execute dream pipeline.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: Contains turn_id, scenario, query, guidance, conversation_id

        Returns:
            CommandResult with complete=True, status="done" or "fail"
        """
        turn_id = kwargs.get("turn_id", "unknown")
        scenario = kwargs.get("scenario", "")
        query = kwargs.get("query") or None
        guidance = kwargs.get("guidance") or None
        # Explicit conversation_id for analysis commands
        target_conversation_id = kwargs.get("conversation_id") or None

        if not scenario:
            logger.error("Dream turn missing scenario")
            return CommandResult(
                complete=True,
                flush_drain=False,
                saved_event_id=None,
                status="fail",
                message="Dream turn missing scenario"
            )

        logger.info(f"Processing dream turn: {scenario}")
        result = await worker.process_dream_turn(
            scenario=scenario,
            query=query,
            guidance=guidance,
            triggered_by="manual",
            target_conversation_id=target_conversation_id,
        )

        if result.success:
            logger.info(
                f"Dream completed: {result.pipeline_id} "
                f"in {result.duration_seconds:.1f}s"
            )
            # Update conversation report
            await worker._update_conversation_report()
            return CommandResult(
                complete=True,
                flush_drain=False,
                saved_event_id=None,
                status="done",
                message=f"Dream completed: {scenario}"
            )
        else:
            logger.error(f"Dream failed: {result.error}")
            return CommandResult(
                complete=True,
                flush_drain=False,
                saved_event_id=None,
                status="fail",
                message=result.error or "Dream failed"
            )
