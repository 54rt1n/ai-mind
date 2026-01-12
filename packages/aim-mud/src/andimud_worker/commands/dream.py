# andimud_worker/commands/dream.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Dream command - run dreamer pipeline."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import TurnRequestStatus
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
            **kwargs: Contains turn_id, metadata (with scenario, query, guidance, conversation_id)

        Returns:
            CommandResult with complete=True, status=TurnRequestStatus.DONE or "fail"
        """
        turn_id = kwargs.get("turn_id", "unknown")

        # Read from metadata
        metadata = kwargs.get("metadata") or {}
        scenario = metadata.get("scenario", "")
        query = metadata.get("query")
        guidance = metadata.get("guidance")
        # Explicit conversation_id for analysis commands
        target_conversation_id = metadata.get("conversation_id")

        if not scenario:
            logger.error("Dream turn missing scenario in metadata")
            return CommandResult(
                complete=True,
                flush_drain=False,
                saved_event_id=None,
                status=TurnRequestStatus.FAIL,
                message="Dream turn missing scenario in metadata"
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
            # Refresh index and update conversation report
            worker.cvm.refresh()
            await worker._update_conversation_report()
            return CommandResult(
                complete=True,
                flush_drain=False,
                saved_event_id=None,
                status=TurnRequestStatus.DONE,
                message=f"Dream completed: {scenario}"
            )
        else:
            logger.error(f"Dream failed: {result.error}")
            return CommandResult(
                complete=True,
                flush_drain=False,
                saved_event_id=None,
                status=TurnRequestStatus.FAIL,
                message=result.error or "Dream failed"
            )
