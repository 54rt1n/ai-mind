# andimud_worker/commands/flush.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Flush command - write conversation to CVM."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDAction, TurnRequestStatus
from .base import Command
from .result import CommandResult

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class FlushCommand(Command):
    """@write console command - flush conversation to CVM.

    Extracted from worker.py lines 272-286
    """

    @property
    def name(self) -> str:
        return "flush"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Flush conversation manager entries to CVM.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: Unused for this command

        Returns:
            CommandResult with complete=True (no fall-through)
        """
        if worker.conversation_manager:
            flushed = await worker.conversation_manager.flush_to_cvm(worker.cvm)
            logger.info(f"@write: Flushed {flushed} entries to CVM")
            # Update conversation report
            await worker._update_conversation_report()
            # Emote completion
            action = MUDAction(tool="emote", args={"action": "feels more knowledgeable."})
            await worker._emit_actions([action])
        else:
            logger.warning("Flush requested but no conversation manager")

        return CommandResult(
            complete=True,
            flush_drain=False,
            saved_event_id=None,
            status=TurnRequestStatus.DONE,
            message="Conversation flushed to memory"
        )
