# andimud_worker/commands/clear.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Clear command - clear conversation history."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import TurnRequestStatus
from .base import Command
from .result import CommandResult

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class ClearCommand(Command):
    """@clear console command - clear conversation history.

    Extracted from worker.py lines 287-296
    """

    @property
    def name(self) -> str:
        return "clear"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Clear conversation history.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: Unused for this command

        Returns:
            CommandResult with complete=True (no fall-through)
        """
        if worker.conversation_manager:
            await worker.conversation_manager.clear()
            logger.info("@clear: Cleared conversation history")
        else:
            logger.warning("Clear requested but no conversation manager")

        return CommandResult(
            complete=True,
            status=TurnRequestStatus.DONE,
            message="Messages cleared"
        )
