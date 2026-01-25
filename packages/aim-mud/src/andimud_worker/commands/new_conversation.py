# andimud_worker/commands/new_conversation.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""New conversation command - set new conversation_id."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDAction, MUDTurnRequest, TurnRequestStatus
from ..conversation.storage import generate_conversation_id
from .base import Command
from .result import CommandResult

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class NewConversationCommand(Command):
    """@new console command - set new conversation_id.

    Extracted from worker.py lines 297-328
    """

    @property
    def name(self) -> str:
        return "new"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Set new conversation_id.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: May contain conversation_id to use (optional)

        Returns:
            CommandResult with complete=True (no fall-through)
        """
        # Use Pydantic to parse metadata JSON
        turn_request = MUDTurnRequest.model_validate(kwargs)
        metadata = turn_request.metadata or {}
        conversation_id = metadata.get("conversation_id", "") if metadata else ""

        if worker.conversation_manager:

            if not conversation_id:
                # Generate new conversation_id
                conversation_id = generate_conversation_id()

            # Re-tag unsaved entries and renumber
            retagged = await worker.conversation_manager.retag_unsaved_entries(conversation_id)

            # Update instance variable for future entries
            worker.conversation_manager.set_conversation_id(conversation_id)

            # Persist to agent profile
            await worker._save_agent_profile()

            logger.info(f"@new: Set conversation_id to {conversation_id}, re-tagged {retagged} entries")

            # Emote completion
            action = MUDAction(
                tool="emote",
                args={"action": f"starts a new conversation thread: {conversation_id}"},
                metadata={"skip_worker": True},
            )
            await worker._emit_actions([action])
        else:
            logger.warning("New conversation requested but no conversation manager")

        return CommandResult(
            complete=True,
            status=TurnRequestStatus.DONE,
            message=f"New conversation: {conversation_id}" if conversation_id else "Failed to create conversation"
        )
