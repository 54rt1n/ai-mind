# andimud_worker/commands/wake.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Wake command - @wake turn that generates a waking up emote."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDTurnRequest, MUDAction, TurnRequestStatus
from .base import Command
from .result import CommandResult


if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)

# Simple prompt for generating a waking up emote
WAKE_PROMPT = """[~~ Wake Turn ~~]

You are waking up from sleep. Output a single brief emote (1-2 sentences) describing how you wake up.
Focus on sensory details - eyes opening, stretching, becoming aware of surroundings.

Output ONLY the emote text, no tags or formatting. Example:
stirs gently, eyes fluttering open as awareness slowly returns.

Your emote:"""


class WakeCommand(Command):
    """@wake command - generate a brief waking up emote.

    Generates a simple emote describing the agent waking up.
    The emote is emitted with sleep_aware metadata for consistency,
    though it's not strictly needed since the agent is now awake.
    """

    @property
    def name(self) -> str:
        return "wake"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Process @wake turn.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: Contains turn_id, metadata, turn_request

        Returns:
            CommandResult with complete=True
        """
        turn_id = kwargs.get("turn_id", "unknown")

        # Parse turn_request
        turn_request = MUDTurnRequest.model_validate(kwargs)
        events = worker.pending_events

        logger.info(
            "[%s] Processing @wake turn with %d pending events",
            turn_id,
            len(events),
        )

        # Setup turn context
        await worker._setup_turn_context(events)

        # Build simple chat turns for emote generation
        chat_turns = await worker._response_strategy.build_turns(
            persona=worker.persona,
            user_input=WAKE_PROMPT,
            session=worker.session,
            coming_online=False,
            max_context_tokens=worker.model.max_tokens,
            max_output_tokens=256,  # Short emote
            memory_query="",
        )

        # Call LLM for emote generation
        response = await worker._call_llm(chat_turns, role="chat")
        emote_text = (response or "").strip()

        # Fallback if LLM fails
        if not emote_text:
            emote_text = "stirs gently and opens eyes, slowly becoming aware of surroundings."

        # Clean up the emote text - remove any leading "I" or quotes
        emote_text = emote_text.strip('"\'')
        if emote_text.lower().startswith("i "):
            emote_text = emote_text[2:]

        logger.info("[%s] Wake emote: %s", turn_id, emote_text[:100])

        # Emit emote action with sleep_aware metadata
        action = MUDAction(
            tool="emote",
            args={"action": emote_text},
            metadata={"sleep_aware": True},
        )
        await worker._emit_actions([action])

        return CommandResult(
            complete=True,
            flush_drain=False,
            saved_event_id=None,
            status=TurnRequestStatus.DONE,
            message=f"Wake turn processed: {emote_text[:50]}...",
        )
