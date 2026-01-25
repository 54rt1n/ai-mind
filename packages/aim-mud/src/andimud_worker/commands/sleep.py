# andimud_worker/commands/sleep.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Sleep command - @sleep turn that generates a falling asleep emote."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDTurnRequest, MUDAction, TurnRequestStatus
from aim.utils.think import extract_think_tags
from .base import Command
from .result import CommandResult


if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)

# Simple prompt for generating a falling asleep emote
SLEEP_PROMPT = """[~~ Sleep Turn ~~]

You are falling asleep. Output a single brief emote (1-2 sentences) describing how you drift off to sleep.
Focus on sensory details - eyes closing, breathing slowing, settling into comfort.

Output ONLY the emote text, no tags or formatting. Example:
settles into a comfortable position, eyes growing heavy as consciousness gently fades.

Your emote:"""


class SleepCommand(Command):
    """@sleep command - generate a brief falling asleep emote.

    Generates a simple emote describing the agent falling asleep.
    The emote is emitted with sleep_aware metadata so it bypasses
    sleep filtering and appears in the agent's conversation history.
    """

    @property
    def name(self) -> str:
        return "sleep"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Process @sleep turn.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: Contains turn_id, metadata, turn_request

        Returns:
            CommandResult with complete=True
        """
        turn_id = kwargs.get("turn_id", "unknown")

        # Parse turn_request
        turn_request = MUDTurnRequest.model_validate(kwargs)
        events = kwargs.get("events", [])

        logger.info(
            "[%s] Processing @sleep turn with %d pending events",
            turn_id,
            len(events),
        )

        # Setup turn context
        await worker._setup_turn_context(events)

        # Build simple chat turns for emote generation
        chat_turns = await worker._response_strategy.build_turns(
            persona=worker.persona,
            user_input=SLEEP_PROMPT,
            session=worker.session,
            coming_online=False,
            max_context_tokens=worker.model.max_tokens,
            max_output_tokens=256,  # Short emote
            memory_query="",
        )

        # Call LLM for emote generation
        response = await worker._call_llm(chat_turns, role="chat")
        raw_response = (response or "").strip()
        emote_text, _think_content = extract_think_tags(raw_response)
        emote_text = emote_text.strip()

        # Fallback if LLM fails
        if not emote_text:
            emote_text = "settles into a comfortable position and drifts off to sleep."

        # Clean up the emote text - remove any leading "I" or quotes
        emote_text = emote_text.strip('"\'')
        if emote_text.lower().startswith("i "):
            emote_text = emote_text[2:]

        logger.info("[%s] Sleep emote: %s", turn_id, emote_text[:100])

        # Emit emote action with sleep_aware metadata
        action = MUDAction(
            tool="emote",
            args={"action": emote_text},
            metadata={MUDAction.META_SLEEP_AWARE: True},
        )
        action_ids, expects_echo = await worker._emit_actions([action])

        return CommandResult(
            complete=True,
            status=TurnRequestStatus.DONE,
            message=f"Sleep turn processed: {emote_text[:50]}...",
            emitted_action_ids=action_ids,
            expects_echo=expects_echo,
        )
