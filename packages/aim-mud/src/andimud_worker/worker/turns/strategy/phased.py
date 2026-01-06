# aim/app/mud/worker/turns/strategy/phased.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Phased turn processor: decision phase → conditional response phase."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDAction
from aim.dreamer.executor import extract_think_tags
from ....adapter import build_current_context
from ....session import MUDEvent
from ....utils import sanitize_response
from ..response import (
    normalize_response,
    has_emotional_state_header,
    extract_speak_text_from_tool_call,
)
from .base import BaseTurnProcessor

if TYPE_CHECKING:
    from ..orchestrator import TurnsMixin

logger = logging.getLogger(__name__)


class PhasedTurnProcessor(BaseTurnProcessor):
    """Two-phase turn processing strategy.

    Phase 1 (Decision): Call LLM with TOOL role to choose action (fast, cheap)
    - Returns: move, take, drop, give, speak, wait, or confused
    - Immediately emits physical actions
    - If decision is "speak" → proceed to Phase 2

    Phase 2 (Response): Call LLM with CHAT role for full narrative (expensive)
    - Only if Phase 1 decided to speak
    - Includes full memory context
    - Validates emotional state header
    """

    def __init__(self, worker: "TurnsMixin"):
        """Initialize with worker and set user_guidance to empty string.

        Args:
            worker: MUDAgentWorker instance
        """
        super().__init__(worker)
        self.user_guidance = ""

    async def _decide_action(self, events: list[MUDEvent]) -> tuple[list[MUDAction], str]:
        """Execute phased decision strategy.

        Args:
            events: List of events to process

        Returns:
            Tuple of (actions_taken, thinking)
        """
        idle_mode = len(events) == 0
        thinking_parts: list[str] = []
        actions_taken: list[MUDAction] = []

        try:
            # Check for abort before decision LLM call
            if await self.worker._check_abort_requested():
                from ..orchestrator import AbortRequestedException
                raise AbortRequestedException("Turn aborted before decision")

            # Phase 1: Decision (use tool role - fast)
            decision_tool, decision_args, decision_raw, decision_thinking, decision_cleaned = (
                await self.worker._decide_action(
                    idle_mode=idle_mode,
                    role="tool",
                    action_guidance=self._action_guidance,
                    user_guidance=self.user_guidance,
                )
            )
            if decision_thinking:
                thinking_parts.append(decision_thinking)

            if decision_tool == "move":
                action = MUDAction(tool="move", args=decision_args)
                actions_taken.append(action)
                await self.worker._emit_actions(actions_taken)

            elif decision_tool == "take":
                obj = decision_args.get("object")
                if obj:
                    action = MUDAction(tool="get", args={"object": obj})
                    actions_taken.append(action)
                    await self.worker._emit_actions(actions_taken)
                else:
                    logger.warning("Phase1 take missing object; no action emitted")

            elif decision_tool == "drop":
                obj = decision_args.get("object")
                if obj:
                    action = MUDAction(tool="drop", args={"object": obj})
                    actions_taken.append(action)
                    await self.worker._emit_actions(actions_taken)
                else:
                    logger.warning("Phase1 drop missing object; no action emitted")

            elif decision_tool == "give":
                obj = decision_args.get("object")
                target = decision_args.get("target")
                if obj and target:
                    action = MUDAction(tool="give", args={"object": obj, "target": target})
                    actions_taken.append(action)
                    await self.worker._emit_actions(actions_taken)
                else:
                    logger.warning("Phase1 give missing object or target; no action emitted")

            elif decision_tool == "wait":
                logger.info("Phase 1 decided to wait; no action this turn")

            elif decision_tool == "speak":
                # Phase 2: full response turn with memory via response strategy
                coming_online = await self.worker._is_fresh_session()

                # Extract memory query from speak args (enhances CVM search)
                memory_query = decision_args.get("query") or decision_args.get("focus") or ""
                if memory_query:
                    logger.info(f"Phase 2 memory query: {memory_query[:100]}...")

                # Build user input with current context (events/guidance)
                user_input = build_current_context(
                    self.worker.session,
                    idle_mode=idle_mode,
                    guidance=None,
                    coming_online=coming_online,
                    include_events=False,
                    action_guidance=self._action_guidance,
                )

                # Use response strategy for full context (consciousness + memory)
                chat_turns = await self.worker._response_strategy.build_turns(
                    persona=self.worker.persona,
                    user_input=user_input,
                    session=self.worker.session,
                    coming_online=coming_online,
                    max_context_tokens=self.worker.model.max_tokens,
                    max_output_tokens=self.worker.chat_config.max_tokens,
                    memory_query=memory_query,
                )

                # Retry loop for emotional state header validation
                max_format_retries = 3
                cleaned_response = ""
                for format_attempt in range(max_format_retries):
                    # Check for abort before response LLM call
                    if await self.worker._check_abort_requested():
                        from ..orchestrator import AbortRequestedException
                        raise AbortRequestedException("Turn aborted before response")

                    # Phase 2: Response (use chat role - general chat model)
                    response = await self.worker._call_llm(chat_turns, role="chat")
                    logger.debug(f"LLM response: {response[:500]}...")
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("LLM response (full):\n%s", response)

                    cleaned_response, think_content = extract_think_tags(response)
                    cleaned_response = sanitize_response(cleaned_response)
                    cleaned_response = cleaned_response.strip()
                    if think_content:
                        thinking_parts.append(think_content)

                    # Validate emotional state header
                    if has_emotional_state_header(cleaned_response):
                        break  # Valid format, continue

                    # Missing header - retry with stronger guidance
                    logger.warning(
                        f"Response missing Emotional State header (attempt {format_attempt + 1}/{max_format_retries})"
                    )
                    if format_attempt < max_format_retries - 1:
                        persona_name = self.worker.session.persona_id if self.worker.session else "Agent"
                        format_guidance = (
                            f"\n\n[Gentle reminder from your link: Please begin with your emotional state, "
                            f"e.g. [== {persona_name}'s Emotional State: <list of your +Emotions+> ==] then continue with prose.]"
                        )
                        if chat_turns and chat_turns[-1]["role"] == "user":
                            chat_turns[-1]["content"] += format_guidance
                        else:
                            chat_turns.append({"role": "user", "content": format_guidance})

                extracted_text = extract_speak_text_from_tool_call(cleaned_response)
                if extracted_text is not None:
                    logger.debug(
                        "Phase2 response looked like a tool call; extracted speak text (%d chars)",
                        len(extracted_text),
                    )
                normalized = normalize_response(
                    extracted_text if extracted_text is not None else cleaned_response
                )

                if normalized:
                    action = MUDAction(tool="speak", args={"text": normalized})
                    actions_taken.append(action)
                    logger.info("Prepared speak action (%d chars)", len(normalized))
                    await self.worker._emit_actions(actions_taken)
                else:
                    logger.info("No response content to emit")

            elif decision_tool == "confused":
                # Phase 1 failed to parse a valid decision - emit confused emote
                logger.info("Phase 1 returned confused; emitting confused emote")
                action = MUDAction(tool="emote", args={"action": "looks confused."})
                actions_taken.append(action)
                await self.worker._emit_actions(actions_taken)

            else:
                # Unknown decision tool - log warning and skip
                logger.warning(
                    "Unknown phase 1 decision tool '%s'; skipping turn",
                    decision_tool,
                )

        except Exception as e:
            logger.error(f"Error during turn processing: {e}", exc_info=True)
            thinking_parts.append(f"[ERROR] Turn processing failed: {e}")
            # Emit a graceful emote when LLM fails
            action = MUDAction(tool="emote", args={"action": "was at a loss for words."})
            actions_taken.append(action)
            await self.worker._emit_actions(actions_taken)

        thinking = "\n\n".join(thinking_parts).strip()
        return actions_taken, thinking
