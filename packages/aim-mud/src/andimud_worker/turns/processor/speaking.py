# aim/app/mud/worker/turns/strategy/speaking.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Speaking turn processor: Phase 2 full narrative with memory.

Called after DecisionProcessor returns SPEAK decision.
Builds full context with CVM memory retrieval and generates narrative response.
"""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDAction, MUDEvent, MUDTurnRequest
from aim.utils.think import extract_think_tags
from ...adapter import build_current_context
from ..response import (
    sanitize_response,
    normalize_response,
    has_emotional_state_header,
)
from .base import BaseTurnProcessor
from ...exceptions import AbortRequestedException

if TYPE_CHECKING:
    from ...mixins.turns import TurnsMixin

logger = logging.getLogger(__name__)


class SpeakingProcessor(BaseTurnProcessor):
    """Phase 2: Full narrative speaking processor.

    Called after DecisionProcessor returns SPEAK decision.
    Builds full context with memory and generates narrative response.
    """

    def __init__(self, worker: "TurnsMixin"):
        """Initialize with worker and set memory query to empty string.

        Args:
            worker: MUDAgentWorker instance
        """
        super().__init__(worker)
        self.memory_query = ""  # From Phase 1 speak args
        self.user_guidance = ""  # Optional user guidance

    async def _decide_action(
        self,
        turn_request: MUDTurnRequest,
        events: list[MUDEvent]
    ) -> tuple[list[MUDAction], str]:
        """Generate full narrative response with memory.

        Extracts memory query from Phase 1 decision, builds full context with
        response strategy (including CVM memory retrieval), and calls LLM with
        chat role. Validates emotional state header with retry loop (3 chat + 1 fallback).

        Args:
            turn_request: Current turn request with sequence_id
            events: List of events to process

        Returns:
            Tuple of (actions_taken, thinking_text)
        """
        idle_mode = len(events) == 0
        thinking_parts: list[str] = []
        actions_taken: list[MUDAction] = []

        try:
            # Phase 2: full response turn with memory via response strategy
            coming_online = await self.worker._is_fresh_session()

            # Memory query should be set by caller from Phase 1 decision args
            if self.memory_query:
                logger.info(f"Phase 2 memory query: {self.memory_query[:100]}...")

            # Build guidance for Phase 2
            guidance_parts: list[str] = []
            if self.memory_query:
                guidance_parts.append(f"Speech focus: {self.memory_query}")
            if self.user_guidance:
                guidance_parts.append(self.user_guidance)
            phase2_guidance = "\n".join(guidance_parts).strip() if guidance_parts else None

            # Build user input with current context (events/guidance)
            user_input = build_current_context(
                self.worker.session,
                idle_mode=idle_mode,
                guidance=phase2_guidance,
                coming_online=coming_online,
                include_events=False,
            )

            # Get pre-computed embedding for FAISS reranking
            query_embedding = self.worker.get_current_turn_embedding()

            # Use response strategy for full context (consciousness + memory)
            chat_turns = await self.worker._response_strategy.build_turns(
                persona=self.worker.persona,
                user_input=user_input,
                session=self.worker.session,
                coming_online=coming_online,
                max_context_tokens=self.worker.model.max_tokens,
                max_output_tokens=self.worker.model.max_output_tokens,
                memory_query=self.memory_query,
                query_embedding=query_embedding,
            )

            # Create heartbeat callback for phase 2 response generation
            async def heartbeat_callback() -> None:
                """Refresh turn request heartbeat during long-running phase 2 generation."""
                result = await self.worker.atomic_heartbeat_update()

                if result == 0:
                    logger.debug("Turn request deleted during phase 2, stopping heartbeat")
                elif result == -1:
                    logger.error("Corrupted turn_request during phase 2 heartbeat")

            # Retry loop for emotional state header validation
            max_chat_retries = 3
            cleaned_response = ""
            used_fallback = False

            for format_attempt in range(max_chat_retries + 1):  # +1 for fallback
                # Check abort
                if await self.worker._check_abort_requested():
                    raise AbortRequestedException("Turn aborted before response")

                # Determine model role
                if format_attempt < max_chat_retries:
                    model_role = "chat"
                    attempt_label = f"{format_attempt + 1}/{max_chat_retries}"
                else:
                    # Try fallback if configured and different from chat
                    fallback_model_name = self.worker.model_set.get_model_name("fallback")
                    chat_model_name = self.worker.model_set.get_model_name("chat")

                    if fallback_model_name == chat_model_name:
                        logger.warning("Fallback model not configured or same as chat; skipping")
                        break

                    model_role = "fallback"
                    used_fallback = True
                    attempt_label = "fallback"
                    logger.info(f"Attempting fallback model ({fallback_model_name}) after {max_chat_retries} failures")

                # Call LLM
                response = await self.worker._call_llm(chat_turns, role=model_role, heartbeat_callback=heartbeat_callback)

                # Extract and validate
                cleaned_response, think_content = extract_think_tags(response)
                cleaned_response = sanitize_response(cleaned_response)
                cleaned_response = cleaned_response.strip()
                if think_content:
                    thinking_parts.append(think_content)

                # Check format
                if has_emotional_state_header(cleaned_response):
                    if used_fallback:
                        logger.info("Fallback model succeeded")
                    break

                logger.warning(f"Response missing Emotional State header (attempt {attempt_label})")
                logger.info(f"Response preview (first 1000 chars): {cleaned_response[:1000]}")

                # Add format guidance (stronger for fallback)
                if format_attempt < max_chat_retries:
                    # Normal guidance
                    persona_name = self.worker.session.persona_id if self.worker.session else "Agent"
                    format_guidance = (
                        f"\n\n[Gentle reminder from your link: Please begin with your emotional state, "
                        f"e.g. [== {persona_name}'s Emotional State: <list of your +Emotions+> ==] then continue with prose.]"
                    )
                    if chat_turns and chat_turns[-1]["role"] == "user":
                        chat_turns[-1]["content"] += format_guidance
                    else:
                        chat_turns.append({"role": "user", "content": format_guidance})
                elif not used_fallback:
                    # Stronger guidance before fallback
                    persona_name = self.worker.session.persona_id if self.worker.session else "Agent"
                    format_guidance = (
                        f"\n\n[CRITICAL FORMAT REQUIREMENT: You MUST begin your response with "
                        f"[== {persona_name}'s Emotional State: <your emotions> ==] followed by prose.]"
                    )
                    if chat_turns and chat_turns[-1]["role"] == "user":
                        chat_turns[-1]["content"] += format_guidance
                    else:
                        chat_turns.append({"role": "user", "content": format_guidance})

            # After loop - check if valid
            if not has_emotional_state_header(cleaned_response):
                logger.error(f"Failed after {max_chat_retries} chat attempts" + (" and fallback" if used_fallback else ""))
                action = MUDAction(
                    tool="emote",
                    args={"action": "struggles with expressing themselves."},
                    metadata={MUDAction.META_NON_PUBLISHED: True},
                )
                actions_taken.append(action)
                thinking_parts.append("[ERROR] Failed to generate valid response format after all retry attempts")
                await self.worker._emit_actions(actions_taken)
                thinking = "\n\n".join(thinking_parts).strip()
                return actions_taken, thinking

            # Normal flow continues...
            normalized = normalize_response(cleaned_response)

            if normalized:
                action = MUDAction(tool="speak", args={"text": normalized})
                actions_taken.append(action)
                logger.info("Prepared speak action (%d chars)", len(normalized))
                await self.worker._emit_actions(actions_taken)
            else:
                logger.info("No response content to emit")

        except Exception as e:
            logger.error(f"Error during speaking turn processing: {e}", exc_info=True)
            thinking_parts.append(f"[ERROR] Speaking turn processing failed: {e}")
            # Emit a graceful emote when LLM fails
            action = MUDAction(
                tool="emote",
                args={"action": "looks sick and unable to speak."},
                metadata={MUDAction.META_NON_PUBLISHED: True},
            )
            actions_taken.append(action)
            await self.worker._emit_actions(actions_taken)

        thinking = "\n\n".join(thinking_parts).strip()
        return actions_taken, thinking
