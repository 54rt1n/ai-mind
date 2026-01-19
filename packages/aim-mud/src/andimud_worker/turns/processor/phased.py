# aim/app/mud/worker/turns/strategy/phased.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Phased turn processor: decision phase → conditional response phase."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDAction, MUDTurnRequest, MUDEvent
from aim.utils.think import extract_think_tags
from ...adapter import build_current_context
from ..response import (
    sanitize_response,
    normalize_response,
    has_emotional_state_header,
)
from .base import BaseTurnProcessor
from ...tools.helper import ToolHelper
from ...exceptions import AbortRequestedException

if TYPE_CHECKING:
    from ...mixins.turns import TurnsMixin

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

    async def _decide_action(self, turn_request: MUDTurnRequest, events: list[MUDEvent]) -> tuple[list[MUDAction], str]:
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
                raise AbortRequestedException("Turn aborted before decision")

            # Phase 1: Decision (use decision role - fast tool selection)
            decision_tool, decision_args, decision_raw, decision_thinking, decision_cleaned = (
                await self.worker._decide_action(
                    idle_mode=idle_mode,
                    role="decision",
                    action_guidance="",
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

            elif self.worker._decision_strategy.is_aura_tool(decision_tool):
                # Generic aura tool handling - emit MUDAction for Evennia to execute
                action = MUDAction(tool=decision_tool, args=decision_args)
                actions_taken.append(action)
                logger.info("Aura tool '%s' emitting action with args: %s", decision_tool, decision_args)
                await self.worker._emit_actions(actions_taken)

            elif decision_tool == "emote":
                action_text = (decision_args.get("action") or "").strip()
                if action_text:
                    action = MUDAction(tool="emote", args={"action": action_text})
                    actions_taken.append(action)

                    await self.worker._emit_actions(actions_taken)
                else:
                    logger.warning("Phase1 emote missing action; no action emitted")

            elif decision_tool == "wait":
                logger.info("Phase 1 decided to wait; no action this turn")
                # Emit a subtle emote based on mood (if provided)
                mood = decision_args.get("mood", "").strip()
                if mood:
                    emote_text = f"waits {mood}."
                    logger.info(f"Wait emote with mood: {emote_text}")
                else:
                    emote_text = "waits quietly."
                    logger.info("Wait emote with default mood")
                action = MUDAction(tool="emote", args={"action": emote_text})
                actions_taken.append(action)

                await self.worker._emit_actions(actions_taken)

            elif decision_tool == "speak":
                # Phase 2: full response turn with memory via response strategy
                coming_online = await self.worker._is_fresh_session()

                # Extract memory query from speak args (enhances CVM search)
                memory_query = decision_args.get("query") or decision_args.get("focus") or ""
                if memory_query:
                    logger.info(f"Phase 2 memory query: {memory_query[:100]}...")

                guidance_parts: list[str] = []
                if memory_query:
                    guidance_parts.append(f"Speech focus: {memory_query}")
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
                    action = MUDAction(tool="emote", args={"action": "was at a loss for words."})
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

            elif decision_tool == "plan_update":
                # Plan task status was updated - emit emote about progress
                plan_status = decision_args.get("plan_status", "unknown")
                next_task = decision_args.get("next_task")

                if plan_status == "completed":
                    emote_text = "completed the plan successfully."
                elif next_task:
                    emote_text = f"completed a task and moved on to: {next_task}"
                else:
                    emote_text = "updated the plan status."

                logger.info(f"Plan update: status={plan_status}, next_task={next_task}")
                action = MUDAction(tool="emote", args={"action": emote_text})
                actions_taken.append(action)

                await self.worker._emit_actions(actions_taken)

            elif decision_tool == "confused":
                # Phase 1 failed to parse a valid decision - emit confused emote
                logger.info("Phase 1 returned confused; emitting confused emote")
                emote_text = "looks confused."
                action = MUDAction(tool="emote", args={"action": emote_text})
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
