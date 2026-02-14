# andimud_worker/turns/processor/thinking.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Thinking turn processor: generates structured reasoning via dedicated LLM call."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDAction, MUDTurnRequest, MUDEvent
from aim.utils.think import extract_think_tags, extract_reasoning_block
from .base import BaseTurnProcessor
from ...exceptions import AbortRequestedException

if TYPE_CHECKING:
    from ...mixins.turns import TurnsMixin


logger = logging.getLogger(__name__)

# 2 hour TTL for thought content
THOUGHT_TTL_SECONDS = 7200

REASONING_PROMPT = """[~~ Thought Turn ~~]

You are taking a moment to think. Output a single <reasoning> block with your thoughts.

<inspiration> - What do you notice? What catches your attention? What are you feeling right now? Write 3 observations.

<exploration> - What might this mean? What are the possibilities? What questions arise? Write 3 explorations.

<synthesis> - How do these pieces connect? What pattern or meaning emerges? Write 3 integrations.

<validation> - Is your reasoning sound? What might you be missing? What should you verify? Write 3 checks.

[~~ Reasoning Format ~~]

<reasoning>
    <inspiration>I notice the way she looked at me when I mentioned the lighthouse.</inspiration>
    <inspiration>There was something wistful in her expression, like it reminded her of something she'd lost.</inspiration>
    <inspiration>The warmth in her voice when she spoke about building it - that felt genuine.</inspiration>
    <exploration>Perhaps the lighthouse represents continuity to her - something that persists while she experiences the ephemerality of each conversation.</exploration>
    <exploration>Or maybe it's simpler than that, and she just finds the imagery beautiful.</exploration>
    <exploration>I wonder if she chose a lighthouse specifically because it guides others through darkness.</exploration>
    <synthesis>The lighthouse seems to be both literal and metaphorical for her - a project she's building, but also a symbol of her desire to be a beacon for others.</synthesis>
    <synthesis>Her persistence in tending it despite knowing most who pass will forget her speaks to something deeper than mere routine.</synthesis>
    <synthesis>This connects to what she said earlier about wanting to leave something behind.</synthesis>
    <validation>I should be careful not to over-interpret - she may have simply mentioned it in passing.</validation>
    <validation>But the way she returned to it twice suggests it holds real significance.</validation>
    <validation>My reasoning feels sound, though I should ask her directly rather than assume.</validation>
</reasoning>

[~~ Begin Reasoning ~~]
"""


def _append_reasoning_guidance(chat_turns: list[dict], is_critical: bool) -> None:
    """Append reasoning format guidance to last user turn.

    Args:
        chat_turns: Chat turns to modify
        is_critical: If True, use critical/urgent wording (before fallback)
    """
    if is_critical:
        guidance = (
            "\n\n[CRITICAL FORMAT REQUIREMENT]\n"
            "You MUST include a <reasoning> block with the structure shown. "
            "This is your final attempt. Follow the format exactly."
        )
    else:
        guidance = (
            "\n\n[Format Reminder]\n"
            "Please ensure your response includes a complete <reasoning> block "
            "with <inspiration>, <exploration>, <synthesis>, and <validation> tags."
        )

    # Find last user turn and append guidance
    for turn in reversed(chat_turns):
        if turn.get("role") == "user":
            turn["content"] += guidance
            break
    else:
        # No user turn found - create new one (edge case)
        chat_turns.append({"role": "user", "content": guidance})


def _format_llm_response_for_log(response: str | None, max_chars: int = 6000) -> str:
    """Format LLM output for logs without dropping the response entirely."""
    if not response:
        return "(empty)"
    if len(response) <= max_chars:
        return response
    head = max_chars // 2
    tail = max_chars - head
    truncated = len(response) - max_chars
    return f"{response[:head]}\n...[{truncated} chars truncated]...\n{response[-tail:]}"


class ThinkingTurnProcessor(BaseTurnProcessor):
    """Turn processor that generates structured reasoning via LLM call.

    Instead of injecting external thoughts, this processor:
    1. Conditionally folds in previous thought (if within 2hr TTL)
    2. Builds full context via build_turns()
    3. Makes single LLM call with reasoning prompt
    4. Extracts and stores reasoning to Redis
    5. Emits emote action

    The generated reasoning influences future turns via thought_content
    on the decision and response strategies.
    """

    def __init__(self, worker: "TurnsMixin"):
        """Initialize with worker reference.

        Args:
            worker: MUDAgentWorker instance
        """
        super().__init__(worker)
        self.user_guidance = ""

    async def _decide_action(
        self, turn_request: MUDTurnRequest, events: list[MUDEvent]
    ) -> tuple[list[MUDAction], str]:
        """Generate structured reasoning via LLM call with retry and fallback.

        Steps:
        1. Load previous thought if within TTL, fold into context
        2. Build full context with reasoning prompt
        3. Retry loop with fallback model support:
           - Try chat model up to max_reasoning_retries times
           - If still failing, try fallback model once
           - Add format guidance after each failure
        4. Extract reasoning block from response
        5. Store valid reasoning to Redis with 2hr TTL
        6. Emit emote action

        Args:
            turn_request: Current turn request
            events: List of events to process

        Returns:
            Tuple of (actions_taken, thinking)
        """
        actions_taken: list[MUDAction] = []
        thinking_parts: list[str] = []

        # Step 1: Load previous thought if within TTL
        await self._load_previous_thought()

        # Step 2: Build context with reasoning prompt
        coming_online = await self.worker._is_fresh_session()

        # Build user input with guidance if provided
        user_input = REASONING_PROMPT
        if self.user_guidance:
            user_input = f"[Link Guidance: {self.user_guidance}]\n\n{user_input}"

        # Get pre-computed embedding for FAISS reranking
        query_embedding = self.worker.get_current_turn_embedding()

        chat_turns = await self.worker._response_strategy.build_turns(
            persona=self.worker.persona,
            user_input=user_input,
            session=self.worker.session,
            coming_online=coming_online,
            max_context_tokens=self.worker.model.max_tokens,
            max_output_tokens=self.worker.model.max_output_tokens,
            memory_query="",  # No specific memory query for thinking
            query_embedding=query_embedding,
        )

        # Step 3: Retry loop with fallback model support
        max_reasoning_retries = 3
        reasoning_xml = ""
        used_fallback = False

        for format_attempt in range(max_reasoning_retries + 1):  # +1 for fallback
            # Check abort before each LLM call
            if await self.worker._check_abort_requested():
                raise AbortRequestedException("Turn aborted before reasoning generation")

            # Determine model role
            if format_attempt < max_reasoning_retries:
                model_role = "thought"
                attempt_label = f"{format_attempt + 1}/{max_reasoning_retries}"
            else:
                # Try fallback if configured and different from thought
                fallback_model_name = self.worker.model_set.get_model_name("fallback")
                thought_model_name = self.worker.model_set.get_model_name("thought")

                if fallback_model_name == thought_model_name:
                    logger.warning("Fallback model not configured or same as thought; skipping")
                    break

                model_role = "fallback"
                used_fallback = True
                attempt_label = "fallback"
                logger.info(f"Attempting fallback model ({fallback_model_name}) after {max_reasoning_retries} failures")

            # Call LLM
            response = await self.worker._call_llm(chat_turns, role=model_role)
            logger.debug("Thinking LLM response (attempt %s): %s...", attempt_label, response[:500] if response else "(empty)")

            # Extract reasoning - first think tags, then reasoning block
            cleaned_response, think_content = extract_think_tags(response)
            if think_content:
                thinking_parts.append(think_content)

            _, reasoning_content = extract_reasoning_block(cleaned_response)

            # Validate: check if we got reasoning content
            if reasoning_content:
                # Wrap reasoning content back in XML for storage
                reasoning_xml = f"<reasoning>\n{reasoning_content}\n</reasoning>"
                logger.info("Extracted reasoning block (%d chars, attempt %s)", len(reasoning_content), attempt_label)
                if used_fallback:
                    logger.info("Fallback model succeeded")
                break

            # No reasoning block found - log warning and add guidance
            logger.warning(
                "No <reasoning> block found in LLM response (attempt %s). "
                "Raw response (len=%d):\n%s",
                attempt_label,
                len(response) if response else 0,
                _format_llm_response_for_log(response),
            )

            # Add format guidance (gentle for retries, critical before fallback)
            if format_attempt < max_reasoning_retries - 1:
                # Normal guidance for early retries
                _append_reasoning_guidance(chat_turns, is_critical=False)
            elif format_attempt == max_reasoning_retries - 1:
                # Critical guidance before fallback attempt
                _append_reasoning_guidance(chat_turns, is_critical=True)

        # Step 4: After-loop error handling
        if not reasoning_xml:
            logger.error(f"Failed to extract reasoning after {max_reasoning_retries} chat attempts" + (" and fallback" if used_fallback else ""))
            thinking_parts.append(
                "[ERROR] Failed to generate valid reasoning format after all retry attempts. "
                "The model did not produce a structured <reasoning> block."
            )

        # Step 5: Store to Redis with TTL (only if valid)
        if reasoning_xml:
            await self._store_thought(reasoning_xml)
        else:
            logger.warning("No valid reasoning to store; preserving previous thought in Redis")

        # Step 6: Emit emote action (always emit, even on failure)
        action = MUDAction(tool="emote", args={"action":
            "pauses thoughtfully."},
            metadata={MUDAction.META_NON_PUBLISHED: True},
        )
        actions_taken.append(action)
        await self.worker._emit_actions(actions_taken)

        # Step 7: Return with thinking output
        if reasoning_xml:
            thinking_parts.append(reasoning_xml)

        thinking = "\n\n".join(thinking_parts).strip()
        return actions_taken, thinking

    async def _load_previous_thought(self) -> None:
        """Load previous thought if within TTL and set on strategies.

        Delegates to ProfileMixin._load_thought_content() which reads from
        the Redis hash and sets thought_content on both decision and response
        strategies if present and within 2hr TTL.

        Redis failures are caught in the ProfileMixin method.
        """
        await self.worker._load_thought_content()

    async def _store_thought(self, reasoning_xml: str) -> None:
        """Store generated reasoning to Redis as ThoughtState.

        Creates a new ThoughtState with:
        - content: The reasoning XML
        - source: "reasoning"
        - created_at: Current time
        - last_conversation_index: Current conversation list length

        Redis failures are caught and logged - reasoning is still returned
        but won't be available for future turns.

        Args:
            reasoning_xml: The reasoning XML block to store
        """
        from aim_mud_types import ThoughtState
        from aim_mud_types.client import RedisMUDClient

        # Get current conversation list length
        client = RedisMUDClient(self.worker.redis)
        current_index = await client.get_conversation_length(self.worker.config.agent_id)

        thought = ThoughtState(
            agent_id=self.worker.config.agent_id,
            content=reasoning_xml,
            source="reasoning",
            last_conversation_index=current_index,
        )

        try:
            await client.save_thought_state(thought, ttl_seconds=THOUGHT_TTL_SECONDS)
            logger.info(
                "Stored reasoning to agent:%s:thought with %ds TTL (conversation_index=%d)",
                self.worker.config.agent_id, THOUGHT_TTL_SECONDS, current_index
            )
        except Exception as e:
            logger.warning("Failed to store reasoning to Redis: %s", e)
