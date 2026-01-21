# andimud_worker/turns/processor/thinking.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Thinking turn processor: generates structured reasoning via dedicated LLM call."""

import json
import logging
import time
from typing import TYPE_CHECKING

from aim_mud_types import MUDAction, MUDTurnRequest, MUDEvent, RedisKeys
from aim.utils.think import extract_think_tags, extract_reasoning_block
from .base import BaseTurnProcessor

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
        """Generate structured reasoning via LLM call.

        Steps:
        1. Load previous thought if within TTL, fold into context
        2. Build full context with reasoning prompt
        3. Call LLM
        4. Extract reasoning block from response
        5. Store to Redis with 2hr TTL
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
            user_input = f"[User Guidance: {self.user_guidance}]\n\n{user_input}"

        chat_turns = await self.worker._response_strategy.build_turns(
            persona=self.worker.persona,
            user_input=user_input,
            session=self.worker.session,
            coming_online=coming_online,
            max_context_tokens=self.worker.model.max_tokens,
            max_output_tokens=self.worker.chat_config.max_tokens,
            memory_query="",  # No specific memory query for thinking
        )

        # Step 3: Call LLM
        response = await self.worker._call_llm(chat_turns, role="chat")
        logger.debug("Thinking LLM response: %s...", response[:500] if response else "(empty)")

        # Step 4: Extract reasoning - first think tags, then reasoning block
        cleaned_response, think_content = extract_think_tags(response)
        if think_content:
            thinking_parts.append(think_content)

        _, reasoning_content = extract_reasoning_block(cleaned_response)

        if reasoning_content:
            # Wrap reasoning content back in XML for storage
            reasoning_xml = f"<reasoning>\n{reasoning_content}\n</reasoning>"
            logger.info("Extracted reasoning block (%d chars)", len(reasoning_content))
        else:
            # No reasoning block found - log warning and store empty
            logger.warning("No <reasoning> block found in LLM response")
            reasoning_xml = ""

        # Step 5: Store to Redis with TTL
        await self._store_thought(reasoning_xml)

        # Step 6: Emit emote action
        action = MUDAction(tool="emote", args={"action": "pauses thoughtfully."})
        actions_taken.append(action)
        await self.worker._emit_actions(actions_taken)

        # Include reasoning in thinking output
        if reasoning_xml:
            thinking_parts.append(reasoning_xml)

        thinking = "\n\n".join(thinking_parts).strip()
        return actions_taken, thinking

    async def _load_previous_thought(self) -> None:
        """Load previous thought if within TTL and set on response strategy.

        Checks agent:{id}:thought for existing thought content. If present
        and within 2hr TTL, sets thought_content on response strategy so
        it gets folded into context via insert_at_fold().

        Redis failures are caught and logged - turn continues without previous thought.
        """
        thought_key = RedisKeys.agent_thought(self.worker.config.agent_id)

        try:
            thought_raw = await self.worker.redis.get(thought_key)
        except Exception as e:
            logger.warning("Failed to load previous thought from Redis: %s", e)
            return

        if not thought_raw:
            logger.debug("No previous thought found")
            return

        try:
            raw_str = thought_raw.decode("utf-8") if isinstance(thought_raw, bytes) else thought_raw
            thought_data = json.loads(raw_str)
            thought_content = thought_data.get("content", "")
            timestamp = thought_data.get("timestamp", 0)

            age_seconds = time.time() - timestamp
            if age_seconds < THOUGHT_TTL_SECONDS and thought_content:
                self.worker._response_strategy.thought_content = thought_content
                logger.info(
                    "Loaded previous thought (%d chars, %.0fs old) into context",
                    len(thought_content),
                    age_seconds,
                )
            else:
                logger.debug(
                    "Previous thought expired (%.0fs old, TTL=%ds)",
                    age_seconds,
                    THOUGHT_TTL_SECONDS,
                )
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse previous thought JSON: %s", e)

    async def _store_thought(self, reasoning_xml: str) -> None:
        """Store generated reasoning to Redis with TTL.

        Redis failures are caught and logged - reasoning is still returned
        but won't be available for future turns.

        Args:
            reasoning_xml: The reasoning XML block to store
        """
        thought_data = {
            "content": reasoning_xml,
            "source": "reasoning",
            "timestamp": int(time.time()),
        }
        thought_key = RedisKeys.agent_thought(self.worker.config.agent_id)

        try:
            await self.worker.redis.set(
                thought_key,
                json.dumps(thought_data),
                ex=THOUGHT_TTL_SECONDS,
            )
            logger.info("Stored reasoning to %s with %ds TTL", thought_key, THOUGHT_TTL_SECONDS)
        except Exception as e:
            logger.warning("Failed to store reasoning to Redis: %s", e)
