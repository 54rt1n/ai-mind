# aim/refiner/engine.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
ExplorationEngine - Autonomous exploration engine for AI-Mind.

Implements a 3-step agentic flow:
1. BROAD CONTEXT GATHERING + TOPIC SELECTION
2. TARGETED RETRIEVAL + VALIDATION
3. SCENARIO LAUNCH (if validated)

Uses proper tool calling via ToolUser for LLM interactions.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from typing import Optional, TYPE_CHECKING

import tiktoken

from aim.agents.persona import Persona
from aim.llm.llm import is_retryable_error
from aim.refiner.context import ContextGatherer
from aim.refiner.prompts import build_topic_selection_prompt, build_validation_prompt
from aim.tool.formatting import ToolUser
from aim.tool.dto import Tool, ToolFunction, ToolFunctionParameters

if TYPE_CHECKING:
    from aim.config import ChatConfig
    from aim.conversation.model import ConversationModel
    from aim.app.dream_agent.client import DreamerClient
    from aim.utils.redis_cache import RedisCache

logger = logging.getLogger(__name__)


# Shared tiktoken encoder
_encoder: tiktoken.Encoding = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def _count_tokens(text: str) -> int:
    """Count tokens in a string using tiktoken."""
    return len(_get_encoder().encode(text))


def _get_select_topic_tool() -> Tool:
    """Get the select_topic tool definition."""
    return Tool(
        type="refiner",
        function=ToolFunction(
            name="select_topic",
            description="Select a topic to explore in depth based on gathered context",
            parameters=ToolFunctionParameters(
                type="object",
                properties={
                    "topic": {
                        "type": "string",
                        "description": "The topic, theme, or concept to explore"
                    },
                    "approach": {
                        "type": "string",
                        "enum": ["philosopher", "journaler", "daydream", "critique"],
                        "description": "The exploration approach"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this topic is worth exploring"
                    },
                },
                required=["topic", "approach", "reasoning"],
                examples=[
                    {"select_topic": {"topic": "consciousness", "approach": "philosopher", "reasoning": "Underexplored"}}
                ],
            ),
        ),
    )


def _get_validate_tool() -> Tool:
    """Get the validate_exploration tool definition."""
    return Tool(
        type="refiner",
        function=ToolFunction(
            name="validate_exploration",
            description="Validate whether a topic is truly worth exploring",
            parameters=ToolFunctionParameters(
                type="object",
                properties={
                    "accept": {
                        "type": "boolean",
                        "description": "Whether to proceed with the exploration"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Explanation for accepting or rejecting"
                    },
                    "query_text": {
                        "type": "string",
                        "description": "The refined query to explore (if accept=true)"
                    },
                    "guidance": {
                        "type": "string",
                        "description": "Optional guidance for the exploration"
                    },
                    "redirect_to": {
                        "type": "string",
                        "enum": ["philosopher", "researcher", "daydream", "critique"],
                        "description": "Alternative scenario to redirect to (if rejecting but topic has potential)"
                    },
                    "suggested_query": {
                        "type": "string",
                        "description": "If rejecting, suggest an alternative unexplored topic to try next"
                    },
                },
                required=["accept", "reasoning"],
                examples=[
                    {"validate_exploration": {"accept": True, "reasoning": "Rich topic", "query_text": "What is consciousness?"}},
                    {"validate_exploration": {"accept": False, "reasoning": "Needs deeper pondering first", "redirect_to": "philosopher"}},
                    {"validate_exploration": {"accept": False, "reasoning": "Already explored this", "suggested_query": "the nature of forgetting"}}
                ],
            ),
        ),
    )


class ExplorationEngine:
    """
    Autonomous exploration engine for dream_watcher.

    Implements the 3-step agentic flow:
    1. Broad context gathering + topic selection with LLM
    2. Targeted retrieval + validation with LLM
    3. Scenario launch (only if step 2 accepts)

    Uses ToolUser for proper tool formatting and validation.
    """

    def __init__(
        self,
        config: "ChatConfig",
        cvm: "ConversationModel",
        dreamer_client: "DreamerClient",
        idle_threshold_seconds: int = 300,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the ExplorationEngine.

        Args:
            config: ChatConfig for API keys and settings
            cvm: ConversationModel for context queries
            dreamer_client: DreamerClient for triggering pipelines
            idle_threshold_seconds: Seconds of inactivity before considering idle
            model_name: Model to use for LLM decisions
        """
        self.config = config
        self.cvm = cvm
        self.dreamer_client = dreamer_client
        self.idle_threshold = idle_threshold_seconds
        self.model_name = model_name or config.default_model

        if not self.model_name:
            raise ValueError("No model specified and DEFAULT_MODEL not set in config")

        # Create a config copy with adequate max_tokens for thinking models
        # Thinking models need ~4096 tokens for <think> blocks + JSON response
        from dataclasses import replace
        self.llm_config = replace(config, max_tokens=4096)

        # Load persona for prompts
        self.persona = Persona.from_config(config)

        # Create context gatherer with token counter
        self.context_gatherer = ContextGatherer(
            cvm=cvm,
            token_counter=_count_tokens,
        )

        self._redis_cache: Optional["RedisCache"] = None

        # Tool definitions
        self._select_tool = _get_select_topic_tool()
        self._validate_tool = _get_validate_tool()

    def _get_redis_cache(self) -> "RedisCache":
        """Get or create RedisCache instance."""
        if self._redis_cache is None:
            from aim.utils.redis_cache import RedisCache
            self._redis_cache = RedisCache(self.config)
        return self._redis_cache

    def _get_llm_provider(self):
        """Get LLM provider for decision making."""
        from aim.llm.models import LanguageModelV2

        models = LanguageModelV2.index_models(self.config)
        model = models.get(self.model_name)

        if not model:
            raise ValueError(f"Model {self.model_name} not available")

        return model.llm_factory(self.config)

    async def is_api_idle(self) -> bool:
        """
        Check if the API has been idle for the threshold duration.

        Returns:
            True if API is idle (no recent activity), False if active
        """
        cache = self._get_redis_cache()
        last_activity = cache.get_api_last_activity()

        if last_activity is None:
            logger.debug("No API activity recorded, considering idle")
            return True

        elapsed = time.time() - last_activity
        is_idle = elapsed >= self.idle_threshold

        if is_idle:
            logger.debug(f"API idle for {elapsed:.0f}s (threshold: {self.idle_threshold}s)")
        else:
            logger.debug(f"API active: last activity {elapsed:.0f}s ago")

        return is_idle

    async def run_exploration(self, skip_idle_check: bool = False, seed_query: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
        """
        Three-step agentic flow:
        1. Broad gather + topic selection (LLM call 1)
        2. Targeted gather + validation (LLM call 2)
        3. Launch scenario (if accepted)

        Args:
            skip_idle_check: If True, skip the API idle check
            seed_query: Optional seed query from previous rejection's suggested_query

        Returns:
            Tuple of (pipeline_id, suggested_query):
            - pipeline_id: Pipeline ID if exploration started, None otherwise
            - suggested_query: If rejected, an alternative topic to try next
        """
        # Check API idle
        if not skip_idle_check and not await self.is_api_idle():
            logger.debug("API not idle, skipping exploration")
            return None, None

        # Random paradigm selection
        paradigm = random.choice(["brainstorm", "daydream", "knowledge", "critique"])
        logger.info(f"Selected paradigm: {paradigm}")

        # Step 1: Broad gather + topic selection
        broad_context = await self.context_gatherer.broad_gather(paradigm, token_budget=16000)
        if broad_context.empty:
            logger.info("No documents found for exploration")
            return None, None

        broad_docs = broad_context.to_records()
        topic_result = await self._select_topic(paradigm, broad_docs, seed_query=seed_query)
        if topic_result is None:
            logger.info("LLM declined to select a topic")
            return None, None

        # Step 2: Targeted gather + validation
        approach = topic_result.get("approach", paradigm)
        topic = topic_result.get("topic", "")
        reasoning = topic_result.get("reasoning", "")

        logger.info(f"Step 1 complete: topic='{topic}', approach={approach}")

        targeted_context = await self.context_gatherer.targeted_gather(
            topic=topic,
            approach=approach,
            token_budget=16000,
        )

        if targeted_context.empty:
            logger.warning(f"No targeted documents found for topic '{topic}'")
            return None, None

        targeted_docs = targeted_context.to_records()
        validation = await self._validate_exploration(paradigm, topic_result, targeted_docs)

        if not validation.get("accept", False):
            # Get suggested_query for next attempt
            suggested_query = validation.get("suggested_query")

            # Check for redirect - validation rejected but suggests another scenario
            redirect_to = validation.get("redirect_to")
            if redirect_to:
                logger.info(f"Exploration redirected to '{redirect_to}': {validation.get('reasoning', '')}")
                # Use the topic as query_text for the redirected scenario
                pipeline_id = await self._trigger_pipeline(
                    scenario=redirect_to,
                    query_text=topic,
                    guidance=validation.get("guidance"),
                    context_documents=targeted_docs,
                )
                return pipeline_id, None
            else:
                logger.info(f"Exploration rejected: {validation.get('reasoning', 'No reason')}")
                if suggested_query:
                    logger.info(f"Suggested next topic: {suggested_query}")
                return None, suggested_query

        query_text = validation.get("query_text", topic)
        guidance = validation.get("guidance")

        if not query_text:
            logger.warning("Validation accepted but no query_text provided")
            return None, None

        logger.info(f"Step 2 complete: exploration accepted with query '{query_text[:60]}...'")

        # Step 3: Launch scenario
        # Map paradigms to scenarios:
        # - daydream paradigm → daydream scenario
        # - knowledge paradigm → researcher scenario (librarian-led knowledge curation)
        # - critique paradigm → critique scenario (psychologist-led self-examination)
        # - brainstorm paradigm → approach (philosopher or journaler)
        if paradigm == "daydream":
            scenario = "daydream"
        elif paradigm == "knowledge":
            scenario = "researcher"
        elif paradigm == "critique":
            scenario = "critique"
        else:
            scenario = approach
        pipeline_id = await self._trigger_pipeline(
            scenario=scenario,
            query_text=query_text,
            guidance=guidance,
            context_documents=targeted_docs,
        )
        return pipeline_id, None

    async def _select_topic(self, paradigm: str, documents: list[dict], max_retries: int = 5, seed_query: Optional[str] = None) -> Optional[dict]:
        """
        LLM call 1: Select topic using paradigm-specific prompts.

        Args:
            paradigm: The exploration paradigm ("brainstorm", "daydream", "knowledge")
            documents: List of document dicts from broad gathering
            max_retries: Maximum retry attempts for invalid JSON responses
            seed_query: Optional suggested topic from previous rejection

        Returns:
            Dict with topic, approach, reasoning if valid, None otherwise
        """
        provider = self._get_llm_provider()

        # Build prompts using prompts.py
        system_msg, user_msg = build_topic_selection_prompt(paradigm, documents, self.persona, seed_query=seed_query)

        # Call LLM with adequate max_tokens for <think> + JSON
        turns = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        tool_user = ToolUser([self._select_tool])
        cache = self._get_redis_cache()

        for attempt in range(max_retries):
            try:
                # Update activity timestamp during streaming to prevent cascading triggers
                cache.update_api_activity()

                chunks = []
                chunk_count = 0
                for chunk in provider.stream_turns(turns, self.llm_config):
                    if chunk:
                        chunks.append(chunk)
                        chunk_count += 1
                        if chunk_count % 50 == 0:
                            cache.update_api_activity()
                cache.update_api_activity()
                response = "".join(chunks)

                logger.debug(f"Topic selection response (attempt {attempt + 1}): {response[:200]}...")

                # Parse tool call using ToolUser
                result = tool_user.process_response(response)

                if result.is_valid:
                    return result.arguments

                logger.warning(f"Invalid tool call (attempt {attempt + 1}/{max_retries}): {result.error}")

            except Exception as e:
                logger.error(f"Error in topic selection (attempt {attempt + 1}/{max_retries}): {e}")

                # Add exponential backoff for retryable errors
                if is_retryable_error(e) and attempt < max_retries - 1:
                    delay = min(30 * (2 ** attempt), 120)  # 30s, 60s, 120s max
                    logger.info(f"Retryable error, waiting {delay}s before retry...")
                    await asyncio.sleep(delay)

        logger.error(f"Topic selection failed after {max_retries} attempts")
        return None

    async def _validate_exploration(
        self, paradigm: str, topic_result: dict, documents: list[dict], max_retries: int = 5
    ) -> dict:
        """
        LLM call 2: Validate with accept/reject.

        Args:
            paradigm: The exploration paradigm
            topic_result: Result from topic selection step
            documents: Targeted context documents
            max_retries: Maximum retry attempts for invalid JSON responses

        Returns:
            Dict with accept, reasoning, query_text, guidance
        """
        topic = topic_result.get("topic", "")
        approach = topic_result.get("approach", paradigm)
        reasoning = topic_result.get("reasoning", "")

        provider = self._get_llm_provider()

        # Build prompts
        system_msg, user_msg = build_validation_prompt(
            paradigm=paradigm,
            topic=topic,
            approach=approach,
            reasoning=reasoning,
            documents=documents,
            persona=self.persona,
        )

        # Call LLM with adequate max_tokens
        turns = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        tool_user = ToolUser([self._validate_tool])
        cache = self._get_redis_cache()

        for attempt in range(max_retries):
            try:
                # Update activity timestamp during streaming to prevent cascading triggers
                cache.update_api_activity()

                chunks = []
                chunk_count = 0
                for chunk in provider.stream_turns(turns, self.llm_config):
                    if chunk:
                        chunks.append(chunk)
                        chunk_count += 1
                        if chunk_count % 50 == 0:
                            cache.update_api_activity()
                cache.update_api_activity()
                response = "".join(chunks)

                logger.debug(f"Validation response (attempt {attempt + 1}): {response[:200]}...")

                # Validate with ToolUser
                result = tool_user.process_response(response)

                if result.is_valid:
                    return result.arguments

                logger.warning(f"Invalid validation (attempt {attempt + 1}/{max_retries}): {result.error}")

            except Exception as e:
                logger.error(f"Error in validation (attempt {attempt + 1}/{max_retries}): {e}")

                # Add exponential backoff for retryable errors
                if is_retryable_error(e) and attempt < max_retries - 1:
                    delay = min(30 * (2 ** attempt), 120)  # 30s, 60s, 120s max
                    logger.info(f"Retryable error, waiting {delay}s before retry...")
                    await asyncio.sleep(delay)

        logger.error(f"Validation failed after {max_retries} attempts")
        return {"accept": False, "reasoning": f"Failed to get valid response after {max_retries} attempts"}

    async def _trigger_pipeline(
        self, scenario: str, query_text: str, guidance: Optional[str], context_documents: list[dict]
    ) -> Optional[str]:
        """
        Launch the scenario pipeline.

        Args:
            scenario: The scenario to run (philosopher, journaler, daydream)
            query_text: The exploration query
            guidance: Optional guidance for the pipeline
            context_documents: The targeted context documents

        Returns:
            Pipeline ID if successful, None otherwise
        """
        # Generate conversation ID matching webui format: {scenario}_{timestamp}_{random9}
        timestamp = int(time.time() * 1000)
        random_suffix = uuid.uuid4().hex[:9]
        conversation_id = f"{scenario}_{timestamp}_{random_suffix}"

        logger.info(
            f"Triggering exploration: scenario={scenario} "
            f"query={query_text[:60]}... "
            f"conversation={conversation_id}"
        )

        try:
            result = await self.dreamer_client.start(
                scenario_name=scenario,
                conversation_id=conversation_id,
                model_name=self.model_name,
                query_text=query_text,
                persona_id=self.config.persona_id,
                guidance=guidance,
                context_documents=context_documents,
            )

            if result.success:
                logger.info(f"Exploration started: pipeline={result.pipeline_id}")
                return result.pipeline_id
            else:
                logger.warning(f"Failed to start exploration: {result.error}")
                return None

        except Exception as e:
            logger.error(f"Error triggering pipeline: {e}")
            return None
