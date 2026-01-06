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
from dataclasses import replace
from typing import Optional, TYPE_CHECKING

from ..agents.persona import Persona
from ..llm.llm import is_retryable_error
from .context import ContextGatherer
from .paradigm import Paradigm
from ..tool.formatting import ToolUser
from ..utils.tokens import count_tokens

if TYPE_CHECKING:
    from ..config import ChatConfig
    from ..conversation.model import ConversationModel
    from ..utils.redis_cache import RedisCache

logger = logging.getLogger(__name__)


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
        self.llm_config = replace(config, max_tokens=4096)

        # Load persona for prompts
        self.persona = Persona.from_config(config)

        # Create context gatherer with token counter
        self.context_gatherer = ContextGatherer(
            cvm=cvm,
            token_counter=count_tokens,
        )

        self._redis_cache: Optional["RedisCache"] = None

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

        # Random paradigm selection from available configs (exclude journaler - it's an approach, not a paradigm)
        available = Paradigm.available(exclude=["journaler"])
        if not available:
            logger.error("No paradigm configs found")
            return None, None

        paradigm_name = random.choice(available)
        logger.info(f"Selected paradigm: {paradigm_name}")

        # Load paradigm strategy
        try:
            paradigm = Paradigm.load(paradigm_name)
        except ValueError as e:
            logger.error(f"Failed to load paradigm '{paradigm_name}': {e}")
            return None, None

        # Step 1: Broad gather + topic selection
        broad_context = await self.context_gatherer.broad_gather(paradigm_name, token_budget=16000)
        if broad_context.empty:
            logger.info("No documents found for exploration")
            return None, None

        broad_docs = broad_context.to_records()
        topic_result = await self._select_topic(paradigm, broad_docs, seed_query=seed_query)  # paradigm is now Paradigm object
        if topic_result is None:
            logger.info("LLM declined to select a topic")
            return None, None

        # Step 2: Targeted gather + validation
        approach = topic_result.get("approach", paradigm.name)  # Fallback to paradigm name
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

        # Step 3: Launch scenario - routing comes from paradigm config
        scenario = paradigm.get_scenario(approach)
        logger.info(f"Routing to scenario: {scenario}")

        pipeline_id = await self._trigger_pipeline(
            scenario=scenario,
            query_text=query_text,
            guidance=guidance,
            context_documents=targeted_docs,
        )
        return pipeline_id, None

    async def _select_topic(self, paradigm: Paradigm, documents: list[dict], max_retries: int = 5, seed_query: Optional[str] = None) -> Optional[dict]:
        """
        LLM call 1: Select topic using paradigm-specific prompts.

        Args:
            paradigm: The Paradigm strategy object
            documents: List of document dicts from broad gathering
            max_retries: Maximum retry attempts for invalid JSON responses
            seed_query: Optional suggested topic from previous rejection

        Returns:
            Dict with topic, approach, reasoning if valid, None otherwise
        """
        provider = self._get_llm_provider()

        # Build prompts using Paradigm strategy
        system_msg, user_msg = paradigm.build_selection_prompt(documents, self.persona, seed_query=seed_query)

        # Call LLM with adequate max_tokens for <think> + JSON
        turns = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        select_tool = paradigm.get_select_tool()
        tool_user = ToolUser([select_tool])
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

                logger.debug(f"Topic selection response ({paradigm.name}, attempt {attempt + 1}): {response[:200]}...")

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
        self, paradigm: Paradigm, topic_result: dict, documents: list[dict], max_retries: int = 5
    ) -> dict:
        """
        LLM call 2: Validate with accept/reject.

        Args:
            paradigm: The Paradigm strategy object
            topic_result: Result from topic selection step
            documents: Targeted context documents
            max_retries: Maximum retry attempts for invalid JSON responses

        Returns:
            Dict with accept, reasoning, query_text, guidance
        """
        topic = topic_result.get("topic", "")
        approach = topic_result.get("approach", paradigm.name)
        reasoning = topic_result.get("reasoning", "")

        provider = self._get_llm_provider()

        # Build prompts using Paradigm strategy
        system_msg, user_msg = paradigm.build_validation_prompt(documents, self.persona, topic, approach)

        # Call LLM with adequate max_tokens
        turns = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        validate_tool = paradigm.get_validate_tool()
        tool_user = ToolUser([validate_tool])
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

                logger.debug(f"Validation response ({paradigm.name}, attempt {attempt + 1}): {response[:200]}...")

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
            # For autonomous explorations, persona is both the actor and the "user"
            # (persona talking to itself / self-directed exploration)
            result = await self.dreamer_client.start(
                scenario_name=scenario,
                conversation_id=conversation_id,
                model_name=self.model_name,
                query_text=query_text,
                persona_id=self.config.persona_id,
                user_id=self.config.persona_id,
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
