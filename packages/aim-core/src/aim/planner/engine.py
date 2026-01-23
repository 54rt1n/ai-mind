# aim/planner/engine.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Planner engine - orchestrates two-stage plan creation."""

import asyncio
import logging
import time
from typing import Optional, TYPE_CHECKING

from aim.config import ChatConfig
from aim.conversation.model import ConversationModel
from aim.llm.models import LanguageModelV2
from aim.tool.formatting import ToolUser, ToolCallResult
from aim.tool.loader import ToolLoader

from .constants import (
    DELIBERATION_TIMEOUT,
    FORM_BUILDER_MAX_ITERATIONS,
    LLM_MAX_RETRIES,
    LLM_RETRY_DELAY,
)
from .form import PlanFormBuilder, FormState

if TYPE_CHECKING:
    from dream_agent.client import DreamerClient
    from aim_mud_types.models.plan import AgentPlan

logger = logging.getLogger(__name__)


class PlannerEngine:
    """Orchestrates the two-stage planner pipeline.

    Stage 1: Run deliberation scenario (dialogue, no tools)
    Stage 2: Run form builder loop (tool calling)

    Thread-safe: No. Single-threaded use only per engine instance.
    """

    def __init__(
        self,
        config: ChatConfig,
        dreamer_client: "DreamerClient",
        redis_client,
    ):
        """Initialize planner engine.

        Args:
            config: Chat configuration with model settings.
            dreamer_client: Client for running scenarios.
            redis_client: Async Redis client for plan storage.
        """
        self.config = config
        self.dreamer_client = dreamer_client
        self.redis = redis_client

    async def create_plan(
        self,
        agent_id: str,
        objective: str,
    ) -> Optional["AgentPlan"]:
        """Run the full planner pipeline.

        Args:
            agent_id: Agent to create plan for.
            objective: High-level goal for the plan.

        Returns:
            AgentPlan if successfully created and confirmed, None otherwise.
        """
        logger.info(f"Starting planner pipeline for {agent_id}: {objective[:50]}...")

        # Stage 1: Deliberation
        deliberation_result = await self._run_deliberation(agent_id, objective)
        if not deliberation_result:
            logger.warning(f"Deliberation failed for {agent_id}")
            return None

        logger.info("Deliberation complete, starting form builder")

        # Stage 2: Form building
        plan = await self._run_form_builder(
            agent_id,
            objective,
            deliberation_result,
        )

        if plan:
            # Save to Redis
            from aim_mud_types.client import RedisMUDClient
            client = RedisMUDClient(self.redis)
            await client.create_plan(plan)
            logger.info(f"Plan created and saved: {plan.plan_id}")

        return plan

    async def _run_deliberation(
        self,
        agent_id: str,
        objective: str,
    ) -> Optional[str]:
        """Run Stage 1: Deliberation scenario.

        Args:
            agent_id: Agent identifier.
            objective: Plan objective.

        Returns:
            Deliberation output text from CVM, or None on failure.
        """
        conversation_id = f"planner_{agent_id}_{int(time.time() * 1000)}"

        # Use run_and_wait for synchronous execution
        result = await self.dreamer_client.run_and_wait(
            scenario_name="planner_deliberation",
            conversation_id=conversation_id,
            model_name=self.config.model_name,
            query_text=objective,
            persona_id=self.config.persona_id,
            user_id=self.config.persona_id,
            timeout=DELIBERATION_TIMEOUT,
        )

        if not result.success:
            logger.error(f"Deliberation scenario failed: {result.error}")
            return None

        # result.status is Optional[PipelineStatus], status field is string
        if result.status and result.status.status != "complete":
            logger.error(f"Deliberation incomplete: {result.status.status}")
            return None

        # Extract final output from CVM - document type 'plan-deliberation'
        # ConversationModel.query() returns pd.DataFrame
        cvm = ConversationModel.from_config(self.config)
        df = cvm.query(
            query_texts=[""],  # Empty query - filter by conversation_id and type
            query_conversation_id=conversation_id,
            query_document_type=["plan-deliberation"],
            top_n=1,
        )

        if df.empty:
            logger.error(f"No plan-deliberation document found for {conversation_id}")
            return None

        # Access content from DataFrame row
        return df.iloc[0]["content"]

    async def _run_form_builder(
        self,
        agent_id: str,
        objective: str,
        deliberation_context: str,
    ) -> Optional["AgentPlan"]:
        """Run Stage 2: Form builder loop.

        Args:
            agent_id: Agent identifier.
            objective: Plan objective.
            deliberation_context: Output from Stage 1.

        Returns:
            AgentPlan if confirmed, None otherwise.
        """
        form = PlanFormBuilder(objective, deliberation_context)

        # Load tools using ToolLoader
        loader = ToolLoader(self.config.tools_path)
        loader.load_tools()
        tools = loader.get_tools_by_type("planner_form")

        if not tools:
            logger.error("No planner_form tools found")
            return None

        tool_user = ToolUser(tools)

        # Build initial context
        system_prompt = self._build_form_system_prompt(deliberation_context)

        for iteration in range(FORM_BUILDER_MAX_ITERATIONS):
            if form.state == FormState.COMPLETE:
                break

            # Get current prompt
            user_prompt = form.get_prompt()

            # Call LLM with tools
            response = await self._call_llm_with_tools(
                system_prompt,
                user_prompt,
                tool_user,
            )

            if response is None:
                logger.error("LLM call failed in form builder")
                continue  # Retry with same prompt

            # Process tool call - returns ToolCallResult dataclass
            result: ToolCallResult = tool_user.process_response(response)
            if not result.is_valid:
                logger.warning(f"Invalid tool response: {result.error}")
                # LLM didn't call a tool correctly - retry with same prompt
                continue

            # Execute tool using function_name and arguments from result
            tool_result = self._execute_form_tool(
                form,
                result.function_name,
                result.arguments or {},
            )

            logger.debug(f"Tool {result.function_name} result: {tool_result}")

            # Check for errors that should halt
            if tool_result.get("status") == "error":
                logger.warning(f"Tool error: {tool_result.get('message')}")
                # Continue - let LLM see the error and adjust

        if form.state == FormState.COMPLETE:
            return form.to_plan(agent_id)

        logger.error(f"Form builder did not complete after {FORM_BUILDER_MAX_ITERATIONS} iterations")
        return None

    def _build_form_system_prompt(self, deliberation_context: str) -> str:
        """Build system prompt for form builder loop.

        Args:
            deliberation_context: Output from Stage 1 deliberation.

        Returns:
            System prompt string.
        """
        return f"""You are the Coder aspect, responsible for structuring plans into executable form.

Based on the following deliberation, build a structured plan by calling the provided tools.
Follow the tool instructions exactly. Call ONE tool at a time.

## Deliberation Context

{deliberation_context}

## Instructions

1. Use **plan_add_task** to add each task (description, summary, context)
2. When done adding tasks, call **plan_done_adding**
3. For each task, call **plan_verify_task** with approved=true or approved=false
4. Use **plan_set_summary** to set the one-sentence plan summary
5. Finally, call **plan_confirm** to save and activate the plan

Always call exactly one tool per response. Do not include explanatory text."""

    async def _call_llm_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_user: ToolUser,
    ) -> Optional[str]:
        """Call LLM with tool definitions.

        Args:
            system_prompt: System message content.
            user_prompt: User message content.
            tool_user: Tool user with tool definitions.

        Returns:
            LLM response string, or None on failure.
        """
        models = LanguageModelV2.index_models(self.config)
        model = models.get(self.config.model_name)
        if not model:
            logger.error(f"Model {self.config.model_name} not available")
            return None

        provider = model.llm_factory(self.config)

        # Build turns - use get_tool_guidance() for tool definitions
        tool_guidance = tool_user.get_tool_guidance()
        turns = [
            {"role": "system", "content": f"{system_prompt}\n\n{tool_guidance}"},
            {"role": "user", "content": user_prompt},
        ]

        # Retry logic
        for attempt in range(LLM_MAX_RETRIES):
            try:
                chunks = []
                for chunk in provider.stream_turns(turns, self.config):
                    if chunk:
                        chunks.append(chunk)
                return "".join(chunks)
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < LLM_MAX_RETRIES - 1:
                    delay = LLM_RETRY_DELAY * (2 ** attempt)
                    await asyncio.sleep(delay)

        return None

    def _execute_form_tool(
        self,
        form: PlanFormBuilder,
        name: str,
        args: dict,
    ) -> dict:
        """Execute a form builder tool.

        Args:
            form: Form builder instance.
            name: Tool function name.
            args: Tool arguments.

        Returns:
            Tool result dictionary.
        """
        if name == "plan_add_task":
            return form.add_task(**args)
        elif name == "plan_done_adding":
            return form.done_adding()
        elif name == "plan_verify_task":
            return form.verify_task(**args)
        elif name == "plan_edit_task":
            return form.edit_task(**args)
        elif name == "plan_set_summary":
            return form.set_summary(**args)
        elif name == "plan_confirm":
            return form.confirm()
        else:
            return {"status": "error", "message": f"Unknown tool: {name}"}
