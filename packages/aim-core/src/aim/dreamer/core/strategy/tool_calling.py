# aim/dreamer/core/strategy/tool_calling.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""ToolCallingStrategy - LLM with tools, conditionals, and no document output."""

import logging
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, Any

from .base import BaseStepStrategy, ScenarioStepResult, ScenarioExecutor
from .functions import execute_context_actions, load_memory_docs

if TYPE_CHECKING:
    from ..models import ToolCallingStepDefinition, StepResult, Condition


logger = logging.getLogger(__name__)


class ToolCallingStrategy(BaseStepStrategy):
    """Executes LLM with tools, evaluates conditions, sets next step.

    This is the most complex strategy, handling:
    - Tool calling with retry logic
    - Conditional flow control based on tool results
    - Collection of results for iteration
    - Max iteration limits

    No document output - only modifies state.

    Attributes:
        step_def: ToolCallingStepDefinition with prompt, tools, next_conditions, config
    """

    step_def: "ToolCallingStepDefinition"

    async def execute(self) -> ScenarioStepResult:
        """Execute tool-calling step.

        1. Execute context DSL if present
        2. Call LLM with tools (with retries)
        3. Parse tool call from response
        4. Record turn and step result
        5. Evaluate conditions to determine next step
        6. Handle collection and iteration limits

        Returns:
            ScenarioStepResult with next step from conditions
        """
        step_id = self.step_def.id
        executor = self.executor
        max_retries = self.step_def.config.tool_retries

        logger.info(
            f"ToolCallingStrategy executing step '{step_id}' "
            f"(tools={self.step_def.tools}, max_retries={max_retries})"
        )

        # 1. Execute context DSL if present (clears and populates memory_refs)
        if self.step_def.context:
            execute_context_actions(executor, self.step_def)

        # 2. Call LLM with tools (with retries)
        tool_call = None
        response = None
        rendered_prompt = None
        tool_user = None

        for attempt in range(max_retries):
            response, tool_user, rendered_prompt = await self._call_llm_with_tools()
            tool_call = tool_user.process_response(response)

            if tool_call.is_valid:
                break

            logger.warning(
                f"Step '{step_id}': LLM didn't call tool "
                f"(attempt {attempt + 1}/{max_retries})"
            )

        # 3. Record turn (even if failed)
        executor.state.record_turn(step_id, rendered_prompt, response)

        # 4. Check if we got a valid tool call
        if not tool_call or not tool_call.is_valid:
            logger.error(
                f"Step '{step_id}': No valid tool call after {max_retries} attempts"
            )
            return ScenarioStepResult(
                success=False,
                next_step="abort",
                state_changed=True,
                error=f"No valid tool call after {max_retries} attempts",
            )

        tool_name = tool_call.function_name
        tool_result = tool_call.arguments

        logger.info(
            f"Step '{step_id}': Got tool call '{tool_name}' "
            f"with args: {list(tool_result.keys()) if tool_result else []}"
        )

        # 5. Record step result
        step_result = self._create_step_result(response, tool_name, tool_result)
        executor.state.record_step_result(step_result)

        # 6. Increment iteration count
        iterations = executor.state.increment_iteration(step_id)

        # 7. Evaluate conditions to determine next step
        next_step = self._evaluate_conditions(step_result)

        logger.info(
            f"Step '{step_id}' complete, "
            f"tool='{tool_name}', iteration={iterations}, "
            f"next_step='{next_step}'"
        )

        return ScenarioStepResult(
            success=True,
            next_step=next_step,
            state_changed=True,
            doc_created=False,  # Tool calling steps don't create documents
        )

    async def _call_llm_with_tools(self) -> tuple[str, Any, str]:
        """Build turns and call LLM with tools.

        Returns:
            Tuple of (raw_response, tool_user, rendered_prompt)
        """
        from aim.tool.formatting import ToolUser
        from aim.utils.xml import XmlFormatter

        executor = self.executor
        step_def = self.step_def

        # 1. Get tools from framework
        tools = executor.framework.get_tools(step_def.tools)
        tool_user = ToolUser(tools)

        # 2. Build system message with persona + tools
        system_message = self._build_system_message(tool_user)

        # 3. Load memory docs
        memory_docs = load_memory_docs(executor)

        # 4. Build user turn with prompt and context
        rendered_prompt = self._build_user_turn(memory_docs, tool_user)

        # 5. Build turns from conversation history
        turns = self._build_turns(rendered_prompt)

        # 6. Get model and provider
        model = self._get_model()
        if not model:
            raise ValueError(f"Model not available for step {step_def.id}")

        provider = model.llm_factory(executor.config)

        # 7. Build LLM config
        llm_config = replace(
            executor.config,
            system_message=system_message,
            max_tokens=min(
                step_def.config.max_tokens or 1024,
                model.max_output_tokens
            ),
            temperature=step_def.config.temperature or executor.config.temperature,
        )

        # 8. Stream response with heartbeat
        chunks = []
        chunk_count = 0
        for chunk in provider.stream_turns(turns, llm_config):
            if chunk:
                chunks.append(chunk)
                chunk_count += 1
                if chunk_count % 50 == 0:
                    await self._heartbeat()

        response = ''.join(chunks)
        return response, tool_user, rendered_prompt

    def _build_system_message(self, tool_user: Any) -> str:
        """Build system message with persona + tools."""
        from aim.utils.xml import XmlFormatter

        xml = XmlFormatter()
        xml = self.executor.persona.xml_decorator(xml)
        xml = tool_user.xml_decorator(xml)
        return xml.render()

    def _build_user_turn(self, memory_docs: list[dict], tool_user: Any) -> str:
        """Build current user turn with memory context, prompt, and tool guidance."""
        from aim.utils.xml import XmlFormatter
        from ..scenario import render_template

        executor = self.executor

        xml = XmlFormatter()

        # Memory context
        if memory_docs:
            for i, doc in enumerate(memory_docs):
                content = doc.get('content', '')
                doc_type = doc.get('document_type', 'memory')
                xml.add_element(
                    f"memory_{i}",
                    f"Memory ({doc_type})",
                    content=content,
                    priority=2,
                )

        # Render prompt
        ctx = executor.state.build_template_context()
        ctx['persona'] = executor.persona
        ctx['pronouns'] = executor.persona.pronouns
        prompt = render_template(self.step_def.prompt, ctx)
        xml.add_element("prompt", "Current Task", content=prompt, priority=1)

        # Tool guidance
        tool_guidance = tool_user.get_tool_guidance()
        xml.add_element(
            "tool_guidance",
            "Available Tools",
            content=tool_guidance,
            priority=1
        )

        # External guidance
        if executor.state.guidance:
            xml.add_element(
                "guidance",
                "Guidance",
                content=executor.state.guidance,
                priority=1
            )

        return xml.render()

    def _build_turns(self, current_prompt: str) -> list[dict]:
        """Build turns from conversation history."""
        turns = []

        # Prior turns from state
        for turn in self.executor.state.turns:
            turns.append({'role': 'user', 'content': turn.prompt})
            turns.append({'role': 'assistant', 'content': turn.response})

        # Current prompt
        turns.append({'role': 'user', 'content': current_prompt})

        return turns

    def _get_model(self):
        """Get the language model for this step."""
        from aim.llm.models import LanguageModelV2

        executor = self.executor
        step_config = self.step_def.config

        # Tool calling defaults to 'tool' role
        model_role = step_config.model_role or "tool"

        if step_config.model_override:
            model_name = step_config.model_override
        else:
            model_name = executor.model_set.get_model_name(model_role)

        models = LanguageModelV2.index_models(executor.config)
        return models.get(model_name)

    def _create_step_result(
        self,
        response: str,
        tool_name: str,
        tool_result: dict,
    ) -> "StepResult":
        """Create StepResult with tool information."""
        from ..models import StepResult
        from aim.utils.tokens import count_tokens
        from aim.conversation.message import ConversationMessage

        return StepResult(
            step_id=self.step_def.id,
            response=response,
            doc_id=ConversationMessage.next_doc_id(),  # Placeholder, not actually stored
            document_type="tool-call",
            document_weight=0.0,
            tokens_used=count_tokens(response),
            timestamp=datetime.now(timezone.utc),
            tool_name=tool_name,
            tool_result=tool_result,
        )

    def _evaluate_conditions(self, step_result: "StepResult") -> str:
        """Evaluate conditions, handle collection, return next step.

        Collection happens FIRST (so final iteration's result isn't lost).
        Iteration limit only affects WHERE we go, not WHETHER we collect.
        """
        step_id = self.step_def.id
        config = self.step_def.config
        executor = self.executor

        # Find matching condition and collect FIRST
        matched_goto = "end"
        for condition in self.step_def.next_conditions:
            if self._condition_matches(condition, step_result):
                # Collect if configured
                if condition.collect_to and step_result.tool_result:
                    executor.state.collect_result(
                        condition.collect_to,
                        step_result.tool_result
                    )
                matched_goto = condition.goto or condition.default or "end"
                break

        # Check iteration limit - overrides goto but NOT collection
        if config.max_iterations and config.on_limit:
            iterations = executor.state.step_iterations.get(step_id, 0)
            if iterations >= config.max_iterations:
                logger.info(
                    f"Step '{step_id}' hit max_iterations ({iterations}), "
                    f"redirecting to '{config.on_limit}'"
                )
                return config.on_limit

        return matched_goto

    def _condition_matches(
        self,
        condition: "Condition",
        step_result: "StepResult"
    ) -> bool:
        """Check if a condition matches the step result."""
        # Default always matches
        if condition.default:
            return True

        # Resolve source field value
        source_value = self._resolve_field(condition.source, step_result)

        # Compare based on operator
        op = condition.condition
        target = condition.target

        if op == "==":
            return str(source_value) == str(target) if source_value is not None else target is None
        elif op == "!=":
            return str(source_value) != str(target) if source_value is not None else target is not None
        elif op == "in":
            return str(source_value) in target if isinstance(target, list) else False
        elif op == "not_in":
            return str(source_value) not in target if isinstance(target, list) else True

        return False

    def _resolve_field(
        self,
        field_path: str,
        step_result: "StepResult"
    ) -> Any:
        """Resolve dotted field path like 'tool_result.accept'."""
        context = {
            "tool_result": step_result.tool_result or {},
            "tool_name": step_result.tool_name,
        }
        parts = field_path.split(".")
        value = context
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value
