# aim/dreamer/core/strategy/standard.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""StandardStrategy - LLM prose generation with document output."""

import logging
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from .base import BaseStepStrategy, ScenarioStepResult, ScenarioExecutor
from .functions import execute_context_actions, load_memory_docs, load_step_docs
from .validation_mixin import FormatValidationMixin

if TYPE_CHECKING:
    from ..models import StandardStepDefinition, StepResult


logger = logging.getLogger(__name__)


class StandardStrategy(FormatValidationMixin, BaseStepStrategy):
    """Executes LLM prose generation - produces document output.

    Used for steps that generate narrative content, summaries, analyses, etc.
    This is the most common strategy for traditional dreamer scenarios.

    Attributes:
        step_def: StandardStepDefinition with prompt, config, output, and next
    """

    step_def: "StandardStepDefinition"

    async def execute(self) -> ScenarioStepResult:
        """Execute LLM generation and create document.

        1. Execute context DSL if present
        2. Build template context and render prompt
        3. Build turns from memory context and prior outputs
        4. Stream LLM response with heartbeat
        5. Extract think tags
        6. Create document in CVM
        7. Record step result in state

        Returns:
            ScenarioStepResult with success=True, doc_created=True, and next step
        """
        step_id = self.step_def.id
        executor = self.executor

        logger.info(
            f"StandardStrategy executing step '{step_id}' "
            f"(model_role={self.step_def.config.model_role or 'default'})"
        )

        # 1. Execute context DSL if present (clears and populates memory_refs)
        if self.step_def.context:
            execute_context_actions(executor, self.step_def)

        # 2. Build and render prompt
        prompt = self._render_prompt()

        # 3. Load memory docs for context
        memory_docs = load_memory_docs(executor)

        # 4. Build turns
        turns, system_message = self._build_turns(prompt, memory_docs)

        # 5. Stream LLM response with format validation
        response, used_fallback = await self._stream_with_format_validation(
            turns, system_message
        )

        # 6. Extract think tags
        from aim.utils.think import extract_think_tags
        response, think = extract_think_tags(response)

        if used_fallback:
            logger.info(f"Step '{step_id}' used fallback model for format compliance")

        # 7. Create document
        doc_id = await self._create_document(response, think)

        # 8. Record step result
        step_result = self._create_step_result(doc_id, response, think)
        executor.state.record_step_result(step_result)

        # 9. Record turn for conversation history
        executor.state.record_turn(step_id, prompt, response)

        # Track document
        executor.state.add_doc_id(doc_id)

        # Get next step
        next_step = self._get_next_step()

        logger.info(
            f"Step '{step_id}' complete, "
            f"created doc '{doc_id}' ({len(response)} chars), "
            f"has_think={think is not None}, "
            f"next_step='{next_step}'"
        )

        return ScenarioStepResult(
            success=True,
            next_step=next_step,
            state_changed=True,
            doc_created=True,
        )

    def _render_prompt(self) -> str:
        """Render the step prompt with template context."""
        from ..scenario import render_template

        executor = self.executor

        # Build context
        ctx = executor.state.build_template_context(
            framework=executor.framework,
            persona=executor.persona,
        )
        ctx['persona'] = executor.persona
        ctx['pronouns'] = executor.persona.pronouns

        return render_template(self.step_def.prompt, ctx)

    def _build_turns(
        self,
        prompt: str,
        memory_docs: list[dict],
    ) -> tuple[list[dict], str]:
        """Build turns list and system message for LLM.

        Args:
            prompt: Rendered prompt string
            memory_docs: Documents from memory context

        Returns:
            Tuple of (turns list, system_message)
        """
        from ..executor import build_turns

        executor = self.executor

        # Get model limits
        model = self._get_model()
        max_context = getattr(model, 'max_tokens', 32768)
        max_output = min(
            self.step_def.config.max_tokens,
            getattr(model, 'max_output_tokens', 4096)
        )

        # Build prior outputs from step outputs
        prior_outputs = load_step_docs(executor)

        # Get thought content from state if available
        thought_content = executor.state.thought_content if executor.state else None

        return build_turns(
            state=None,  # We'll build from state manually
            prompt=prompt,
            memories=memory_docs,
            prior_outputs=prior_outputs,
            persona=executor.persona,
            max_context_tokens=max_context,
            max_output_tokens=max_output,
            thought_content=thought_content,
        )

    def _get_model(self):
        """Get the language model for this step."""
        from aim.llm.models import LanguageModelV2

        executor = self.executor
        step_config = self.step_def.config

        # Determine model name
        if step_config.model_override:
            model_name = step_config.model_override
        elif step_config.model_role:
            model_name = executor.model_set.get_model_name(step_config.model_role)
        elif step_config.is_thought:
            model_name = executor.model_set.thought_model
        elif step_config.is_codex:
            model_name = executor.model_set.codex_model
        else:
            model_name = executor.model_set.default_model

        # Get model object
        models = LanguageModelV2.index_models(executor.config)
        return models.get(model_name)

    def _get_model_by_role(self, role: str):
        """Get a language model by role name.

        Args:
            role: Model role name (e.g., "fallback", "thought", "codex")

        Returns:
            LanguageModelV2 instance or None if not found
        """
        from aim.llm.models import LanguageModelV2

        executor = self.executor
        model_name = executor.model_set.get_model_name(role)
        if not model_name:
            return None

        models = LanguageModelV2.index_models(executor.config)
        return models.get(model_name)

    async def _stream_response_inner(
        self,
        turns: list[dict],
        system_message: str,
    ) -> str:
        """Stream LLM response with heartbeat using the default model.

        Args:
            turns: Conversation turns
            system_message: System prompt

        Returns:
            Complete response string
        """
        model = self._get_model()
        if not model:
            raise ValueError(f"Model not available for step {self.step_def.id}")

        return await self._stream_with_model_obj(turns, system_message, model)

    async def _stream_response_with_model(
        self,
        turns: list[dict],
        system_message: str,
        model_role: str,
    ) -> str:
        """Stream LLM response using a specific model role.

        Args:
            turns: Conversation turns
            system_message: System prompt
            model_role: Model role name (e.g., "fallback")

        Returns:
            Complete response string
        """
        model = self._get_model_by_role(model_role)
        if not model:
            raise ValueError(f"Model role '{model_role}' not available for step {self.step_def.id}")

        return await self._stream_with_model_obj(turns, system_message, model)

    async def _stream_with_model_obj(
        self,
        turns: list[dict],
        system_message: str,
        model,
    ) -> str:
        """Stream LLM response with a specific model object.

        Args:
            turns: Conversation turns
            system_message: System prompt
            model: LanguageModelV2 instance

        Returns:
            Complete response string
        """
        executor = self.executor
        step_config = self.step_def.config

        provider = model.llm_factory(executor.config)

        # Build step config
        llm_config = replace(
            executor.config,
            system_message=system_message,
            max_tokens=min(step_config.max_tokens, model.max_output_tokens),
            temperature=step_config.temperature or executor.config.temperature,
        )

        # Stream with heartbeat
        chunks = []
        chunk_count = 0
        for chunk in provider.stream_turns(turns, llm_config):
            if chunk:
                chunks.append(chunk)
                chunk_count += 1

                # Heartbeat every 50 chunks
                if chunk_count % 50 == 0:
                    await self._heartbeat()

        return ''.join(chunks)

    async def _create_document(self, content: str, think: Optional[str]) -> str:
        """Create document in CVM.

        Args:
            content: Document content
            think: Optional think content

        Returns:
            Document ID
        """
        from aim.conversation.message import ConversationMessage

        executor = self.executor
        step_def = self.step_def

        doc_id = ConversationMessage.next_doc_id()

        message = ConversationMessage.create(
            doc_id=doc_id,
            conversation_id=executor.state.conversation_id,
            user_id=executor.state.user_id,
            persona_id=executor.persona.persona_id,
            sequence_no=0,
            branch=executor.state.branch,
            role='assistant',
            content=content,
            think=think,
            document_type=step_def.output.document_type,
            weight=step_def.output.weight,
            speaker_id=executor.persona.persona_id,
            inference_model=self._get_model().name if self._get_model() else None,
            scenario_name=executor.framework.name,
            step_name=step_def.id,
        )

        executor.insert_message(message)
        return doc_id

    def _create_step_result(
        self,
        doc_id: str,
        response: str,
        think: Optional[str],
    ) -> "StepResult":
        """Create StepResult for recording in state."""
        from ..models import StepResult
        from aim.utils.tokens import count_tokens

        return StepResult(
            step_id=self.step_def.id,
            response=response,
            think=think,
            doc_id=doc_id,
            document_type=self.step_def.output.document_type,
            document_weight=self.step_def.output.weight,
            tokens_used=count_tokens(response),
            timestamp=datetime.now(timezone.utc),
        )
