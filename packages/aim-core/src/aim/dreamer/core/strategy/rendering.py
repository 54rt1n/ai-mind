# aim/dreamer/core/strategy/rendering.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""RenderingStrategy - template rendering to document, no LLM."""

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .base import BaseStepStrategy, ScenarioStepResult, ScenarioExecutor

if TYPE_CHECKING:
    from ..models import RenderingStepDefinition


logger = logging.getLogger(__name__)


class RenderingStrategy(BaseStepStrategy):
    """Renders a Jinja2 template to create a document - no LLM call.

    Used for steps that produce output by rendering templates with
    accumulated data (e.g., rendering a plan from collected tasks).

    Attributes:
        step_def: RenderingStepDefinition with template, output, and next
    """

    step_def: "RenderingStepDefinition"

    async def execute(self) -> ScenarioStepResult:
        """Render template and create document.

        1. Build template context from state (steps, collections, guidance, etc.)
        2. Render Jinja2 template with context
        3. Create document in CVM
        4. Return result with next step

        Returns:
            ScenarioStepResult with success=True, doc_created=True, and next step
        """
        step_id = self.step_def.id
        executor = self.executor

        logger.info(f"RenderingStrategy executing step '{step_id}'")

        # Build template context from state
        ctx = self._build_template_context()

        # Render template
        rendered = self._render_template(ctx)

        # Create document
        doc_id = await self._create_document(rendered)

        # Track document in state
        executor.state.add_doc_id(doc_id)

        # Get next step
        next_step = self._get_next_step()

        logger.info(
            f"Step '{step_id}' complete, "
            f"created doc '{doc_id}' ({len(rendered)} chars), "
            f"next_step='{next_step}'"
        )

        return ScenarioStepResult(
            success=True,
            next_step=next_step,
            state_changed=True,  # Added doc_id to state
            doc_created=True,
        )

    def _build_template_context(self) -> dict:
        """Build Jinja2 context from state and persona.

        Returns:
            Dictionary with template variables
        """
        executor = self.executor

        # Start with state's template context (steps, collections, etc.)
        ctx = executor.state.build_template_context()

        # Add persona
        ctx['persona'] = executor.persona
        ctx['pronouns'] = executor.persona.pronouns

        return ctx

    def _render_template(self, ctx: dict) -> str:
        """Render the Jinja2 template with context.

        Args:
            ctx: Template context dictionary

        Returns:
            Rendered template string
        """
        from ..scenario import render_template
        return render_template(self.step_def.template, ctx)

    async def _create_document(self, content: str) -> str:
        """Create document in CVM.

        Args:
            content: Rendered document content

        Returns:
            Document ID
        """
        from aim.conversation.message import ConversationMessage

        executor = self.executor
        step_def = self.step_def

        # Generate doc_id
        doc_id = ConversationMessage.next_doc_id()

        # Create message with output configuration
        message = ConversationMessage.create(
            doc_id=doc_id,
            conversation_id=executor.state.conversation_id,
            user_id="system",  # Rendering is system-generated
            persona_id=executor.persona.id,
            sequence_no=0,  # Not part of conversation sequence
            branch=0,
            role='assistant',
            content=content,
            document_type=step_def.output.document_type,
            weight=step_def.output.weight,
            speaker_id=executor.persona.id,
            scenario_name=executor.framework.name,
            step_name=step_def.id,
        )

        # Insert into CVM
        executor.cvm.insert(message)

        return doc_id
