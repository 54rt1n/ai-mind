# aim/dreamer/dialogue/scenario.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""DialogueScenario: Executes a DialogueStrategy with proper role flipping."""

from dataclasses import replace
from datetime import datetime
from typing import Optional
import logging
import uuid

from .models import (
    DialogueState,
    DialogueStep,
    DialogueSpeaker,
    DialogueTurn,
    SpeakerType,
)
from .strategy import DialogueStrategy
from ..executor import format_memories_xml, extract_think_tags
from ...agents.persona import Persona
from ...agents.aspects import get_aspect_or_default
from ...config import ChatConfig
from ...constants import DOC_DIALOGUE_CODER, DOC_DIALOGUE_LIBRARIAN
from ...conversation.model import ConversationModel
from ...conversation.message import ConversationMessage
from ...llm.models import LanguageModelV2
from ...utils.tokens import count_tokens

logger = logging.getLogger(__name__)


class DialogueScenario:
    """
    Executes a DialogueStrategy.

    Manages the runtime execution:
    - Tracks dialogue state (accumulated turns)
    - Manages persona and aspect system prompts
    - Builds LLM prompts from strategy
    - Handles role flipping based on speaker perspective
    - Calls LLM and stores results

    The key insight is context engineering: each participant is the assistant
    responding to the other as user. Roles flip based on whose perspective
    we're in for the current step.
    """

    def __init__(
        self,
        strategy: DialogueStrategy,
        persona: Persona,
        config: ChatConfig,
        cvm: Optional[ConversationModel] = None,
    ):
        """
        Initialize the dialogue scenario.

        Args:
            strategy: The dialogue strategy to execute
            persona: The persona participating in the dialogue
            config: Chat configuration with model settings
            cvm: Optional ConversationModel for memory queries and storage
        """
        self.strategy = strategy
        self.persona = persona
        self.config = config
        self.cvm = cvm
        self.state: Optional[DialogueState] = None

    def start(
        self,
        conversation_id: Optional[str] = None,
        guidance: Optional[str] = None,
        query_text: Optional[str] = None,
        user_id: str = "user",
        model: Optional[str] = None,
    ) -> DialogueState:
        """
        Initialize dialogue state for execution.

        Args:
            conversation_id: Optional conversation ID for context
            guidance: Optional user guidance for the scenario
            query_text: Optional query text for memory searches
            user_id: User identifier
            model: Model name to use (defaults to config.default_model)

        Returns:
            Initialized DialogueState
        """
        # Use provided model, or fall back to config.default_model
        model_name = model or getattr(self.config, 'default_model', None) or 'gpt-4'

        self.state = DialogueState(
            pipeline_id=str(uuid.uuid4()),
            strategy_name=self.strategy.name,
            conversation_id=conversation_id,
            persona_id=self.persona.persona_id,
            user_id=user_id,
            model=model_name,
            thought_model=getattr(self.config, 'thought_model', None),
            codex_model=getattr(self.config, 'codex_model', None),
            guidance=guidance,
            query_text=query_text,
        )
        return self.state

    async def execute_step(self, step_id: str) -> DialogueTurn:
        """
        Execute a single dialogue step.

        Process:
        1. Get step from strategy
        2. Determine current speaker and build system prompt
        3. Build turns with role flipping based on perspective
        4. Call LLM with appropriate system prompt
        5. Store result and update state

        Args:
            step_id: ID of the step to execute

        Returns:
            The generated DialogueTurn

        Raises:
            ValueError: If state not initialized
            KeyError: If step not found
        """
        if self.state is None:
            raise ValueError("Dialogue state not initialized. Call start() first.")

        step = self.strategy.get_step(step_id)
        self.state.current_step_id = step_id

        # Build template context for Jinja2 rendering
        template_context = self._build_template_context(step)

        # Get speaker ID for this step
        speaker_id = step.speaker.get_speaker_id(self.persona.persona_id)

        logger.info(
            f"Executing step '{step_id}' | speaker={speaker_id} | "
            f"prior_turns={len(self.state.turns)}"
        )

        # Build system prompt based on speaker type
        system_message = self._build_system_prompt(step.speaker)

        # Query memories if configured
        memories = []
        if step.memory.top_n > 0 and self.cvm is not None:
            memories = self._query_memories(step)

        # Build turns with role flipping
        turns = self._build_turns(step, template_context, memories, speaker_id)

        # Select and configure model
        model_name = self._select_model(step)
        models = LanguageModelV2.index_models(self.config)
        model = models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not available")

        provider = model.llm_factory(self.config)
        max_output_tokens = min(step.config.max_tokens, model.max_output_tokens)

        # Build config for this step
        step_config = replace(
            self.config,
            max_tokens=max_output_tokens,
            temperature=step.config.temperature or self.config.temperature,
            system_message=system_message,
        )

        logger.info(
            f"Step '{step_id}' calling LLM | model={model_name} | "
            f"turns={len(turns)} | max_tokens={max_output_tokens}"
        )

        # Generate response (streaming)
        chunks = []
        for chunk in provider.stream_turns(turns, step_config):
            if chunk:
                chunks.append(chunk)

        response = ''.join(chunks)

        # Extract think tags if present
        response, think = extract_think_tags(response)

        if not response.strip():
            raise ValueError(f"Empty response from model for step '{step_id}'")

        # Determine document type based on speaker
        if step.speaker.type == SpeakerType.ASPECT:
            # Aspect turns use aspect-specific dialogue document type
            aspect_name = step.speaker.aspect_name or self.strategy.dialogue.primary_aspect
            aspect_doc_types = {
                'coder': DOC_DIALOGUE_CODER,
                'librarian': DOC_DIALOGUE_LIBRARIAN,
            }
            document_type = aspect_doc_types.get(aspect_name, DOC_DIALOGUE_CODER)
        else:
            # Persona turns use the step's output document type
            document_type = step.output.document_type

        # Create turn
        doc_id = ConversationMessage.next_doc_id()
        turn = DialogueTurn(
            speaker_id=speaker_id,
            content=response,
            step_id=step_id,
            doc_id=doc_id,
            document_type=document_type,
        )

        # Update state
        self.state.add_turn(turn)
        self.state.completed_steps.append(step_id)
        self.state.step_counter += 1

        logger.info(
            f"Step '{step_id}' complete | "
            f"tokens={count_tokens(response)} | "
            f"response_len={len(response)}"
        )

        return turn

    async def run(self) -> list[DialogueTurn]:
        """
        Execute all steps in the strategy.

        Returns:
            List of all generated DialogueTurns
        """
        if self.state is None:
            raise ValueError("Dialogue state not initialized. Call start() first.")

        execution_order = self.strategy.get_execution_order()
        logger.info(f"Running dialogue '{self.strategy.name}' | steps={execution_order}")

        for step_id in execution_order:
            await self.execute_step(step_id)

        return self.state.turns

    def _build_template_context(self, step: DialogueStep) -> dict:
        """
        Build Jinja2 template context for rendering.

        Args:
            step: Current step being executed

        Returns:
            Context dictionary for Jinja2 templates
        """
        ctx = {
            'persona': self.persona,
            'pronouns': self.persona.pronouns,
            'step_num': self.state.step_counter,
            'guidance': self.state.guidance,
            'query_text': self.state.query_text,
            'conversation_id': self.state.conversation_id,
        }

        # Add required aspects as top-level variables
        for aspect_name in self.strategy.context.required_aspects:
            if aspect_name in self.persona.aspects:
                ctx[aspect_name] = self.persona.aspects[aspect_name]

        # Add primary aspect as 'aspect' for convenience
        primary = self.strategy.dialogue.primary_aspect
        if primary in self.persona.aspects:
            ctx['aspect'] = self.persona.aspects[primary]
        else:
            ctx['aspect'] = get_aspect_or_default(self.persona, primary)

        return ctx

    def _build_system_prompt(self, speaker: DialogueSpeaker) -> str:
        """
        Build system prompt based on speaker type.

        For persona speakers: use standard persona.system_prompt()
        For aspect speakers: build aspect-specific system prompt

        Args:
            speaker: Speaker configuration

        Returns:
            System prompt string
        """
        if speaker.type == SpeakerType.PERSONA:
            return self.persona.system_prompt(
                mood=self.state.persona_mood,
                location=None,
                user_id=self.state.user_id,
            )

        # Aspect speaker - build aspect system prompt
        aspect_name = speaker.aspect_name or self.strategy.dialogue.primary_aspect
        aspect = get_aspect_or_default(self.persona, aspect_name)

        parts = [
            f"{aspect.name}, {aspect.title} - A facet of {self.persona.full_name}.",
            "",
            f"<{aspect.name}>",
        ]

        if hasattr(aspect, 'description') and aspect.description:
            parts.append(f"<description>{aspect.description}</description>")

        if hasattr(aspect, 'appearance') and aspect.appearance:
            parts.append(f"<appearance>{aspect.appearance}</appearance>")

        if hasattr(aspect, 'voice_style') and aspect.voice_style:
            parts.append(f"<voice_style>{aspect.voice_style}</voice_style>")

        if hasattr(aspect, 'core_drive') and aspect.core_drive:
            parts.append(f"<core_drive>{aspect.core_drive}</core_drive>")

        if hasattr(aspect, 'emotional_state') and aspect.emotional_state:
            parts.append(f"<emotional_state>{aspect.emotional_state}</emotional_state>")

        if hasattr(aspect, 'primary_intent') and aspect.primary_intent:
            parts.append(f"<primary_intent>{aspect.primary_intent}</primary_intent>")

        if hasattr(aspect, 'location') and aspect.location:
            parts.append(f"<location>{aspect.location}</location>")

        parts.append(f"</{aspect.name}>")
        parts.append("")

        voice_style = getattr(aspect, 'voice_style', 'thoughtful and measured')
        parts.append(
            f"You are speaking as {aspect.name}, {aspect.title}. "
            f"Your voice style is {voice_style}. "
            f"Embody the aspect fully while maintaining connection to {self.persona.name}'s core identity."
        )

        if self.state.user_id:
            parts.append(f"You are guiding {self.persona.name} through a dialogue.")

        return "\n".join(parts)

    def _build_turns(
        self,
        step: DialogueStep,
        template_context: dict,
        memories: list[dict],
        current_speaker_id: str,
    ) -> list[dict]:
        """
        Build turns with perspective-based role assignment.

        The key insight: roles flip based on whose perspective we're in.
        - If prior turn's speaker == current speaker → role = 'assistant' (my prior words)
        - If prior turn's speaker != current speaker → role = 'user' (their words to me)

        Args:
            step: Current step definition
            template_context: Jinja2 template context
            memories: Memory records to include
            current_speaker_id: Current step's speaker identifier

        Returns:
            List of turn dictionaries with 'role' and 'content'
        """
        turns = []

        # Render scene once
        scene = self.strategy.render_scene(template_context)

        # 1. Add memory context if present
        if memories:
            memory_xml = format_memories_xml(memories)
            turns.append({'role': 'user', 'content': memory_xml})
            wakeup = self.persona.get_wakeup()
            turns.append({'role': 'assistant', 'content': wakeup})

        # 2. Add prior dialogue turns with role flipping
        # Scene is prepended to the FIRST prior turn to establish context
        first_prior = True
        for i, turn in enumerate(self.state.turns):
            if turn.speaker_id == current_speaker_id:
                role = 'assistant'  # My prior words
            else:
                role = 'user'  # Their words to me

            content = turn.content

            # Prepend scene to first prior turn
            if first_prior and scene:
                content = f"{scene}\n\n{content}"
                first_prior = False

            # For the LAST prior turn, append its step's guidance
            # This shapes the current speaker's response
            is_last_turn = (i == len(self.state.turns) - 1)
            if is_last_turn and role == 'user':
                # Get guidance from the prior step
                prior_step = self.strategy.get_step(turn.step_id)
                prior_guidance = self.strategy.render_guidance(prior_step, template_context)
                if prior_guidance:
                    content = f"{content}\n\n[~~ Output Guidance ~~]\n{prior_guidance}"

            turns.append({'role': role, 'content': content})

        # 3. Build final user turn for aspect steps (prompt + guidance for aspect's style)
        # For persona steps responding to aspect, no additional prompt needed
        prompt = self.strategy.render_prompt(step, template_context)

        if step.speaker.type == SpeakerType.ASPECT:
            # Aspect step: needs a prompt
            final_content = []

            if not self.state.turns and scene:
                # First step - include scene to establish context
                final_content.append(scene)

            final_content.append(prompt)

            # Guidance for aspect steps shapes the NEXT response, not this one
            # So we don't append it here - it gets appended when the persona responds

            turns.append({'role': 'user', 'content': '\n\n'.join(final_content)})

        elif not self.state.turns:
            # First step but persona speaking? Include scene + prompt
            final_content = []
            if scene:
                final_content.append(scene)
            if prompt:
                final_content.append(prompt)
            turns.append({'role': 'user', 'content': '\n\n'.join(final_content)})

        # For persona steps with prior turns, no additional user turn needed
        # The prior aspect response (with guidance) IS the user turn

        return turns

    def _query_memories(self, step: DialogueStep) -> list[dict]:
        """
        Query memories from CVM.

        Args:
            step: Current step with memory configuration

        Returns:
            List of memory dictionaries
        """
        if self.cvm is None:
            return []

        from ...constants import CHUNK_LEVEL_768

        query_text = self.state.query_text or step.prompt[:500]
        memories_df = self.cvm.query(
            query_texts=[query_text],
            top_n=step.memory.top_n,
            query_document_type=step.memory.document_type,
            sort_by=step.memory.sort_by,
            chunk_level=CHUNK_LEVEL_768,
        )
        return memories_df.to_dict('records') if not memories_df.empty else []

    def _select_model(self, step: DialogueStep) -> str:
        """
        Select appropriate model for the step.

        Args:
            step: Current step with config

        Returns:
            Model name to use
        """
        if step.config.model_override:
            return step.config.model_override

        if step.config.is_codex and self.state.codex_model:
            return self.state.codex_model

        if step.config.is_thought and self.state.thought_model:
            return self.state.thought_model

        return self.state.model
