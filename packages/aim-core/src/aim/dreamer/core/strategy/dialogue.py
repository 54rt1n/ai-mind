# aim/dreamer/core/strategy/dialogue.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""DialogueStrategy - LLM dialogue with speaker-based role flipping."""

import logging
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from jinja2 import Template

from aim.agents.aspects import get_aspect_or_default
from aim.conversation.message import ConversationMessage
from aim.utils.think import extract_think_tags
from aim.utils.tokens import count_tokens

from .base import BaseStepStrategy, ScenarioStepResult
from .functions import execute_context_actions, load_memory_docs

if TYPE_CHECKING:
    from ..models import DialogueStepDefinition, StepResult, DialogueTurn


logger = logging.getLogger(__name__)


# Document type constants for aspects
ASPECT_DOC_TYPES = {
    'artist': 'dialogue-artist',
    'coder': 'dialogue-coder',
    'dreamer': 'dialogue-dreamer',
    'librarian': 'dialogue-librarian',
    'philosopher': 'dialogue-philosopher',
    'psychologist': 'dialogue-psychologist',
    'revelator': 'dialogue-revelator',
    'writer': 'dialogue-writer',
}


class DialogueStrategy(BaseStepStrategy):
    """Executes LLM dialogue with speaker-based role flipping.

    Used for persona/aspect dialogues where roles flip based on who speaks:
    - When ASPECT speaks: aspects='assistant', persona='user'
    - When PERSONA speaks: aspects='user', persona='assistant'

    This ensures proper user/assistant alternation for the LLM.

    Attributes:
        step_def: DialogueStepDefinition with speaker, guidance, output, and next
    """

    step_def: "DialogueStepDefinition"

    async def execute(self) -> ScenarioStepResult:
        """Execute dialogue step with role flipping.

        1. Execute context DSL if present
        2. Build template context with persona, aspects, pronouns
        3. Render guidance with Jinja2
        4. Build system prompt based on speaker type
        5. Build dialogue turns with role flipping
        6. Stream LLM response with heartbeat
        7. Extract think tags
        8. Create document in CVM with speaker metadata
        9. Record dialogue turn in state

        Returns:
            ScenarioStepResult with success=True, doc_created=True, and next step
        """
        step_id = self.step_def.id
        executor = self.executor
        persona = executor.persona
        state = executor.state

        # Determine speaker
        from ..models import SpeakerType
        speaker = self.step_def.speaker
        speaker_id = speaker.get_speaker_id(persona.persona_id)
        is_persona_speaker = speaker.type == SpeakerType.PERSONA

        logger.info(
            f"DialogueStrategy executing step '{step_id}' | "
            f"speaker={speaker_id} | prior_turns={len(state.dialogue_turns)}"
        )

        # 1. Execute context DSL if present
        if self.step_def.context:
            execute_context_actions(executor, self.step_def)

        # 2. Build template context
        ctx = self._build_template_context()

        # 3. Render guidance
        guidance = self._render_guidance(ctx)

        # 4. Build system prompt based on speaker type
        system_message = self._build_system_prompt(speaker)

        # 5. Load memory docs for context
        memory_docs = load_memory_docs(executor)

        # 6. Build turns with role flipping
        turns = self._build_dialogue_turns(
            guidance=guidance,
            memory_docs=memory_docs,
            is_persona_speaker=is_persona_speaker,
        )

        # 7. Stream LLM response
        response = await self._stream_response(turns, system_message)

        # 8. Extract think tags
        response, think = extract_think_tags(response)

        if not response.strip():
            raise ValueError(f"Empty response from model for step '{step_id}'")

        # 9. Create document in CVM
        doc_id = await self._create_document(response, think, speaker_id)

        # 10. Record dialogue turn in state
        self._record_dialogue_turn(
            content=response,
            think=think,
            doc_id=doc_id,
            speaker_id=speaker_id,
        )

        # 11. Create step result for state tracking
        step_result = self._create_step_result(doc_id, response, think)
        executor.state.record_step_result(step_result)

        # Track document
        executor.state.add_doc_id(doc_id)

        # Get next step
        next_step = self._get_next_step()

        logger.info(
            f"Step '{step_id}' complete | "
            f"created doc '{doc_id}' ({len(response)} chars) | "
            f"has_think={think is not None} | "
            f"next_step='{next_step}'"
        )

        return ScenarioStepResult(
            success=True,
            next_step=next_step,
            state_changed=True,
            doc_created=True,
        )

    def _build_template_context(self) -> dict:
        """Build Jinja2 template context with persona, aspects, and pronouns.

        Returns:
            Context dictionary for template rendering
        """
        executor = self.executor
        persona = executor.persona
        state = executor.state

        # Start with state's template context
        ctx = state.build_template_context()

        # Add persona and pronouns
        ctx['persona'] = persona
        ctx['pronouns'] = persona.pronouns

        # Add step number
        ctx['step_num'] = len(state.dialogue_turns) + 1

        # Add required aspects from dialogue config
        if executor.framework.dialogue:
            for aspect_name in executor.framework.dialogue.required_aspects:
                if aspect_name in persona.aspects:
                    ctx[aspect_name] = persona.aspects[aspect_name]
                else:
                    ctx[aspect_name] = get_aspect_or_default(persona, aspect_name)

            # Add primary aspect as 'aspect' for convenience
            primary = executor.framework.dialogue.primary_aspect
            if primary in persona.aspects:
                ctx['aspect'] = persona.aspects[primary]
            else:
                ctx['aspect'] = get_aspect_or_default(persona, primary)

        return ctx

    def _render_guidance(self, ctx: dict) -> str:
        """Render the guidance template with context.

        Args:
            ctx: Template context dictionary

        Returns:
            Rendered guidance string
        """
        if not self.step_def.guidance:
            return ""

        return Template(self.step_def.guidance).render(ctx)

    def _build_system_prompt(self, speaker) -> str:
        """Build system prompt based on speaker type.

        For persona speakers: use standard persona.system_prompt()
        For aspect speakers: build aspect-specific system prompt

        Args:
            speaker: DialogueSpeaker configuration

        Returns:
            System prompt string
        """
        from ..models import SpeakerType

        executor = self.executor
        persona = executor.persona
        state = executor.state

        if speaker.type == SpeakerType.PERSONA:
            return persona.system_prompt(
                mood=None,
                location=None,
                user_id="system",
            )

        # Aspect speaker - build aspect system prompt
        aspect_name = speaker.aspect_name
        if not aspect_name and executor.framework.dialogue:
            aspect_name = executor.framework.dialogue.primary_aspect

        aspect = get_aspect_or_default(persona, aspect_name)

        parts = [
            f"{aspect.name}, {aspect.title} - A facet of {persona.full_name}.",
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
            f"Embody the aspect fully while maintaining connection to {persona.name}'s core identity."
        )

        parts.append(f"You are guiding {persona.name} through a dialogue.")

        return "\n".join(parts)

    def _build_dialogue_turns(
        self,
        guidance: str,
        memory_docs: list[dict],
        is_persona_speaker: bool,
    ) -> list[dict]:
        """Build turns list with proper role flipping.

        Role assignment rules:
        - For ASPECT speakers: all aspects = 'assistant', persona = 'user'
        - For PERSONA speakers: all aspects = 'user', persona = 'assistant'

        Args:
            guidance: Rendered guidance for this step
            memory_docs: Documents from memory context
            is_persona_speaker: Whether current speaker is persona

        Returns:
            List of turn dicts with 'role' and 'content'
        """
        executor = self.executor
        state = executor.state
        persona = executor.persona

        turns = []

        # 1. Add memory context as first user turn if present
        if memory_docs:
            context_xml = self._format_context_docs(memory_docs)
            turns.append({'role': 'user', 'content': context_xml})

            # Add wakeup after context if needed
            wakeup = persona.get_wakeup()
            if wakeup:
                turns.append({'role': 'assistant', 'content': wakeup})

        # 2. Build dialogue turns with proper roles
        dialogue_turn_contents = []
        last_aspect_name = None

        for turn in state.dialogue_turns:
            role = state.get_dialogue_role(turn.speaker_id, is_persona_speaker)
            content = turn.content

            # Scene handling: prepend at aspect changes for persona speakers
            if is_persona_speaker and turn.speaker_id.startswith("aspect:"):
                current_aspect = turn.speaker_id.split(":", 1)[1]
                if current_aspect != last_aspect_name:
                    # New aspect appearing - generate scene
                    scene = self._generate_scene(current_aspect)
                    if scene:
                        content = f"{scene}\n\n{content}"
                    last_aspect_name = current_aspect

            dialogue_turn_contents.append((role, content))

        # 3. Add dialogue turns to the turns list
        for role, content in dialogue_turn_contents:
            turns.append({'role': role, 'content': content})

        # 4. Add guidance as final user turn
        if not dialogue_turn_contents:
            # No prior dialogue - add scene if persona speaker facing aspect
            final_content = []
            if is_persona_speaker and self.step_def.speaker.aspect_name:
                scene = self._generate_scene(self.step_def.speaker.aspect_name)
                if scene:
                    final_content.append(scene)

            if guidance:
                final_content.append(f"[~~ Output Guidance ~~]\n{guidance}")

            if final_content:
                turns.append({'role': 'user', 'content': '\n\n'.join(final_content)})
        else:
            # Has prior turns - append guidance to final user turn or add new one
            if dialogue_turn_contents[-1][0] == 'user':
                # Last turn was user - append guidance
                last_role, last_content = dialogue_turn_contents[-1]
                if guidance:
                    turns[-1]['content'] = f"{last_content}\n\n[~~ Output Guidance ~~]\n{guidance}"
            else:
                # Last turn was assistant - add guidance as new user turn
                if guidance:
                    turns.append({'role': 'user', 'content': f"[~~ Output Guidance ~~]\n{guidance}"})
                else:
                    turns.append({'role': 'user', 'content': '[Continue]'})

        # Final safety check - must end with user
        if turns and turns[-1]['role'] != 'user':
            if guidance:
                turns.append({'role': 'user', 'content': f"[~~ Output Guidance ~~]\n{guidance}"})
            else:
                turns.append({'role': 'user', 'content': '[Continue]'})

        return turns

    def _generate_scene(self, aspect_name: str) -> str:
        """Generate scene description for aspect.

        Args:
            aspect_name: Name of the aspect to describe

        Returns:
            Scene description string
        """
        executor = self.executor
        persona = executor.persona

        aspect = get_aspect_or_default(persona, aspect_name)

        location = getattr(aspect, 'location', 'a thoughtful space')
        appearance = getattr(aspect, 'appearance', 'a familiar presence')
        emotional_state = getattr(aspect, 'emotional_state', 'calm focus')
        name = getattr(aspect, 'name', aspect_name)
        title = getattr(aspect, 'title', 'Guide')

        return (
            f"*Your awareness shifts to {location}. "
            f"Before you stands {name}, your {title}. {appearance}. "
            f"Their presence radiates {emotional_state}.*"
        )

    def _format_context_docs(self, docs: list[dict]) -> str:
        """Format context documents as XML.

        Args:
            docs: List of document dicts with 'content' field

        Returns:
            XML string with context wrapped in tags
        """
        if not docs:
            return "<context>\n</context>"

        lines = ["<context>"]
        for doc in docs:
            content = doc.get('content', '')
            lines.append(f"  <document>{content}</document>")
        lines.append("</context>")

        return "\n".join(lines)

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

    async def _stream_response(
        self,
        turns: list[dict],
        system_message: str,
    ) -> str:
        """Stream LLM response with heartbeat.

        Args:
            turns: Conversation turns
            system_message: System prompt

        Returns:
            Complete response string
        """
        executor = self.executor
        step_config = self.step_def.config

        # Get provider
        model = self._get_model()
        if not model:
            raise ValueError(f"Model not available for step {self.step_def.id}")

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

    async def _create_document(
        self,
        content: str,
        think: Optional[str],
        speaker_id: str,
    ) -> str:
        """Create document in CVM with speaker metadata.

        Args:
            content: Document content
            think: Optional think content
            speaker_id: Speaker identifier for metadata

        Returns:
            Document ID
        """
        from ..models import SpeakerType

        executor = self.executor
        step_def = self.step_def
        speaker = step_def.speaker

        doc_id = ConversationMessage.next_doc_id()

        # Determine document type based on speaker
        if speaker.type == SpeakerType.ASPECT:
            aspect_name = speaker.aspect_name
            if not aspect_name and executor.framework.dialogue:
                aspect_name = executor.framework.dialogue.primary_aspect
            document_type = ASPECT_DOC_TYPES.get(aspect_name, f'dialogue-{aspect_name}')
        else:
            document_type = step_def.output.document_type

        message = ConversationMessage.create(
            doc_id=doc_id,
            conversation_id=executor.state.conversation_id,
            user_id="system",
            persona_id=executor.persona.persona_id,
            sequence_no=len(executor.state.dialogue_turns) + 1,
            branch=0,
            role='assistant',
            content=content,
            think=think,
            document_type=document_type,
            weight=step_def.output.weight,
            speaker_id=speaker_id,
            inference_model=self._get_model().name if self._get_model() else None,
            scenario_name=executor.framework.name,
            step_name=step_def.id,
        )

        executor.cvm.insert(message)
        return doc_id

    def _record_dialogue_turn(
        self,
        content: str,
        think: Optional[str],
        doc_id: str,
        speaker_id: str,
    ) -> None:
        """Record dialogue turn in state.

        Args:
            content: Turn content
            think: Optional think content
            doc_id: Document ID
            speaker_id: Speaker identifier
        """
        from ..models import DialogueTurn, SpeakerType

        speaker = self.step_def.speaker

        # Determine document type for the turn
        if speaker.type == SpeakerType.ASPECT:
            aspect_name = speaker.aspect_name
            if not aspect_name and self.executor.framework.dialogue:
                aspect_name = self.executor.framework.dialogue.primary_aspect
            document_type = ASPECT_DOC_TYPES.get(aspect_name, f'dialogue-{aspect_name}')
        else:
            document_type = self.step_def.output.document_type

        # Determine speaker_type for the turn
        speaker_type = SpeakerType.PERSONA if speaker_id.startswith("persona:") else SpeakerType.ASPECT

        turn = DialogueTurn(
            speaker_id=speaker_id,
            content=content,
            think=think,
            step_id=self.step_def.id,
            doc_id=doc_id,
            document_type=document_type,
        )

        self.executor.state.add_dialogue_turn(turn)

    def _create_step_result(
        self,
        doc_id: str,
        response: str,
        think: Optional[str],
    ) -> "StepResult":
        """Create StepResult for recording in state."""
        from ..models import StepResult

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
