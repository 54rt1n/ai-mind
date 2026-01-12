# aim/dreamer/core/dialogue/scenario.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""DialogueScenario: Executes a DialogueStrategy with proper role flipping."""

from dataclasses import replace
from datetime import datetime
from typing import Awaitable, Callable, Optional, TYPE_CHECKING
import logging
import uuid

if TYPE_CHECKING:
    from aim.llm.model_set import ModelSet

from ....utils.think import extract_think_tags
from ....agents.persona import Persona
from ....agents.aspects import get_aspect_or_default
from ....config import ChatConfig
from ....constants import (
    DOC_DIALOGUE_ARTIST,
    DOC_DIALOGUE_CODER,
    DOC_DIALOGUE_DREAMER,
    DOC_DIALOGUE_LIBRARIAN,
    DOC_DIALOGUE_PHILOSOPHER,
    DOC_DIALOGUE_PSYCHOLOGIST,
    DOC_DIALOGUE_REVELATOR,
    DOC_DIALOGUE_WRITER,
)
from ....conversation.model import ConversationModel
from ....conversation.message import ConversationMessage
from ....llm.models import LanguageModelV2
from ....utils.tokens import count_tokens
from ....utils.redis_cache import RedisCache

from ..executor import load_prior_outputs, format_memories_xml

from .strategy import DialogueStrategy
from .models import (
    DialogueState,
    DialogueStep,
    DialogueSpeaker,
    DialogueTurn,
    SpeakerType,
)

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
        model_set: "ModelSet",
        cvm: Optional[ConversationModel] = None,
        heartbeat_callback: Optional[Callable[[str, str], Awaitable[None]]] = None,
    ):
        """
        Initialize the dialogue scenario.

        Args:
            strategy: The dialogue strategy to execute
            persona: The persona participating in the dialogue
            config: Chat configuration with model settings
            model_set: ModelSet for persona-aware model selection
            cvm: Optional ConversationModel for memory queries and storage
            heartbeat_callback: Optional callback called every 50 chunks during streaming
                               for liveness detection (ANDIMUD inline execution only)
        """
        self.strategy = strategy
        self.persona = persona
        self.config = config
        self.cvm = cvm
        self.model_set = model_set
        self.heartbeat_callback = heartbeat_callback
        self.state: Optional[DialogueState] = None

    def start(
        self,
        conversation_id: Optional[str] = None,
        guidance: Optional[str] = None,
        query_text: Optional[str] = None,
        user_id: str = "user",
        model: Optional[str] = None,
        branch: Optional[int] = None,
    ) -> DialogueState:
        """
        Initialize dialogue state for execution.

        Args:
            conversation_id: Optional conversation ID for context
            guidance: Optional user guidance for the scenario
            query_text: Optional query text for memory searches
            user_id: User identifier
            model: Model name to use (defaults to config.default_model)
            branch: Branch number for conversation storage (defaults to 0)

        Returns:
            Initialized DialogueState
        """
        # Use provided model, or fall back to config.default_model
        model_name = model or getattr(self.config, 'default_model', None) or 'gpt-4'

        # Get branch from cvm if available, else use provided or default to 0
        if branch is None and self.cvm is not None and conversation_id:
            branch = self.cvm.get_next_branch(conversation_id)
        branch = branch or 0

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
            branch=branch,
        )
        return self.state

    async def execute_step(self, step_id: str) -> DialogueTurn:
        """
        Execute a single dialogue step.

        Mirrors executor.execute_step flow:
        1. Build template context
        2. Render prompt
        3. Load context (from DSL or accumulated)
        4. Query memories
        5. Select model
        6. Build turns with role flipping
        7. Build config
        8. Stream LLM response
        9. Extract think tags
        10. Return DialogueTurn

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

        # Log step start
        step_flags = []
        if step.config.is_codex:
            step_flags.append("codex")
        if step.config.is_thought:
            step_flags.append("thought")
        flags_str = f" [{', '.join(step_flags)}]" if step_flags else ""

        logger.info(
            f"Step '{step_id}' starting{flags_str} | speaker={speaker_id} | "
            f"prior_turns={len(self.state.turns)}"
        )

        # 1. Render guidance for this step
        guidance = self.strategy.render_guidance(step, template_context)

        # 2. Load context using existing executor function
        prior_outputs, context_doc_ids, is_initial_context = load_prior_outputs(
            self.state, step, self.cvm
        )

        # Update state with context doc IDs if this is initial context
        if is_initial_context:
            self.state.context_doc_ids = context_doc_ids

        # 3. Memory retrieval now handled by context DSL in load_prior_outputs
        # prior_outputs includes both context docs and memories from search_memories actions
        memories: list[dict] = []

        # 4. Select and configure model
        model_name = self._select_model(step)
        models = LanguageModelV2.index_models(self.config)
        model = models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not available")

        provider = model.llm_factory(self.config)
        max_output_tokens = min(step.config.max_tokens, model.max_output_tokens)

        logger.info(
            f"Step '{step_id}' using model={model_name} | "
            f"memories={len(memories)} context_docs={len(prior_outputs)} "
            f"max_tokens={max_output_tokens}"
        )

        # 5. Build turns with dialogue-specific role flipping
        turns, system_message = self.build_dialogue_turns(
            step=step,
            template_context=template_context,
            guidance=guidance,
            memories=memories,
            context_docs=prior_outputs,
            speaker_id=speaker_id,
            max_context_tokens=model.max_tokens,
            max_output_tokens=max_output_tokens,
        )

        # 6. Build config for this step
        step_config = replace(
            self.config,
            max_tokens=max_output_tokens,
            temperature=step.config.temperature or self.config.temperature,
            system_message=system_message,
        )

        logger.info(
            f"Step '{step_id}' calling LLM | "
            f"turns={len(turns)} | max_tokens={max_output_tokens}"
        )

        # 7. Generate response (streaming)
        chunks = []
        chunk_count = 0
        for chunk in provider.stream_turns(turns, step_config):
            if chunk:
                chunks.append(chunk)
                chunk_count += 1

                # Heartbeat every 50 chunks for liveness detection
                if chunk_count % 50 == 0 and self.heartbeat_callback:
                    try:
                        await self.heartbeat_callback(self.state.pipeline_id, step_id)
                    except Exception as e:
                        logger.warning(
                            f"Heartbeat callback failed at chunk {chunk_count}: {e}",
                            exc_info=True
                        )

        response = ''.join(chunks)

        # 8. Extract think tags if present
        response, think = extract_think_tags(response)

        # 9. Validate response
        if not response.strip():
            raise ValueError(f"Empty response from model for step '{step_id}'")

        # Determine document type based on speaker
        # Aspect steps default to dialogue-{aspect}, but YAML can override with specific types
        if step.speaker.type == SpeakerType.ASPECT:
            # Check if YAML specifies a non-default document type (override)
            if step.output.document_type not in ('step', 'dialogue'):
                document_type = step.output.document_type
            else:
                aspect_name = step.speaker.aspect_name or self.strategy.dialogue.primary_aspect
                aspect_doc_types = {
                    'artist': DOC_DIALOGUE_ARTIST,
                    'coder': DOC_DIALOGUE_CODER,
                    'dreamer': DOC_DIALOGUE_DREAMER,
                    'librarian': DOC_DIALOGUE_LIBRARIAN,
                    'philosopher': DOC_DIALOGUE_PHILOSOPHER,
                    'psychologist': DOC_DIALOGUE_PSYCHOLOGIST,
                    'revelator': DOC_DIALOGUE_REVELATOR,
                    'writer': DOC_DIALOGUE_WRITER,
                }
                document_type = aspect_doc_types.get(aspect_name, DOC_DIALOGUE_CODER)
        else:
            document_type = step.output.document_type

        # 10. Create turn
        doc_id = ConversationMessage.next_doc_id()
        turn = DialogueTurn(
            speaker_id=speaker_id,
            content=response,
            think=think,
            step_id=step_id,
            doc_id=doc_id,
            document_type=document_type,
        )

        # 11. Save to CVM (using state.branch like standard executor)
        if self.cvm is not None:
            message = ConversationMessage.create(
                doc_id=doc_id,
                conversation_id=self.state.conversation_id,
                user_id=self.state.user_id,
                persona_id=self.state.persona_id,
                sequence_no=self.state.step_counter,  # Sequential turns in dialogue
                branch=self.state.branch,
                role='assistant',
                content=response,
                think=think,
                document_type=document_type,
                weight=step.output.weight,
                speaker_id=speaker_id,
                inference_model=model_name,
                scenario_name=self.strategy.name,
                step_name=step_id,
            )
            self.cvm.insert(message)
            logger.info(f"Saved turn to CVM: doc_id={doc_id} type={document_type}")

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

    def build_dialogue_turns(
        self,
        step: DialogueStep,
        template_context: dict,
        guidance: str,
        memories: list[dict],
        context_docs: list[dict],
        speaker_id: str,
        max_context_tokens: int,
        max_output_tokens: int,
    ) -> tuple[list[dict], str]:
        """
        Build turns with proper role assignment and token budgeting.

        Role assignment rules:
        - For ASPECT speakers: all aspects = 'assistant', persona = 'user'
        - For PERSONA speakers: all aspects = 'user', persona = 'assistant'

        This ensures proper user/assistant alternation since dialogue naturally
        alternates between aspect and persona.

        Additional rules:
        - Wakeup: only include if first dialogue turn is 'user'
        - Scene: prepend at aspect changes (persona steps only)
        - Guidance: append to last 'user' turn

        Args:
            step: Current step definition
            template_context: Jinja2 template context
            guidance: Rendered guidance for current step
            memories: Memory records from CVM query
            context_docs: Context documents (conversation to analyze)
            speaker_id: Current speaker's identifier
            max_context_tokens: Model's max context window
            max_output_tokens: Max tokens to generate

        Returns:
            Tuple of (turns list, system_message string)
        """
        turns = []
        is_persona_speaker = step.speaker.type == SpeakerType.PERSONA

        # Build system message based on speaker type
        system_message = self._build_system_prompt(step.speaker)

        # Calculate available tokens (mirroring executor.build_turns)
        safety_margin = 1024
        available_tokens = max_context_tokens - max_output_tokens - safety_margin

        wakeup = self.persona.get_wakeup()
        fixed_tokens = (
            count_tokens(system_message) +
            count_tokens(guidance) +
            count_tokens(wakeup)
        )

        remaining_budget = available_tokens - fixed_tokens

        # Calculate tokens for context docs and dialogue turns
        context_tokens = sum(count_tokens(d.get('content', '')) for d in context_docs)
        dialogue_tokens = sum(count_tokens(t.content) for t in self.state.turns)

        # Eviction priority: context docs first, then memories
        # (No thought stream in dialogue - each turn stands alone)

        # 1. Evict oldest context docs if over budget
        trimmed_context = list(context_docs)
        while trimmed_context and context_tokens + dialogue_tokens > remaining_budget:
            removed = trimmed_context.pop(0)
            context_tokens -= count_tokens(removed.get('content', ''))
            logger.info(f"Evicting context doc to fit budget")

        # 2. Budget remaining for memories
        memory_budget = remaining_budget - context_tokens - dialogue_tokens
        trimmed_memories = []
        memory_tokens = 0
        for mem in reversed(memories):
            content = mem.get('content', '')
            mem_tokens = count_tokens(content)
            if memory_tokens + mem_tokens <= memory_budget:
                trimmed_memories.insert(0, mem)
                memory_tokens += mem_tokens
            else:
                break

        if len(trimmed_memories) < len(memories):
            logger.info(f"Trimmed memories: {len(memories)} -> {len(trimmed_memories)}")

        # Helper to determine role for a dialogue turn
        def get_role_for_turn(turn: DialogueTurn) -> str:
            """
            Determine role based on speaker type relationship.
            - For persona speaker: aspects = 'user', persona = 'assistant'
            - For aspect speaker: aspects = 'assistant', persona = 'user'
            """
            turn_is_persona = turn.speaker_id.startswith("persona:")
            if is_persona_speaker:
                return 'assistant' if turn_is_persona else 'user'
            else:
                return 'user' if turn_is_persona else 'assistant'

        # Determine if first dialogue turn is 'user' (for wakeup decision)
        first_dialogue_role = None
        if self.state.turns:
            first_dialogue_role = get_role_for_turn(self.state.turns[0])

        # Build turns list

        # 1. Add memories as context (user turn)
        context_parts = []
        if trimmed_context:
            context_parts.append(self._format_context_docs(trimmed_context))
        if trimmed_memories:
            context_parts.append(format_memories_xml(trimmed_memories))

        if context_parts:
            turns.append({'role': 'user', 'content': '\n\n'.join(context_parts)})
            # Only add wakeup if first dialogue turn is 'user' or no dialogue turns
            # (need assistant between two user turns)
            if first_dialogue_role == 'user' or first_dialogue_role is None:
                turns.append({'role': 'assistant', 'content': wakeup})
            # If first dialogue turn is 'assistant', skip wakeup to avoid consecutive assistants

        # 2. Build dialogue turns with proper roles
        dialogue_turn_contents = []  # Collect (role, content) pairs
        last_aspect_name = None

        for i, turn in enumerate(self.state.turns):
            role = get_role_for_turn(turn)
            content = turn.content

            # Scene handling: prepend at aspect changes, but ONLY for persona speakers
            if is_persona_speaker and turn.speaker_id.startswith("aspect:"):
                current_aspect = turn.speaker_id.split(":", 1)[1]
                if current_aspect != last_aspect_name:
                    # New aspect appearing - generate scene
                    prior_step = self.strategy.get_step(turn.step_id)
                    if prior_step:
                        scene = self._generate_scene(prior_step, template_context)
                        if scene:
                            content = f"{scene}\n\n{content}"
                    last_aspect_name = current_aspect

            dialogue_turn_contents.append((role, content))

        # 3. Add guidance to last user turn
        # Each step's guidance instructs that step's speaker how to format their output
        if dialogue_turn_contents:
            # Find last user turn index
            last_user_idx = None
            for i in range(len(dialogue_turn_contents) - 1, -1, -1):
                if dialogue_turn_contents[i][0] == 'user':
                    last_user_idx = i
                    break

            if last_user_idx is not None:
                role, content = dialogue_turn_contents[last_user_idx]

                # Build combined guidance (step guidance first, then user guidance)
                combined_guidance = guidance or ""

                # Append user-provided guidance if use_guidance is enabled
                if step.config.use_guidance and self.state.guidance:
                    if combined_guidance:
                        combined_guidance = f"{combined_guidance}\n\n[Guidance: {self.state.guidance}]"
                    else:
                        combined_guidance = f"[Guidance: {self.state.guidance}]"

                # Append guidance to the last user turn
                if combined_guidance:
                    content = f"{content}\n\n[~~ Output Guidance ~~]\n{combined_guidance}"

                dialogue_turn_contents[last_user_idx] = (role, content)

        # 4. Add dialogue turns to the turns list
        for role, content in dialogue_turn_contents:
            turns.append({'role': role, 'content': content})

        # 5. Handle cases where we need a final user turn
        if not dialogue_turn_contents:
            # No prior dialogue - add guidance as user turn
            final_content = []
            if is_persona_speaker:
                # First persona step with no prior turns - add scene
                scene = self._generate_scene(step, template_context)
                if scene:
                    final_content.append(scene)

            # Build combined guidance (step guidance first, then user guidance)
            combined_guidance = guidance or ""
            if step.config.use_guidance and self.state.guidance:
                if combined_guidance:
                    combined_guidance = f"{combined_guidance}\n\n[Guidance: {self.state.guidance}]"
                else:
                    combined_guidance = f"[Guidance: {self.state.guidance}]"

            if combined_guidance:
                final_content.append(f"[~~ Output Guidance ~~]\n{combined_guidance}")
            if final_content:
                turns.append({'role': 'user', 'content': '\n\n'.join(final_content)})
        elif is_persona_speaker:
            # Persona with prior turns - guidance already appended above
            pass
        else:
            # Aspect speaker - guidance was appended to last user turn above
            # But if last turn was assistant (another aspect), we need a user turn
            if dialogue_turn_contents and dialogue_turn_contents[-1][0] == 'assistant':
                # Last dialogue turn was assistant, need to add guidance as user
                # Build combined guidance (step guidance first, then user guidance)
                combined_guidance = guidance or ""
                if step.config.use_guidance and self.state.guidance:
                    if combined_guidance:
                        combined_guidance = f"{combined_guidance}\n\n[Guidance: {self.state.guidance}]"
                    else:
                        combined_guidance = f"[Guidance: {self.state.guidance}]"

                turns.append({'role': 'user', 'content': f"[~~ Output Guidance ~~]\n{combined_guidance}" if combined_guidance else '[Continue]'})

        # Final safety check - must end with user
        if turns and turns[-1]['role'] != 'user':
            # Build combined guidance (step guidance first, then user guidance)
            combined_guidance = guidance or ""
            if step.config.use_guidance and self.state.guidance:
                if combined_guidance:
                    combined_guidance = f"{combined_guidance}\n\n[Guidance: {self.state.guidance}]"
                else:
                    combined_guidance = f"[Guidance: {self.state.guidance}]"

            if combined_guidance:
                turns.append({'role': 'user', 'content': f"[~~ Output Guidance ~~]\n{combined_guidance}"})
            else:
                turns.append({'role': 'user', 'content': '[Continue]'})

        return turns, system_message

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

    def _generate_scene(self, step: DialogueStep, template_context: dict) -> str:
        """
        Generate scene dynamically based on speaker type.

        For Persona speaking: Scene shows the Aspect they're facing
        For Aspect speaking: Scene shows the Persona they're guiding

        Args:
            step: Current step definition
            template_context: Jinja2 template context with persona/aspect data

        Returns:
            Generated scene string
        """
        if step.speaker.type == SpeakerType.PERSONA:
            # Persona is speaking - show the Aspect they're facing
            aspect_name = self.strategy.dialogue.primary_aspect
            if step.speaker.aspect_name:
                # If step specifies which aspect, use that
                aspect_name = step.speaker.aspect_name

            aspect = template_context.get(aspect_name) or template_context.get('aspect')
            if not aspect:
                return ""

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

        else:
            # Aspect is speaking - show the Persona they're guiding
            persona = self.persona
            location = getattr(persona, 'location', None) or 'a reflective space'
            appearance = getattr(persona, 'appearance', None) or 'a thoughtful presence'
            emotional_state = self.state.persona_mood or 'quiet contemplation'

            return (
                f"*You observe {persona.name} in {location}. "
                f"{appearance}. "
                f"You sense {emotional_state} within them.*"
            )

    def _format_context_docs(self, context_docs: list[dict]) -> str:
        """
        Format context documents as XML for context injection.

        Args:
            context_docs: List of document dictionaries with 'content' field

        Returns:
            XML string with context wrapped in <context> tags
        """
        if not context_docs:
            return "<context>\n</context>"

        lines = ["<context>"]
        for doc in context_docs:
            content = doc.get('content', '')
            lines.append(f"  <document>{content}</document>")
        lines.append("</context>")

        return "\n".join(lines)

    def _select_model(self, step: DialogueStep) -> str:
        """
        Select appropriate model for the step.

        Uses the same priority as executor.select_model_name for consistency.

        Args:
            step: Current step with config

        Returns:
            Model name to use
        """
        # Priority: model_override > model_role > is_codex > is_thought > default

        # Explicit model override (highest priority)
        if step.config.model_override:
            return step.config.model_override

        # Model role from step definition
        if step.config.model_role:
            return self.model_set.get_model_name(step.config.model_role)

        # Legacy flags for backward compatibility (DEPRECATED)
        if step.config.is_codex and self.state.codex_model:
            return self.state.codex_model

        if step.config.is_thought and self.state.thought_model:
            return self.state.thought_model

        # Default
        return self.state.model

