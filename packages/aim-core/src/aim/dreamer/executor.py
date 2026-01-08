# aim/dreamer/executor.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Step execution logic composing existing infrastructure."""

from dataclasses import replace
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING
import logging
import re

if TYPE_CHECKING:
    from aim.llm.model_set import ModelSet

logger = logging.getLogger(__name__)

from .models import PipelineState, StepDefinition, StepResult, StepConfig, Scenario
from .scenario import render_template, build_template_context
from .context import prepare_step_context
from ..agents.persona import Persona
from ..config import ChatConfig
from ..conversation.model import ConversationModel
from ..conversation.message import ConversationMessage
from ..llm.models import LanguageModelV2
from ..utils.tokens import count_tokens
from ..utils.think import extract_think_tags


class RetryableError(Exception):
    """Error that indicates the step should be retried."""
    pass


def select_model_name(
    state: PipelineState,
    step_config: StepConfig,
    model_set: "ModelSet"
) -> str:
    """
    Select the appropriate model based on step configuration and persona's ModelSet.

    Priority order:
    1. step_config.model_override (explicit override string)
    2. step_config.model_role (role name resolved via ModelSet)
    3. Legacy is_codex/is_thought flags (DEPRECATED, for backward compatibility)
    4. state.model (pipeline default)

    Args:
        state: Current pipeline state with model configuration
        step_config: Step configuration with model preferences
        model_set: ModelSet for resolving roles to model names

    Returns:
        Model name to use for this step
    """
    # Explicit model override (highest priority)
    if step_config.model_override:
        return step_config.model_override

    # Model role from step definition
    if step_config.model_role:
        return model_set.get_model_name(step_config.model_role)

    # Legacy flags for backward compatibility (DEPRECATED)
    if step_config.is_codex and state.codex_model:
        return state.codex_model

    if step_config.is_thought and state.thought_model:
        return state.thought_model

    # Pipeline default
    return state.model


def build_turns(
    state: PipelineState,
    prompt: str,
    memories: list[dict],
    prior_outputs: list[dict],
    persona: Persona,
    max_context_tokens: int,
    max_output_tokens: int,
    include_thought_stream: bool = True,
) -> tuple[list[dict], str]:
    """
    Build the turns list for LLM call and system message.

    Creates a conversation history with:
    1. Memory context (if memories present)
    2. Wakeup message (if memories present)
    3. Thought stream from prior outputs (evicted first if over budget)
    4. Prior step outputs loaded from CVM
    5. Current prompt

    The system message is returned separately to be set via ChatConfig,
    as the LLM provider handles system messages through config.system_message.

    Args:
        state: Current pipeline state
        prompt: Rendered prompt for current step
        memories: List of memory records from CVM query
        prior_outputs: List of prior step outputs loaded from CVM
        persona: Persona object for system prompt
        max_context_tokens: Model's max context window
        max_output_tokens: Max tokens to generate (min of requested and model limit)
        include_thought_stream: Whether to include thought stream in context

    Returns:
        Tuple of (turns list, system_message string)
    """
    turns = []

    # System message from persona (returned separately for config)
    system_message = persona.system_prompt(
        mood=state.persona_mood,
        location=None,  # Location is in the step prompts
        user_id=state.user_id,
        system_message=None,
    )

    # Calculate available tokens for content (with safety margin)
    safety_margin = 1024
    available_tokens = max_context_tokens - max_output_tokens - safety_margin

    # Account for fixed content (system, prompt, wakeup)
    wakeup = persona.get_wakeup()
    fixed_tokens = (
        count_tokens(system_message) +
        count_tokens(prompt) +
        count_tokens(wakeup)
    )

    # Calculate tokens for prior outputs content
    prior_output_tokens = sum(count_tokens(o.get('content', '')) for o in prior_outputs)

    # Calculate tokens for thought stream
    thought_stream = format_thought_stream(prior_outputs) if include_thought_stream else ""
    thought_stream_tokens = count_tokens(thought_stream)

    # Budget remaining after fixed content
    remaining_budget = available_tokens - fixed_tokens

    # Eviction priority: thoughts first, then oldest prior outputs, then memories
    # 1. Check if we need to evict thought stream
    if prior_output_tokens + thought_stream_tokens > remaining_budget:
        # Evict thought stream first
        logger.info(f"Evicting thought stream ({thought_stream_tokens} tokens) to fit budget")
        thought_stream = ""
        thought_stream_tokens = 0

    # 2. Evict oldest prior outputs if still over budget
    trimmed_outputs = list(prior_outputs)
    evicted_count = 0
    while trimmed_outputs and prior_output_tokens > remaining_budget:
        # Remove oldest (first) output
        removed = trimmed_outputs.pop(0)
        removed_tokens = count_tokens(removed.get('content', ''))
        prior_output_tokens -= removed_tokens
        evicted_count += 1
        logger.info(f"Evicting prior output {evicted_count} ({removed_tokens} tokens)")

    if evicted_count > 0:
        logger.info(f"Evicted {evicted_count} prior outputs, {len(trimmed_outputs)} remaining")

    # 3. Budget remaining for memories
    memory_budget = remaining_budget - prior_output_tokens - thought_stream_tokens

    # Trim memories to fit budget (keep newest, trim oldest first)
    trimmed_memories = []
    memory_tokens = 0
    for mem in reversed(memories):
        content = mem.get('content', '')
        mem_tokens = count_tokens(content)
        if memory_tokens + mem_tokens <= memory_budget:
            trimmed_memories.insert(0, mem)  # Prepend to maintain order
            memory_tokens += mem_tokens
        else:
            break  # Stop adding memories when budget exceeded

    if len(trimmed_memories) < len(memories):
        logger.info(f"Trimmed memories: {len(memories)} -> {len(trimmed_memories)} ({memory_tokens} tokens)")

    # Build turns list
    # Add memories as context if present
    if trimmed_memories:
        memory_xml = format_memories_xml(trimmed_memories)
        turns.append({'role': 'user', 'content': memory_xml})
        turns.append({'role': 'assistant', 'content': wakeup})

    # Add thought stream if present (after eviction check)
    if thought_stream:
        turns.append({'role': 'user', 'content': thought_stream})

    # Add prior step outputs as conversation (loaded from CVM by doc_id)
    for output in trimmed_outputs:
        content = output.get('content', '')
        if content:
            turns.append({'role': 'assistant', 'content': content})

    # Add current prompt
    turns.append({'role': 'user', 'content': prompt})

    return turns, system_message


def load_prior_outputs(
    state: PipelineState,
    step_def: StepDefinition,
    cvm: ConversationModel,
) -> tuple[list[dict], list[str], bool]:
    """
    Load context documents from CVM or use pre-provided documents.

    Context sources (in priority order):
    1. Pre-provided context_documents in state (from refiner/external source)
    2. Accumulated context_doc_ids from prior steps
    3. Context DSL execution for first step

    Args:
        state: Current pipeline state with context_doc_ids and optional context_documents
        step_def: Step definition with optional context DSL
        cvm: ConversationModel for loading documents

    Returns:
        Tuple of (outputs, context_doc_ids, is_initial_context):
        - outputs: List of output dictionaries with 'content' and optional 'think'
        - context_doc_ids: The doc_ids used (for state update)
        - is_initial_context: True if this is new context from DSL or pre-provided docs
    """
    outputs = []

    # Check for pre-provided context documents (from refiner)
    # Only use for first step (when context_doc_ids is empty)
    if state.context_documents and not state.context_doc_ids:
        logger.info(f"Using {len(state.context_documents)} pre-provided context documents")
        for doc in state.context_documents:
            outputs.append({
                'content': doc.get('content', ''),
                'think': doc.get('think'),
                'source': 'pre-provided',
            })
        # Return empty context_doc_ids since these aren't from CVM
        # The documents are already in outputs
        return outputs, [], True

    # Get context - either from DSL or accumulated from prior steps
    context_doc_ids, is_initial_context = prepare_step_context(step_def, state, cvm)

    for doc_id in context_doc_ids:
        doc = cvm.get_by_doc_id(doc_id)
        if doc is not None:
            outputs.append({
                'content': doc.get('content', ''),
                'think': doc.get('think'),
                'source': 'context',
            })

    return outputs, context_doc_ids, is_initial_context


def format_memories_xml(memories: list[dict]) -> str:
    """
    Format memory records as XML for context injection.

    Args:
        memories: List of memory dictionaries with 'content' field

    Returns:
        XML string with memories wrapped in <memories> tags
    """
    if not memories:
        return "<memories>\n</memories>"

    lines = ["<memories>"]

    for memory in memories:
        content = memory.get('content', '')
        lines.append(f"  <memory>{content}</memory>")

    lines.append("</memories>")

    return "\n".join(lines)


def format_thought_stream(prior_outputs: list[dict]) -> str:
    """
    Extract and format think content from prior outputs.

    Args:
        prior_outputs: List of output dictionaries with optional 'think' field

    Returns:
        XML string with thought stream, or empty string if no think content
    """
    thoughts = [
        output.get('think') for output in prior_outputs
        if output.get('think')
    ]

    if not thoughts:
        return ""

    lines = ["<thought_stream>"]
    for i, thought in enumerate(thoughts):
        lines.append(f"\t<prior_thought turn=\"{i+1}\">{thought}</prior_thought>")
    lines.append("</thought_stream>")

    return "\n".join(lines)

def create_message(
    state: PipelineState,
    step_def: StepDefinition,
    result: StepResult,
) -> ConversationMessage:
    """
    Create a ConversationMessage from step result.

    Args:
        state: Current pipeline state
        step_def: Step definition
        result: Step execution result

    Returns:
        ConversationMessage ready to insert into CVM
    """
    return ConversationMessage.create(
        doc_id=result.doc_id,
        conversation_id=state.conversation_id,
        user_id=state.user_id,
        persona_id=state.persona_id,
        sequence_no=state.step_counter * 2,  # Even numbers for assistant
        branch=state.branch,
        role='assistant',
        content=result.response,
        think=result.think,
        document_type=result.document_type,
        weight=result.document_weight,
        speaker_id=state.persona_id,
        inference_model=state.model,
        scenario_name=state.scenario_name,
        step_name=step_def.id,
    )


async def execute_step(
    state: PipelineState,
    scenario: Scenario,
    step_def: StepDefinition,
    cvm: ConversationModel,
    persona: Persona,
    config: ChatConfig,
    model_set: "ModelSet",
) -> tuple[StepResult, list[str], bool]:
    """
    Execute a single pipeline step.

    Full execution flow:
    1. Build template context from persona
    2. Render prompt with Jinja2
    3. Append guidance if use_guidance is set
    4. Load context (from DSL or accumulated)
    5. Handle memory operations (query)
    6. Build turns for LLM
    7. Select model
    8. Build step config
    9. Generate response (streaming)
    10. Extract think tags
    11. Validate and return StepResult

    Args:
        state: Current pipeline state
        scenario: Scenario definition
        step_def: Step to execute
        cvm: ConversationModel for memory queries
        persona: Persona for context
        config: ChatConfig for LLM access
        model_set: ModelSet for persona-aware model selection

    Returns:
        Tuple of (StepResult, context_doc_ids, is_initial_context):
        - StepResult with response, think, and metadata
        - context_doc_ids: Doc IDs used as context (for state update)
        - is_initial_context: True if context came from DSL (vs accumulated)

    Raises:
        RetryableError: If step should be retried
    """
    # Log step start with context
    step_flags = []
    if step_def.config.is_codex:
        step_flags.append("codex")
    if step_def.config.is_thought:
        step_flags.append("thought")
    if step_def.config.use_guidance:
        step_flags.append("guidance")
    flags_str = f" [{', '.join(step_flags)}]" if step_flags else ""

    logger.info(
        f"Step '{step_def.id}' starting{flags_str} | "
        f"pipeline={state.pipeline_id[:8]}... scenario={state.scenario_name} "
        f"step_num={state.step_counter} output_type={step_def.output.document_type}"
    )

    # 1. Build Jinja2 template context from Persona
    context = build_template_context(state, scenario, persona)

    # 2. Render prompt
    prompt = render_template(step_def.prompt, context)

    # 3. Append guidance if use_guidance is set and guidance exists
    if step_def.config.use_guidance and state.guidance:
        prompt = f"{prompt}\n\n[Guidance: {state.guidance}]"

    # 4. Load context (from DSL or accumulated from prior steps)
    # Memory retrieval is now handled by search_memories action in context DSL
    prior_outputs, context_doc_ids, is_initial_context = load_prior_outputs(state, step_def, cvm)

    # 5. Select model
    model_name = select_model_name(state, step_def.config, model_set)
    models = LanguageModelV2.index_models(config)
    model = models.get(model_name)
    if not model:
        raise RetryableError(f"Model {model_name} not available")
    provider = model.llm_factory(config)

    logger.info(
        f"Step '{step_def.id}' using model={model_name} | "
        f"prior_outputs={len(prior_outputs)} "
        f"max_tokens={step_def.config.max_tokens}"
    )

    # 6. Build turns for LLM (system message returned separately for config)
    # Note: memories param is empty - memory retrieval now flows through context DSL
    max_output_tokens = min(step_def.config.max_tokens, model.max_output_tokens)
    turns, system_message = build_turns(
        state, prompt, [], prior_outputs, persona,
        max_context_tokens=model.max_tokens,
        max_output_tokens=max_output_tokens,
    )

    # 8. Build config for this step (ChatConfig is a dataclass, use replace())
    step_config = replace(
        config,
        max_tokens=max_output_tokens,
        temperature=step_def.config.temperature or config.temperature,
        system_message=system_message,
    )

    # 9. Generate response (streaming)
    # Update activity timestamp during streaming to prevent cascading triggers
    from ..utils.redis_cache import RedisCache
    cache = RedisCache(config)
    cache.update_api_activity()

    chunks = []
    chunk_count = 0
    for chunk in provider.stream_turns(turns, step_config):
        if chunk:
            chunks.append(chunk)
            chunk_count += 1
            # Update activity every 50 chunks to keep timestamp fresh during long streams
            if chunk_count % 50 == 0:
                cache.update_api_activity()

    # Final activity update after streaming completes
    cache.update_api_activity()
    response = ''.join(chunks)

    # 10. Extract <think> tags if present
    response, think = extract_think_tags(response)

    # 11. Validate response
    if not response.strip():
        raise RetryableError("Empty response from model")

    tokens_used = count_tokens(response)
    result = StepResult(
        step_id=step_def.id,
        response=response,
        think=think,
        doc_id=ConversationMessage.next_doc_id(),
        document_type=step_def.output.document_type,
        document_weight=step_def.output.weight,
        tokens_used=tokens_used,
        timestamp=datetime.now(timezone.utc),
    )

    logger.info(
        f"Step '{step_def.id}' complete | "
        f"tokens={tokens_used} has_think={think is not None} "
        f"response_len={len(response)}"
    )

    return result, context_doc_ids, is_initial_context
