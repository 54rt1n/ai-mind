# aim/dreamer/executor.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Step execution logic composing existing infrastructure."""

from dataclasses import replace
from datetime import datetime
from typing import Optional
import re
import tiktoken

from .models import PipelineState, StepDefinition, StepResult, StepConfig, Scenario
from .scenario import render_template, build_template_context
from .context import prepare_step_context
from ..agents.persona import Persona
from ..config import ChatConfig
from ..conversation.model import ConversationModel
from ..conversation.message import ConversationMessage
from ..llm.models import LanguageModelV2


class RetryableError(Exception):
    """Error that indicates the step should be retried."""
    pass


def select_model_name(state: PipelineState, step_config: StepConfig) -> str:
    """
    Select the appropriate model based on step configuration.

    Priority order:
    1. model_override if set
    2. codex_model if is_codex and codex_model is set
    3. thought_model if is_thought and thought_model is set
    4. state.model (default)

    Args:
        state: Current pipeline state with model configuration
        step_config: Step configuration with model preferences

    Returns:
        Model name to use for this step
    """
    if step_config.model_override:
        return step_config.model_override

    if step_config.is_codex and state.codex_model:
        return state.codex_model

    if step_config.is_thought and state.thought_model:
        return state.thought_model

    return state.model


def build_turns(
    state: PipelineState,
    prompt: str,
    memories: list[dict],
    prior_outputs: list[dict],
    persona: Persona,
    max_context_tokens: int,
    max_output_tokens: int,
) -> tuple[list[dict], str]:
    """
    Build the turns list for LLM call and system message.

    Creates a conversation history with:
    1. Memory context (if memories present)
    2. Wakeup message (if memories present)
    3. Prior step outputs loaded from CVM
    4. Current prompt

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

    # Add memories as context if present
    if memories:
        memory_xml = format_memories_xml(memories)
        turns.append({'role': 'user', 'content': memory_xml})
        turns.append({'role': 'assistant', 'content': persona.get_wakeup()})

    # Add prior step outputs as conversation (loaded from CVM by doc_id)
    for output in prior_outputs:
        content = output.get('content', '')
        think = output.get('think')
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
    Load context documents from CVM.

    Context accumulates through the pipeline:
    - First step with context DSL: executes DSL to build initial context
    - Subsequent steps: receive accumulated context (initial + all prior outputs)

    Args:
        state: Current pipeline state with context_doc_ids
        step_def: Step definition with optional context DSL
        cvm: ConversationModel for loading documents

    Returns:
        Tuple of (outputs, context_doc_ids, is_initial_context):
        - outputs: List of output dictionaries with 'content' and optional 'think'
        - context_doc_ids: The doc_ids used (for state update)
        - is_initial_context: True if this is new context from DSL
    """
    outputs = []

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


def extract_think_tags(response: str) -> tuple[str, Optional[str]]:
    """
    Extract <think> tags from response, returning (content, think).

    Removes all <think>...</think> blocks from the response and
    concatenates their content as the think output.

    Args:
        response: Raw response from LLM

    Returns:
        Tuple of (cleaned_content, think_content or None)
    """
    # Find all think tags
    think_pattern = r'<think>(.*?)</think>'
    think_matches = re.findall(think_pattern, response, re.DOTALL)

    if not think_matches:
        return response, None

    # Concatenate all think content
    think_content = "\n\n".join(match.strip() for match in think_matches)

    # Remove think tags from response
    cleaned_response = re.sub(think_pattern, '', response, flags=re.DOTALL)

    # Clean up extra whitespace
    cleaned_response = cleaned_response.strip()

    return cleaned_response, think_content


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
    )


async def execute_step(
    state: PipelineState,
    scenario: Scenario,
    step_def: StepDefinition,
    cvm: ConversationModel,
    persona: Persona,
    config: ChatConfig,
    encoder: tiktoken.Encoding,
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
        encoder: Token encoder for counting

    Returns:
        Tuple of (StepResult, context_doc_ids, is_initial_context):
        - StepResult with response, think, and metadata
        - context_doc_ids: Doc IDs used as context (for state update)
        - is_initial_context: True if context came from DSL (vs accumulated)

    Raises:
        RetryableError: If step should be retried
    """
    # 1. Build Jinja2 template context from Persona
    context = build_template_context(state, scenario, persona)

    # 2. Render prompt
    prompt = render_template(step_def.prompt, context)

    # 3. Append guidance if use_guidance is set and guidance exists
    if step_def.config.use_guidance and state.guidance:
        prompt = f"{prompt}\n\n[Guidance: {state.guidance}]"

    # 4. Load context (from DSL or accumulated from prior steps)
    prior_outputs, context_doc_ids, is_initial_context = load_prior_outputs(state, step_def, cvm)

    # 5. Handle memory operations (query for additional context)
    memories = []
    if step_def.memory.top_n > 0:
        # Use CVM query directly
        query_text = state.query_text or prompt[:500]
        memories_df = cvm.query(
            query_texts=[query_text],
            top_n=step_def.memory.top_n,
            query_document_type=step_def.memory.document_type,
            sort_by=step_def.memory.sort_by,
        )
        memories = memories_df.to_dict('records') if not memories_df.empty else []

    # 6. Select model
    model_name = select_model_name(state, step_def.config)
    models = LanguageModelV2.index_models(config)
    model = models.get(model_name)
    if not model:
        raise RetryableError(f"Model {model_name} not available")
    provider = model.llm_factory(config)

    # 7. Build turns for LLM (system message returned separately for config)
    max_output_tokens = min(step_def.config.max_tokens, model.max_output_tokens)
    turns, system_message = build_turns(
        state, prompt, memories, prior_outputs, persona,
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
    chunks = []
    for chunk in provider.stream_turns(turns, step_config):
        if chunk:
            chunks.append(chunk)
    response = ''.join(chunks)

    # 10. Extract <think> tags if present
    response, think = extract_think_tags(response)

    # 11. Validate response
    if not response.strip():
        raise RetryableError("Empty response from model")

    result = StepResult(
        step_id=step_def.id,
        response=response,
        think=think,
        doc_id=ConversationMessage.next_doc_id(),
        document_type=step_def.output.document_type,
        document_weight=step_def.output.weight,
        tokens_used=len(encoder.encode(response)),
        timestamp=datetime.utcnow(),
    )

    return result, context_doc_ids, is_initial_context
