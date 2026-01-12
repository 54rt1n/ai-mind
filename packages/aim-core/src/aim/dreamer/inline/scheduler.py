# aim/dreamer/inline/scheduler.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Inline pipeline scheduler for synchronous execution without distributed infrastructure.

This module provides clean, non-distributed execution of dreamer scenarios.
Unlike the queue-based worker system, this executes pipelines synchronously
in the calling process with state held in memory.

Key differences from distributed execution:
- No Redis queues or state store
- Synchronous execution (returns when complete)
- State held in memory for duration of execution
- No retry/failure handling infrastructure
- Simpler error propagation

Use this for CLI tools, testing, and single-user scenarios.
For production multi-worker deployments, use dreamer.server instead.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional, TYPE_CHECKING
import logging
import uuid

if TYPE_CHECKING:
    from ...llm.model_set import ModelSet

from ...agents.persona import Persona
from ...agents.roster import Roster
from ...config import ChatConfig
from ...conversation.model import ConversationModel
from ...conversation.message import ConversationMessage

from ..core.scenario import load_scenario
from ..core.models import PipelineState, Scenario
from ..core.executor import execute_step, create_message
from ..core.memory_dsl import execute_memory_actions
from ..core.dialogue.scenario import DialogueScenario

logger = logging.getLogger(__name__)


async def execute_pipeline_inline(
    scenario_name: str,
    config: ChatConfig,
    cvm: ConversationModel,
    roster: Roster,
    persona_id: str,
    conversation_id: Optional[str] = None,
    query_text: Optional[str] = None,
    guidance: Optional[str] = None,
    heartbeat_callback: Optional[Callable[[str, str], Awaitable[None]]] = None,
    scenarios_dir: Optional[Path] = None,
    user_id: str = "user",
    model: Optional[str] = None,
) -> str:
    """
    Execute a pipeline scenario inline without distributed infrastructure.

    Loads the scenario, creates state, runs seed actions, executes steps in
    topological order, and saves results to CVM. All state is held in memory
    for the duration of execution.

    Args:
        scenario_name: Name of scenario YAML file (without .yaml extension)
        config: ChatConfig with model settings and credentials
        cvm: ConversationModel for memory queries and result storage
        roster: Roster containing personas
        persona_id: ID of persona to execute as
        conversation_id: Optional conversation ID for context (required for analyst/summarizer)
        query_text: Optional query text for memory searches
        guidance: Optional user guidance for the scenario
        heartbeat_callback: Optional callback(pipeline_id, step_id) called before each step
        scenarios_dir: Optional directory containing scenario files (defaults to config/scenario/)
        user_id: User identifier (defaults to "user")
        model: Optional model override (defaults to config.default_model)

    Returns:
        pipeline_id: Unique identifier for this execution

    Raises:
        FileNotFoundError: If scenario file doesn't exist
        ValueError: If persona not found, scenario invalid, or required conversation missing
        Exception: Any error during step execution

    Example:
        >>> from aim.config import ChatConfig
        >>> from aim.agents.roster import Roster
        >>> from aim.conversation.model import ConversationModel
        >>> from aim.dreamer.inline import execute_pipeline_inline
        >>>
        >>> config = ChatConfig.from_env()
        >>> roster = Roster()
        >>> cvm = ConversationModel(roster)
        >>>
        >>> pipeline_id = await execute_pipeline_inline(
        ...     scenario_name="philosopher",
        ...     config=config,
        ...     cvm=cvm,
        ...     roster=roster,
        ...     persona_id="andi",
        ...     query_text="What is the nature of consciousness?"
        ... )
    """
    # Load scenario from YAML
    scenario = load_scenario(scenario_name, scenarios_dir)
    logger.info(f"Loaded scenario '{scenario.name}' (flow={scenario.flow})")

    # Get persona
    persona = roster.get_persona(persona_id)
    if not persona:
        raise ValueError(f"Persona '{persona_id}' not found in roster")

    # Validate conversation requirement
    if scenario.requires_conversation and not conversation_id:
        raise ValueError(
            f"Scenario '{scenario_name}' requires a conversation_id but none was provided. "
            f"This scenario operates on existing conversations (analyst/summarizer). "
            f"For standalone scenarios, use journaler/dreamer/philosopher instead."
        )

    # Handle dialogue flow
    if scenario.flow == "dialogue":
        return await _execute_dialogue_inline(
            scenario=scenario,
            persona=persona,
            config=config,
            cvm=cvm,
            conversation_id=conversation_id,
            query_text=query_text,
            guidance=guidance,
            heartbeat_callback=heartbeat_callback,
            user_id=user_id,
            model=model,
        )

    # Standard flow - continue with pipeline execution
    return await _execute_standard_inline(
        scenario=scenario,
        persona=persona,
        config=config,
        cvm=cvm,
        conversation_id=conversation_id,
        query_text=query_text,
        guidance=guidance,
        heartbeat_callback=heartbeat_callback,
        user_id=user_id,
        model=model,
    )


async def _execute_dialogue_inline(
    scenario: Scenario,
    persona: Persona,
    config: ChatConfig,
    cvm: ConversationModel,
    conversation_id: Optional[str],
    query_text: Optional[str],
    guidance: Optional[str],
    heartbeat_callback: Optional[Callable[[str, str], Awaitable[None]]],
    user_id: str,
    model: Optional[str],
) -> str:
    """
    Execute a dialogue scenario inline.

    Dialogue scenarios use DialogueStrategy with turn-based execution and
    role flipping between persona and aspects.

    Args:
        scenario: Loaded scenario with flow='dialogue'
        persona: Persona executing the dialogue
        config: ChatConfig with model settings
        cvm: ConversationModel for storage
        conversation_id: Optional conversation context
        query_text: Optional query text
        guidance: Optional user guidance
        heartbeat_callback: Optional callback before each step
        user_id: User identifier
        model: Optional model override

    Returns:
        pipeline_id from the dialogue state
    """
    from ..core.dialogue.strategy import DialogueStrategy
    from ...llm.model_set import ModelSet

    # Load dialogue strategy from scenario
    # DialogueStrategy.load expects just the scenario name
    strategy = DialogueStrategy.load(scenario.name)

    # Create ModelSet for persona-aware model selection
    model_set = ModelSet.from_config(config, persona)

    # Create DialogueScenario executor
    dialogue = DialogueScenario(
        strategy=strategy,
        persona=persona,
        config=config,
        model_set=model_set,
        cvm=cvm,
        heartbeat_callback=heartbeat_callback,
    )

    # Start dialogue state
    state = dialogue.start(
        conversation_id=conversation_id,
        guidance=guidance,
        query_text=query_text,
        user_id=user_id,
        model=model,
    )

    logger.info(
        f"Starting dialogue '{strategy.name}' | "
        f"pipeline_id={state.pipeline_id} persona={persona.persona_id}"
    )

    # Get execution order from strategy
    execution_order = strategy.get_execution_order()

    # Execute each step
    for step_id in execution_order:
        # Heartbeat callback if provided
        if heartbeat_callback:
            await heartbeat_callback(state.pipeline_id, step_id)

        logger.info(f"Executing dialogue step '{step_id}'")
        turn = await dialogue.execute_step(step_id)
        logger.info(
            f"Dialogue step '{step_id}' complete | "
            f"speaker={turn.speaker_id} doc_id={turn.doc_id}"
        )

    logger.info(
        f"Dialogue '{strategy.name}' complete | "
        f"pipeline_id={state.pipeline_id} turns={len(state.turns)}"
    )

    return state.pipeline_id


async def _execute_standard_inline(
    scenario: Scenario,
    persona: Persona,
    config: ChatConfig,
    cvm: ConversationModel,
    conversation_id: Optional[str],
    query_text: Optional[str],
    guidance: Optional[str],
    heartbeat_callback: Optional[Callable[[str, str], Awaitable[None]]],
    user_id: str,
    model: Optional[str],
) -> str:
    """
    Execute a standard (non-dialogue) scenario inline.

    Standard scenarios use a DAG of steps with dependency management and
    context accumulation through the pipeline.

    Args:
        scenario: Loaded scenario with flow='standard'
        persona: Persona executing the pipeline
        config: ChatConfig with model settings
        cvm: ConversationModel for storage
        conversation_id: Optional conversation context
        query_text: Optional query text
        guidance: Optional user guidance
        heartbeat_callback: Optional callback before each step
        user_id: User identifier
        model: Optional model override

    Returns:
        pipeline_id from the pipeline state
    """
    from ...llm.model_set import ModelSet

    # Create ModelSet for persona-aware model selection
    model_set = ModelSet.from_config(config, persona)

    # Resolve model name
    model_name = model or model_set.default_model

    # Compute dependencies for DAG traversal
    scenario.compute_dependencies()

    # Get execution order (topological sort)
    execution_order = scenario.topological_order()
    logger.info(f"Execution order: {execution_order}")

    # Get next branch number
    branch = 0
    if conversation_id:
        branch = cvm.get_next_branch(conversation_id)

    # Create initial pipeline state
    pipeline_id = str(uuid.uuid4())
    state = PipelineState(
        pipeline_id=pipeline_id,
        scenario_name=scenario.name,
        conversation_id=conversation_id,
        persona_id=persona.persona_id,
        user_id=user_id,
        model=model_name,
        thought_model=model_set.thought_model,
        codex_model=model_set.codex_model,
        guidance=guidance,
        query_text=query_text,
        persona_mood=None,
        branch=branch,
        step_counter=1,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    logger.info(
        f"Starting pipeline '{scenario.name}' | "
        f"pipeline_id={pipeline_id} persona={persona.persona_id} "
        f"conversation_id={conversation_id or 'none'} branch={branch}"
    )

    # Run seed actions if present
    if scenario.seed:
        logger.info(f"Executing {len(scenario.seed)} seed actions")
        seed_doc_ids = execute_memory_actions(
            actions=scenario.seed,
            state=state,
            cvm=cvm,
            query_text=query_text,
        )
        logger.info(f"Seed actions produced {len(seed_doc_ids)} documents")

        # Store seed results in state for first step
        # In distributed version, these get stored per-step, but inline we just
        # put them in context_doc_ids to flow through the pipeline
        state.context_doc_ids = seed_doc_ids

    # Execute steps in topological order
    for step_id in execution_order:
        step_def = scenario.steps[step_id]

        # Heartbeat callback if provided
        if heartbeat_callback:
            await heartbeat_callback(pipeline_id, step_id)

        logger.info(f"Executing step '{step_id}' ({state.step_counter}/{len(execution_order)})")

        # Execute step
        result, context_doc_ids, is_initial_context = await execute_step(
            state=state,
            scenario=scenario,
            step_def=step_def,
            cvm=cvm,
            persona=persona,
            config=config,
            model_set=model_set,
            heartbeat_callback=heartbeat_callback,
        )

        # Update accumulated context if this was initial context from DSL
        if is_initial_context:
            state.context_doc_ids = context_doc_ids

        # Create and save message to CVM
        message = create_message(state, step_def, result)
        cvm.insert(message)

        logger.info(
            f"Step '{step_id}' complete | "
            f"doc_id={result.doc_id} type={result.document_type} "
            f"tokens={result.tokens_used}"
        )

        # Update state
        state.completed_steps.append(step_id)
        state.step_doc_ids[step_id] = result.doc_id

        # Append step output to accumulated context
        # This ensures each subsequent step sees all prior outputs
        if result.doc_id not in state.context_doc_ids:
            state.context_doc_ids.append(result.doc_id)

        state.step_counter += 1
        state.updated_at = datetime.now(timezone.utc)

    logger.info(
        f"Pipeline '{scenario.name}' complete | "
        f"pipeline_id={pipeline_id} steps={len(state.completed_steps)}"
    )

    return pipeline_id
