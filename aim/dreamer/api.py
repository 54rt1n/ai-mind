# aim/dreamer/api.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Public API for pipeline management."""

from typing import Optional
from datetime import datetime, timezone
import uuid
import pandas as pd

from .models import PipelineState, Scenario, StepStatus
from .scenario import load_scenario
from .scheduler import Scheduler
from .state import StateStore
from ..config import ChatConfig
from ..conversation.model import ConversationModel
from ..agents.roster import Roster
from ..llm.models import LanguageModelV2


# Mapping from signature document types to scenario names
# Each scenario has characteristic output document types that identify it
# Only final/unique outputs are used - partial runs require --scenario flag
SCENARIO_SIGNATURES = {
    "analysis": "analyst",      # analyst produces analysis
    "summary": "summarizer",    # summarizer produces summary
    "journal": "journaler",     # journaler produces journal
    "pondering": "philosopher", # philosopher produces pondering
    "daydream": "daydream",     # daydream produces daydream
}


def infer_scenario_from_documents(doc_types: set[str]) -> Optional[str]:
    """
    Infer the scenario type from a set of document types found in a conversation.

    Uses signature document types that are unique to each scenario.

    Args:
        doc_types: Set of document_type values from conversation history

    Returns:
        Scenario name if inferred, None if ambiguous or no match
    """
    for signature_type, scenario_name in SCENARIO_SIGNATURES.items():
        if signature_type in doc_types:
            return scenario_name
    return None


class PipelineStatus:
    """Status information for a pipeline."""

    def __init__(
        self,
        pipeline_id: str,
        scenario_name: str,
        status: str,
        current_step: Optional[str],
        completed_steps: list[str],
        failed_steps: list[str],
        step_errors: dict[str, str],
        progress_percent: float,
        created_at: datetime,
        updated_at: datetime,
    ):
        self.pipeline_id = pipeline_id
        self.scenario_name = scenario_name
        self.status = status
        self.current_step = current_step
        self.completed_steps = completed_steps
        self.failed_steps = failed_steps
        self.step_errors = step_errors
        self.progress_percent = progress_percent
        self.created_at = created_at
        self.updated_at = updated_at


def generate_pipeline_id() -> str:
    """Generate a unique pipeline ID using UUID4."""
    return str(uuid.uuid4())


async def run_seed_actions(
    scenario: Scenario,
    state: PipelineState,
    cvm: ConversationModel,
) -> PipelineState:
    """
    Execute seed actions to load initial context as doc_id references.

    Handles two action types:
    - load_conversation: Load conversation history doc_ids
    - query_memories: Query memories and store doc_ids

    Each seed action specifies `accumulate_to` - the step that will
    receive these doc_ids as input context.

    Args:
        scenario: Scenario definition with seed actions
        state: Current pipeline state
        cvm: ConversationModel for querying

    Returns:
        Updated PipelineState with doc_id references in seed_doc_ids
    """
    for seed_action in scenario.seed:
        target_step = seed_action.accumulate_to
        doc_ids = []

        if seed_action.action == "load_conversation":
            # Load conversation history
            params = seed_action.params
            document_type = params.get('document_type', None)
            exclude = params.get('exclude', False)

            # Get conversation history
            if exclude:
                # Exclude these document types
                history_df = cvm.get_conversation_history(
                    state.conversation_id,
                    filter_document_type=document_type
                )
            else:
                history_df = cvm.get_conversation_history(
                    state.conversation_id,
                    query_document_type=document_type
                )

            # Extract doc_ids as references
            if not history_df.empty and 'doc_id' in history_df.columns:
                doc_ids = history_df['doc_id'].tolist()

        elif seed_action.action == "query_memories":
            # Query memories
            from aim.constants import CHUNK_LEVEL_768

            params = seed_action.params
            document_type = params.get('document_type', None)
            top_n = params.get('top_n', 10)
            sort_by = params.get('sort_by', 'relevance')
            temporal_decay = params.get('temporal_decay', 0.99)
            turn_decay = params.get('turn_decay', 0.7)

            # Use query_text or a default query
            query_text = state.query_text or "Recent conversation context"

            # Query memories
            memories_df = cvm.query(
                query_texts=[query_text],
                top_n=top_n,
                query_document_type=document_type,
                query_conversation_id=state.conversation_id,
                sort_by=sort_by,
                temporal_decay=temporal_decay,
                turn_decay=turn_decay,
                chunk_level=CHUNK_LEVEL_768,
            )

            # Extract doc_ids as references
            if not memories_df.empty and 'doc_id' in memories_df.columns:
                doc_ids = memories_df['doc_id'].tolist()

        # Store doc_ids for the target step
        if doc_ids:
            if target_step not in state.seed_doc_ids:
                state.seed_doc_ids[target_step] = []
            state.seed_doc_ids[target_step].extend(doc_ids)

    return state


async def start_pipeline(
    scenario_name: str,
    config: ChatConfig,
    model_name: str,
    state_store: StateStore,
    scheduler: Scheduler,
    conversation_id: Optional[str] = None,
    persona_id: Optional[str] = None,
    user_id: Optional[str] = None,
    query_text: Optional[str] = None,
    guidance: Optional[str] = None,
    mood: Optional[str] = None,
    context_documents: Optional[list[dict]] = None,
) -> str:
    """
    Start a new pipeline execution.

    Args:
        scenario_name: Name of the scenario YAML to load
        config: ChatConfig with provider keys and settings
        model_name: Name of the model to use (from models.yaml)
        state_store: StateStore instance for Redis operations
        scheduler: Scheduler instance for queue operations
        conversation_id: ID of the conversation to analyze (required for analyst/summarizer)
        persona_id: Persona ID to use (falls back to conversation or config)
        user_id: User ID (falls back to config)
        query_text: Optional query for journaler/philosopher scenarios
        guidance: Optional guidance text
        mood: Optional persona mood

    Returns:
        pipeline_id for tracking

    Raises:
        FileNotFoundError: If scenario not found
        ValueError: If model not found or conversation required but not found
    """
    # 1. Load scenario
    scenario = load_scenario(scenario_name)
    scenario.compute_dependencies()

    # 2. Load existing infrastructure
    cvm = ConversationModel.from_config(config)
    roster = Roster.from_config(config)

    # 3. Resolve persona_id - explicit > conversation > config
    resolved_persona_id = persona_id
    if not resolved_persona_id:
        if scenario.requires_conversation:
            if not conversation_id:
                raise ValueError(f"Scenario '{scenario_name}' requires a conversation_id")
            conv_df = cvm.index.search(
                query_conversation_id=conversation_id,
                query_document_type='conversation',
                query_limit=1,
            )
            if conv_df.empty:
                raise ValueError(f"Conversation {conversation_id} not found")
            resolved_persona_id = conv_df.iloc[0]['persona_id']
        else:
            resolved_persona_id = config.persona_id
            if not resolved_persona_id:
                raise ValueError(f"Scenario '{scenario_name}' requires persona_id")

    persona = roster.personas[resolved_persona_id]

    # 4. Resolve user_id - explicit > persona_id > config
    resolved_user_id = user_id or resolved_persona_id or config.user_id

    # 5. Validate model exists
    models = LanguageModelV2.index_models(config)
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found in available models")

    # 6. Initialize state
    pipeline_id = generate_pipeline_id()
    # Get branch from conversation if available, else start at 0
    branch = cvm.get_next_branch(conversation_id) if conversation_id else 0
    state = PipelineState(
        pipeline_id=pipeline_id,
        scenario_name=scenario_name,
        conversation_id=conversation_id,
        persona_id=resolved_persona_id,
        user_id=resolved_user_id,
        model=model_name,
        thought_model=config.thought_model,
        codex_model=config.codex_model,
        guidance=guidance or config.guidance,
        query_text=query_text,
        persona_mood=mood or config.persona_mood,
        branch=branch,
        context_documents=context_documents,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    # 6. Run seed actions (load initial context)
    state = await run_seed_actions(scenario, state, cvm)

    # 7. Persist state to Redis
    await state_store.save_state(state)

    # 8. Initialize DAG (all steps pending)
    await state_store.init_dag(pipeline_id, scenario)

    # 9. Enqueue root steps
    for step_id in scenario.get_root_steps():
        await scheduler.enqueue_step(pipeline_id, step_id)

    return pipeline_id


async def get_status(
    pipeline_id: str,
    state_store: StateStore,
    scenario: Scenario,
) -> Optional[PipelineStatus]:
    """
    Get the current status of a pipeline.

    Args:
        pipeline_id: Pipeline identifier
        state_store: StateStore instance for Redis operations
        scenario: Scenario definition for step information

    Returns:
        PipelineStatus if pipeline exists, None otherwise
    """
    # Load state from Redis
    state = await state_store.load_state(pipeline_id)

    if state is None:
        return None

    # Get step statuses from DAG
    completed_steps = []
    failed_steps = []
    running_steps = []
    pending_steps = []

    for step_id in scenario.steps.keys():
        status = await state_store.get_step_status(pipeline_id, step_id)

        if status == StepStatus.COMPLETE:
            completed_steps.append(step_id)
        elif status == StepStatus.FAILED:
            failed_steps.append(step_id)
        elif status == StepStatus.RUNNING:
            running_steps.append(step_id)
        elif status == StepStatus.PENDING:
            pending_steps.append(step_id)

    # Calculate progress
    total_steps = len(scenario.steps)
    progress_percent = (len(completed_steps) / total_steps * 100) if total_steps > 0 else 0.0

    # Determine overall status
    if failed_steps:
        overall_status = "failed"
    elif len(completed_steps) == total_steps:
        overall_status = "complete"
    elif running_steps or (completed_steps and pending_steps):
        overall_status = "running"
    else:
        overall_status = "pending"

    # Current step is the first running step, or None
    current_step = running_steps[0] if running_steps else None

    # Get step errors
    step_errors = await state_store.get_step_errors(pipeline_id)

    return PipelineStatus(
        pipeline_id=pipeline_id,
        scenario_name=state.scenario_name,
        status=overall_status,
        current_step=current_step,
        completed_steps=completed_steps,
        failed_steps=failed_steps,
        step_errors=step_errors,
        progress_percent=progress_percent,
        created_at=state.created_at,
        updated_at=state.updated_at,
    )


async def cancel_pipeline(
    pipeline_id: str,
    state_store: StateStore,
) -> bool:
    """
    Cancel a running pipeline.

    Completed steps are preserved. Sets all pending/running steps to failed.

    Args:
        pipeline_id: Pipeline identifier
        state_store: StateStore instance for Redis operations

    Returns:
        True if pipeline was cancelled, False if not found
    """
    # Load state from Redis
    state = await state_store.load_state(pipeline_id)

    if state is None:
        return False

    # Load scenario to get all steps
    scenario = load_scenario(state.scenario_name)

    # Mark all pending/running steps as failed
    for step_id in scenario.steps.keys():
        status = await state_store.get_step_status(pipeline_id, step_id)

        if status in (StepStatus.PENDING, StepStatus.RUNNING):
            await state_store.set_step_status(pipeline_id, step_id, StepStatus.FAILED)

    # Update state timestamp
    state.updated_at = datetime.now(timezone.utc)
    await state_store.save_state(state)

    return True


async def delete_pipeline(
    pipeline_id: str,
    state_store: StateStore,
) -> bool:
    """
    Delete a pipeline and all its state from Redis.

    Only allows deletion of completed or failed pipelines.

    Args:
        pipeline_id: Pipeline identifier
        state_store: StateStore instance for Redis operations

    Returns:
        True if pipeline was deleted, False if not found
    """
    # Load state to verify it exists
    state = await state_store.load_state(pipeline_id)

    if state is None:
        return False

    # Delete all state (state, DAG, errors)
    await state_store.delete_state(pipeline_id)

    return True


async def resume_pipeline(
    pipeline_id: str,
    state_store: StateStore,
    scheduler: Scheduler,
) -> bool:
    """
    Resume a failed or cancelled pipeline from where it stopped.

    Re-enqueues failed steps whose dependencies are satisfied.

    Args:
        pipeline_id: Pipeline identifier
        state_store: StateStore instance for Redis operations
        scheduler: Scheduler instance for queue operations

    Returns:
        True if pipeline was resumed, False if not found
    """
    # Load state from Redis
    state = await state_store.load_state(pipeline_id)

    if state is None:
        return False

    # Load scenario
    scenario = load_scenario(state.scenario_name)
    scenario.compute_dependencies()

    # Find failed steps with satisfied dependencies
    for step_id, step_def in scenario.steps.items():
        status = await state_store.get_step_status(pipeline_id, step_id)

        if status == StepStatus.FAILED:
            # Check if dependencies are satisfied
            if await scheduler.all_deps_complete(pipeline_id, step_def):
                # Reset to pending and enqueue
                await state_store.set_step_status(pipeline_id, step_id, StepStatus.PENDING)
                await scheduler.enqueue_step(pipeline_id, step_id)

    # Update state timestamp
    state.updated_at = datetime.now(timezone.utc)
    await state_store.save_state(state)

    return True


async def list_pipelines(
    state_store: StateStore,
    status: Optional[str] = None,
    limit: int = 100,
) -> list[PipelineStatus]:
    """
    List pipelines, optionally filtered by status.

    Note: This is a simplified implementation that scans Redis keys.
    For production, consider maintaining a separate index.

    Args:
        state_store: StateStore instance for Redis operations
        status: Optional status filter (running, complete, failed, pending)
        limit: Maximum number of pipelines to return

    Returns:
        List of PipelineStatus objects
    """
    # Get all pipeline state keys
    pattern = f"{state_store.key_prefix}:pipeline:*:state"
    keys = []

    # Use SCAN to avoid blocking Redis
    cursor = 0
    while True:
        cursor, batch = await state_store.redis.scan(
            cursor=cursor,
            match=pattern,
            count=100
        )
        keys.extend(batch)

        if cursor == 0:
            break

    # Load states and build status objects
    statuses = []

    for key in keys[:limit]:
        # Extract pipeline_id from key
        # Key format: dreamer:pipeline:{pipeline_id}:state
        if isinstance(key, bytes):
            key = key.decode('utf-8')

        parts = key.split(':')
        if len(parts) >= 3:
            pipeline_id = parts[2]

            # Load state
            state = await state_store.load_state(pipeline_id)

            if state is None:
                continue

            # Load scenario
            try:
                scenario = load_scenario(state.scenario_name)
            except FileNotFoundError:
                continue

            # Get status
            pipeline_status = await get_status(pipeline_id, state_store, scenario)

            if pipeline_status is None:
                continue

            # Filter by status if requested
            if status is None or pipeline_status.status == status:
                statuses.append(pipeline_status)

    return statuses[:limit]


async def restart_from_step(
    conversation_id: str,
    branch: int,
    step_id: str,
    config: ChatConfig,
    model_name: str,
    state_store: StateStore,
    scheduler: Scheduler,
    scenario_name: Optional[str] = None,
    persona_id: Optional[str] = None,
    user_id: Optional[str] = None,
    query_text: Optional[str] = None,
    guidance: Optional[str] = None,
    mood: Optional[str] = None,
    include_all_history: bool = False,
    same_branch: bool = False,
) -> str:
    """
    Restart a scenario pipeline from a specific step using existing conversation data.

    This function allows replaying a scenario from a particular step by:
    1. Loading the conversation history for the given branch
    2. Inferring the scenario type (if not provided) from document types
    3. Finding all documents produced up to (but not including) the target step
    4. Creating a new pipeline with that preloaded context
    5. Starting execution from the target step

    Args:
        conversation_id: ID of the conversation to restart from
        branch: Branch number to restart from
        step_id: Step ID to restart from (this step will be re-executed)
        config: ChatConfig with provider keys and settings
        model_name: Name of the model to use
        state_store: StateStore instance for Redis operations
        scheduler: Scheduler instance for queue operations
        scenario_name: Optional scenario name (inferred from docs if not provided)
        persona_id: Persona ID to use (falls back to conversation or config)
        user_id: User ID (falls back to config)
        query_text: Optional query for journaler/philosopher scenarios
        guidance: Optional guidance text
        mood: Optional persona mood
        include_all_history: If True, load entire conversation history (all branches)
            into context, not just the specified branch
        same_branch: If True, continue on the same branch instead of creating a new one

    Returns:
        pipeline_id for tracking the new pipeline

    Raises:
        ValueError: If scenario cannot be inferred or step not found
        FileNotFoundError: If scenario not found
    """
    # 1. Load conversation model and history
    cvm = ConversationModel.from_config(config)
    roster = Roster.from_config(config)

    # Get conversation history for this branch
    history_df = cvm.get_conversation_history(conversation_id)

    if history_df.empty:
        raise ValueError(f"Conversation {conversation_id} not found or empty")

    # Filter to specific branch
    branch_df = history_df[history_df['branch'] == branch]

    if branch_df.empty:
        raise ValueError(f"Branch {branch} not found in conversation {conversation_id}")

    # 2. Infer scenario if not provided
    if scenario_name is None:
        doc_types = set(branch_df['document_type'].unique())
        scenario_name = infer_scenario_from_documents(doc_types)

        if scenario_name is None:
            raise ValueError(
                f"Could not infer scenario from document types: {doc_types}. "
                "Please provide scenario_name explicitly."
            )

    # 3. Load and validate scenario
    scenario = load_scenario(scenario_name)
    scenario.compute_dependencies()

    # Validate step_id exists in scenario
    if step_id not in scenario.steps:
        raise ValueError(
            f"Step '{step_id}' not found in scenario '{scenario_name}'. "
            f"Available steps: {list(scenario.steps.keys())}"
        )

    # 4. Resolve persona_id
    resolved_persona_id = persona_id
    if not resolved_persona_id:
        # Try to get from conversation
        conv_personas = branch_df['persona_id'].unique()
        if len(conv_personas) > 0:
            resolved_persona_id = conv_personas[0]
        else:
            resolved_persona_id = config.persona_id

    if not resolved_persona_id:
        raise ValueError("Could not determine persona_id")

    persona = roster.personas[resolved_persona_id]

    # 5. Resolve user_id
    resolved_user_id = user_id or resolved_persona_id or config.user_id

    # 6. Validate model exists
    models = LanguageModelV2.index_models(config)
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found in available models")

    # 7. Determine which steps to skip (completed steps before target)
    # Get execution order
    topo_order = scenario.topological_order()
    target_idx = topo_order.index(step_id)

    # Steps before target in topo order are "completed"
    completed_steps = topo_order[:target_idx]

    # 8. Build context from existing documents
    # Find doc_ids from branch that correspond to completed steps
    step_doc_ids = {}
    context_doc_ids = []

    # Get documents sorted by sequence number to maintain order
    branch_sorted = branch_df.sort_values('sequence_no')

    # Build mapping of step outputs from existing documents
    for _, row in branch_sorted.iterrows():
        doc_type = row['document_type']
        doc_id = row['doc_id']

        # Match document types to steps that produce them
        for s_id in completed_steps:
            step_def = scenario.steps[s_id]
            if step_def.output.document_type == doc_type:
                step_doc_ids[s_id] = doc_id
                context_doc_ids.append(doc_id)
                break

    # Build base context from conversation documents
    if include_all_history:
        # Load entire conversation history (all branches) as base context
        full_history = cvm.get_conversation_history(conversation_id)
        # Sort by branch then sequence to get chronological order
        full_history = full_history.sort_values(['branch', 'sequence_no'])
        # Include all document types except pipeline outputs from other branches
        all_conv_docs = full_history['doc_id'].tolist()
        context_doc_ids = all_conv_docs + context_doc_ids
    else:
        # Just add conversation docs from the target branch
        conv_docs = branch_df[branch_df['document_type'] == 'conversation']['doc_id'].tolist()
        context_doc_ids = conv_docs + context_doc_ids

    # 9. Create new pipeline state
    pipeline_id = generate_pipeline_id()
    new_branch = branch if same_branch else cvm.get_next_branch(conversation_id)

    state = PipelineState(
        pipeline_id=pipeline_id,
        scenario_name=scenario_name,
        conversation_id=conversation_id,
        persona_id=resolved_persona_id,
        user_id=resolved_user_id,
        model=model_name,
        thought_model=config.thought_model,
        codex_model=config.codex_model,
        guidance=guidance or config.guidance,
        query_text=query_text,
        persona_mood=mood or config.persona_mood,
        branch=new_branch,
        step_counter=target_idx + 1,  # Start numbering from target step
        completed_steps=completed_steps,
        step_doc_ids=step_doc_ids,
        context_doc_ids=context_doc_ids,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    # 10. Run seed actions for any steps that need them
    state = await run_seed_actions(scenario, state, cvm)

    # 11. Persist state to Redis
    await state_store.save_state(state)

    # 12. Initialize DAG with completed steps marked as complete
    await state_store.init_dag(pipeline_id, scenario)

    for completed_step in completed_steps:
        await state_store.set_step_status(pipeline_id, completed_step, StepStatus.COMPLETE)

    # 13. Enqueue the target step (and any other ready steps)
    # Check if target step's dependencies are satisfied
    target_step_def = scenario.steps[step_id]
    deps_satisfied = all(dep in completed_steps for dep in target_step_def.depends_on)

    if deps_satisfied:
        await scheduler.enqueue_step(pipeline_id, step_id)
    else:
        # Find which dependencies need to run first
        missing_deps = [dep for dep in target_step_def.depends_on if dep not in completed_steps]
        raise ValueError(
            f"Cannot restart from step '{step_id}' - missing dependencies: {missing_deps}. "
            f"Try restarting from one of: {missing_deps}"
        )

    return pipeline_id


async def get_restart_info(
    conversation_id: str,
    branch: int,
    config: ChatConfig,
    scenario_name: Optional[str] = None,
) -> dict:
    """
    Get information about what can be restarted from a conversation branch.

    Returns details about the scenario, completed steps, and available restart points.

    Args:
        conversation_id: ID of the conversation
        branch: Branch number to inspect
        config: ChatConfig for loading conversation model
        scenario_name: Optional scenario name (if not provided, will try to infer)

    Returns:
        Dict containing:
        - scenario_name: Inferred scenario name (or None)
        - doc_types: Set of document types found
        - step_outputs: Dict mapping step_id -> doc_id for found step outputs
        - available_restart_points: List of step_ids that can be restarted from
    """
    cvm = ConversationModel.from_config(config)
    history_df = cvm.get_conversation_history(conversation_id)

    if history_df.empty:
        return {
            "scenario_name": None,
            "doc_types": set(),
            "step_outputs": {},
            "available_restart_points": [],
            "error": f"Conversation {conversation_id} not found",
        }

    # Filter to branch
    branch_df = history_df[history_df['branch'] == branch]

    if branch_df.empty:
        return {
            "scenario_name": None,
            "doc_types": set(),
            "step_outputs": {},
            "available_restart_points": [],
            "error": f"Branch {branch} not found",
        }

    doc_types = set(branch_df['document_type'].unique())

    # Use provided scenario_name or try to infer
    if scenario_name is None:
        scenario_name = infer_scenario_from_documents(doc_types)

    result = {
        "scenario_name": scenario_name,
        "doc_types": doc_types,
        "branch": branch,
        "conversation_id": conversation_id,
        "step_outputs": {},
        "available_restart_points": [],
    }

    if scenario_name:
        try:
            scenario = load_scenario(scenario_name)
            scenario.compute_dependencies()

            # Get execution order for step numbering
            topo_order = scenario.topological_order()

            # First, try to match documents using step_name field (preferred)
            branch_sorted = branch_df.sort_values('sequence_no')
            has_step_name = 'step_name' in branch_df.columns

            if has_step_name:
                # Use step_name field directly
                for _, row in branch_sorted.iterrows():
                    step_name = row.get('step_name')
                    doc_id = row['doc_id']
                    if step_name and step_name in scenario.steps:
                        result["step_outputs"][step_name] = doc_id
            else:
                # Fallback: match by document type in execution order
                doc_type_docs = {}  # Track doc_ids by type in order

                for _, row in branch_sorted.iterrows():
                    doc_type = row['document_type']
                    doc_id = row['doc_id']
                    if doc_type not in doc_type_docs:
                        doc_type_docs[doc_type] = []
                    doc_type_docs[doc_type].append(doc_id)

                # Match documents to steps in execution order
                doc_type_index = {dt: 0 for dt in doc_type_docs}

                for step_id in topo_order:
                    step_def = scenario.steps[step_id]
                    output_type = step_def.output.document_type

                    if output_type in doc_type_docs:
                        idx = doc_type_index[output_type]
                        if idx < len(doc_type_docs[output_type]):
                            result["step_outputs"][step_id] = doc_type_docs[output_type][idx]
                            doc_type_index[output_type] = idx + 1

            # Build restart points with step numbers (1-indexed for display)
            result["available_restart_points"] = [
                {"step_num": i + 1, "step_id": step_id}
                for i, step_id in enumerate(topo_order)
            ]

        except FileNotFoundError:
            result["error"] = f"Scenario {scenario_name} not found"

    return result
