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
    conversation_id: str,
    config: ChatConfig,
    model_name: str,
    state_store: StateStore,
    scheduler: Scheduler,
    query_text: Optional[str] = None,
) -> str:
    """
    Start a new pipeline execution.

    Args:
        scenario_name: Name of the scenario YAML to load
        conversation_id: ID of the conversation to analyze
        config: ChatConfig with provider keys and settings
        model_name: Name of the model to use (from models.yaml)
        state_store: StateStore instance for Redis operations
        scheduler: Scheduler instance for queue operations
        query_text: Optional query for journaler/philosopher scenarios

    Returns:
        pipeline_id for tracking

    Raises:
        FileNotFoundError: If scenario not found
        ValueError: If model not found
    """
    # 1. Load scenario
    scenario = load_scenario(scenario_name)
    scenario.compute_dependencies()

    # 2. Load existing infrastructure
    cvm = ConversationModel.from_config(config)
    roster = Roster.from_config(config)

    # 3. Get persona_id from the conversation itself
    conv_df = cvm.index.search(
        query_conversation_id=conversation_id,
        query_document_type='conversation',
        query_limit=1,
    )
    if conv_df.empty:
        raise ValueError(f"Conversation {conversation_id} not found")
    persona_id = conv_df.iloc[0]['persona_id']
    persona = roster.personas[persona_id]

    # 4. Validate model exists
    models = LanguageModelV2.index_models(config)
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found in available models")

    # 5. Initialize state
    pipeline_id = generate_pipeline_id()
    state = PipelineState(
        pipeline_id=pipeline_id,
        scenario_name=scenario_name,
        conversation_id=conversation_id,
        persona_id=persona_id,
        user_id=config.user_id,
        model=model_name,
        thought_model=config.thought_model,
        codex_model=config.codex_model,
        guidance=config.guidance,
        query_text=query_text,
        persona_mood=config.persona_mood,
        branch=cvm.get_next_branch(conversation_id),
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

    return PipelineStatus(
        pipeline_id=pipeline_id,
        scenario_name=state.scenario_name,
        status=overall_status,
        current_step=current_step,
        completed_steps=completed_steps,
        failed_steps=failed_steps,
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
