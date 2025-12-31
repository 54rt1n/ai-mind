# tests/unit/dreamer/test_api.py
"""Unit tests for aim/dreamer/api.py"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from datetime import datetime, timezone
import pandas as pd
import uuid

from aim.dreamer.api import (
    generate_pipeline_id,
    run_seed_actions,
    start_pipeline,
    get_status,
    cancel_pipeline,
    resume_pipeline,
    refresh_pipeline,
    list_pipelines,
    PipelineStatus,
    ResumeResult,
)
from aim.dreamer.models import (
    PipelineState,
    Scenario,
    ScenarioContext,
    StepDefinition,
    StepOutput,
    StepStatus,
    MemoryAction,
)
from aim.dreamer.state import StateStore
from aim.dreamer.scheduler import Scheduler
from aim.config import ChatConfig


# Fixtures

@pytest.fixture
def mock_config():
    """Create a mock ChatConfig."""
    config = MagicMock(spec=ChatConfig)
    config.persona_id = "test_persona"
    config.user_id = "test_user"
    config.thought_model = None
    config.codex_model = None
    config.guidance = None
    config.persona_mood = None
    config.memory_path = "test_memory"
    config.embedding_model = "test_embedding"
    config.user_timezone = None
    return config


@pytest.fixture
def mock_state_store():
    """Create a mock StateStore."""
    store = AsyncMock(spec=StateStore)
    store.key_prefix = "dreamer"
    store.redis = AsyncMock()
    # Default to empty DAG for hgetall
    store.redis.hgetall.return_value = {}
    return store


@pytest.fixture
def mock_scheduler():
    """Create a mock Scheduler."""
    scheduler = AsyncMock(spec=Scheduler)
    return scheduler


@pytest.fixture
def simple_scenario():
    """Create a simple test scenario."""
    return Scenario(
        name="test_scenario",
        version=2,
        description="Test scenario",
        context=ScenarioContext(
            required_aspects=["coder"],
            core_documents=["summary"],
            location="Test location",
        ),
        seed=[],
        steps={
            "step1": StepDefinition(
                id="step1",
                prompt="Step 1 prompt",
                output=StepOutput(document_type="test", weight=1.0),
                next=["step2"],
            ),
            "step2": StepDefinition(
                id="step2",
                prompt="Step 2 prompt",
                output=StepOutput(document_type="test", weight=1.0),
                depends_on=["step1"],
                next=[],
            ),
        },
    )


@pytest.fixture
def mock_pipeline_state():
    """Create a mock PipelineState."""
    return PipelineState(
        pipeline_id="test-pipeline-123",
        scenario_name="test_scenario",
        conversation_id="test-conv",
        persona_id="test_persona",
        user_id="test_user",
        model="test-model",
        branch=1,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_cvm():
    """Create a mock ConversationModel with index.get_document support."""
    cvm = MagicMock()
    cvm.index = MagicMock()
    # Default: all documents exist
    cvm.index.get_document = MagicMock(return_value={"doc_id": "test-doc", "content": "test"})
    return cvm


# Tests for generate_pipeline_id

def test_generate_pipeline_id_returns_string():
    """Test that generate_pipeline_id returns a string."""
    pipeline_id = generate_pipeline_id()
    assert isinstance(pipeline_id, str)


def test_generate_pipeline_id_is_valid_uuid():
    """Test that generate_pipeline_id returns a valid UUID."""
    pipeline_id = generate_pipeline_id()
    # Should not raise an exception
    uuid.UUID(pipeline_id)


def test_generate_pipeline_id_is_unique():
    """Test that generate_pipeline_id returns unique IDs."""
    ids = [generate_pipeline_id() for _ in range(100)]
    # All IDs should be unique
    assert len(ids) == len(set(ids))


# Tests for run_seed_actions

@pytest.mark.asyncio
async def test_run_seed_actions_no_actions():
    """Test run_seed_actions with no seed actions."""
    scenario = Scenario(
        name="test",
        context=ScenarioContext(required_aspects=[]),
        steps={},
    )
    state = PipelineState(
        pipeline_id="test",
        scenario_name="test",
        conversation_id="conv",
        persona_id="persona",
        user_id="user",
        model="model",
        branch=1,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    cvm = AsyncMock()

    result = await run_seed_actions(scenario, state, cvm)

    assert result == state
    assert len(result.context_doc_ids) == 0


@pytest.mark.asyncio
async def test_run_seed_actions_load_conversation():
    """Test run_seed_actions with load_conversation action."""
    scenario = Scenario(
        name="test",
        context=ScenarioContext(required_aspects=[]),
        steps={},
        seed=[
            MemoryAction(
                action="load_conversation",
                document_types=["summary"],
            )
        ],
    )
    state = PipelineState(
        pipeline_id="test",
        scenario_name="test",
        conversation_id="conv",
        persona_id="persona",
        user_id="user",
        model="model",
        branch=1,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    # Mock CVM
    cvm = MagicMock()
    history_data = pd.DataFrame([
        {
            'doc_id': 'doc1',
            'role': 'user',
            'content': 'Hello',
            'document_type': 'conversation',
        },
        {
            'doc_id': 'doc2',
            'role': 'assistant',
            'content': 'Hi there',
            'think': 'Greeting',
            'document_type': 'conversation',
        },
    ])
    cvm.get_conversation_history.return_value = history_data

    result = await run_seed_actions(scenario, state, cvm)

    # Check that doc_ids were stored in context_doc_ids
    assert len(result.context_doc_ids) == 2
    assert result.context_doc_ids[0] == 'doc1'
    assert result.context_doc_ids[1] == 'doc2'


@pytest.mark.asyncio
async def test_run_seed_actions_search_memories():
    """Test run_seed_actions with search_memories action."""
    scenario = Scenario(
        name="test",
        context=ScenarioContext(required_aspects=[]),
        steps={},
        seed=[
            MemoryAction(
                action="search_memories",
                document_types=["journal"],
                top_n=5,
            )
        ],
    )
    state = PipelineState(
        pipeline_id="test",
        scenario_name="test",
        conversation_id="conv",
        persona_id="persona",
        user_id="user",
        model="model",
        branch=1,
        query_text="What happened yesterday?",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    # Mock CVM
    cvm = MagicMock()
    memories_data = pd.DataFrame([
        {
            'doc_id': 'mem1',
            'role': 'assistant',
            'content': 'Memory 1',
            'document_type': 'journal',
        },
        {
            'doc_id': 'mem2',
            'role': 'assistant',
            'content': 'Memory 2',
            'document_type': 'journal',
        },
    ])
    cvm.query.return_value = memories_data

    result = await run_seed_actions(scenario, state, cvm)

    # Check that doc_ids were stored in context_doc_ids
    assert len(result.context_doc_ids) == 2
    assert result.context_doc_ids[0] == 'mem1'
    assert result.context_doc_ids[1] == 'mem2'


# Tests for start_pipeline

@pytest.mark.asyncio
async def test_start_pipeline_success(mock_config, mock_state_store, mock_scheduler, simple_scenario):
    """Test successful pipeline start."""
    with patch('aim.dreamer.api.load_scenario') as mock_load_scenario, \
         patch('aim.dreamer.api.ConversationModel') as mock_cvm_class, \
         patch('aim.dreamer.api.Roster') as mock_roster_class, \
         patch('aim.dreamer.api.LanguageModelV2') as mock_model_class, \
         patch('aim.dreamer.api.run_seed_actions') as mock_run_seed:

        # Setup mocks
        mock_load_scenario.return_value = simple_scenario

        mock_cvm = MagicMock()
        mock_cvm.get_next_branch.return_value = 1
        # Mock index.search to return a valid conversation
        mock_cvm.index.search.return_value = pd.DataFrame([{
            'doc_id': 'conv-doc-1',
            'persona_id': 'test_persona',
            'conversation_id': 'test-conv',
        }])
        mock_cvm_class.from_config.return_value = mock_cvm

        mock_persona = MagicMock()
        mock_roster = MagicMock()
        mock_roster.personas = {"test_persona": mock_persona}
        mock_roster_class.from_config.return_value = mock_roster

        mock_models = {"test-model": MagicMock()}
        mock_model_class.index_models.return_value = mock_models

        mock_run_seed.return_value = MagicMock(spec=PipelineState)

        # Call start_pipeline
        pipeline_id = await start_pipeline(
            scenario_name="test_scenario",
            conversation_id="test-conv",
            config=mock_config,
            model_name="test-model",
            state_store=mock_state_store,
            scheduler=mock_scheduler,
            query_text="test query",
        )

        # Assertions
        assert isinstance(pipeline_id, str)
        uuid.UUID(pipeline_id)  # Should be valid UUID

        # Check that scenario was loaded and computed
        mock_load_scenario.assert_called_once_with("test_scenario")

        # Check that state was saved
        mock_state_store.save_state.assert_called_once()

        # Check that DAG was initialized
        mock_state_store.init_dag.assert_called_once()

        # Check that root steps were enqueued
        assert mock_scheduler.enqueue_step.call_count == 1
        mock_scheduler.enqueue_step.assert_called_with(pipeline_id, "step1")


@pytest.mark.asyncio
async def test_start_pipeline_model_not_found(mock_config, mock_state_store, mock_scheduler, simple_scenario):
    """Test start_pipeline with invalid model name."""
    with patch('aim.dreamer.api.load_scenario') as mock_load_scenario, \
         patch('aim.dreamer.api.ConversationModel') as mock_cvm_class, \
         patch('aim.dreamer.api.Roster') as mock_roster_class, \
         patch('aim.dreamer.api.LanguageModelV2') as mock_model_class:

        # Setup mocks
        mock_load_scenario.return_value = simple_scenario

        mock_cvm = MagicMock()
        mock_cvm.get_next_branch.return_value = 1
        # Mock index.search to return a valid conversation
        mock_cvm.index.search.return_value = pd.DataFrame([{
            'doc_id': 'conv-doc-1',
            'persona_id': 'test_persona',
            'conversation_id': 'test-conv',
        }])
        mock_cvm_class.from_config.return_value = mock_cvm

        mock_persona = MagicMock()
        mock_roster = MagicMock()
        mock_roster.personas = {"test_persona": mock_persona}
        mock_roster_class.from_config.return_value = mock_roster

        # Empty models dict
        mock_model_class.index_models.return_value = {}

        # Should raise ValueError
        with pytest.raises(ValueError, match="Model .* not found"):
            await start_pipeline(
                scenario_name="test_scenario",
                conversation_id="test-conv",
                config=mock_config,
                model_name="nonexistent-model",
                state_store=mock_state_store,
                scheduler=mock_scheduler,
            )


# Tests for get_status

@pytest.mark.asyncio
async def test_get_status_pipeline_not_found(mock_state_store, simple_scenario):
    """Test get_status for non-existent pipeline."""
    mock_state_store.load_state.return_value = None

    status = await get_status("nonexistent", mock_state_store, simple_scenario)

    assert status is None


@pytest.mark.asyncio
async def test_get_status_complete_pipeline(mock_state_store, simple_scenario, mock_pipeline_state):
    """Test get_status for completed pipeline."""
    # Setup mocks
    mock_state_store.load_state.return_value = mock_pipeline_state

    # All steps complete
    async def mock_get_step_status(pipeline_id, step_id):
        return StepStatus.COMPLETE

    mock_state_store.get_step_status.side_effect = mock_get_step_status

    status = await get_status("test-pipeline-123", mock_state_store, simple_scenario)

    assert status is not None
    assert status.pipeline_id == "test-pipeline-123"
    assert status.status == "complete"
    assert status.progress_percent == 100.0
    assert len(status.completed_steps) == 2
    assert len(status.failed_steps) == 0


@pytest.mark.asyncio
async def test_get_status_running_pipeline(mock_state_store, simple_scenario, mock_pipeline_state):
    """Test get_status for running pipeline."""
    # Setup mocks
    mock_state_store.load_state.return_value = mock_pipeline_state

    # First step complete, second running
    step_statuses = {
        "step1": StepStatus.COMPLETE,
        "step2": StepStatus.RUNNING,
    }

    async def mock_get_step_status(pipeline_id, step_id):
        return step_statuses[step_id]

    mock_state_store.get_step_status.side_effect = mock_get_step_status

    status = await get_status("test-pipeline", mock_state_store, simple_scenario)

    assert status is not None
    assert status.status == "running"
    assert status.current_step == "step2"
    assert status.progress_percent == 50.0
    assert len(status.completed_steps) == 1
    assert len(status.failed_steps) == 0


@pytest.mark.asyncio
async def test_get_status_failed_pipeline(mock_state_store, simple_scenario, mock_pipeline_state):
    """Test get_status for failed pipeline."""
    # Setup mocks
    mock_state_store.load_state.return_value = mock_pipeline_state

    # First step failed, second pending
    step_statuses = {
        "step1": StepStatus.FAILED,
        "step2": StepStatus.PENDING,
    }

    async def mock_get_step_status(pipeline_id, step_id):
        return step_statuses[step_id]

    mock_state_store.get_step_status.side_effect = mock_get_step_status

    status = await get_status("test-pipeline", mock_state_store, simple_scenario)

    assert status is not None
    assert status.status == "failed"
    assert status.progress_percent == 0.0
    assert len(status.completed_steps) == 0
    assert len(status.failed_steps) == 1


# Tests for cancel_pipeline

@pytest.mark.asyncio
async def test_cancel_pipeline_success(mock_state_store, mock_pipeline_state, simple_scenario):
    """Test successful pipeline cancellation."""
    with patch('aim.dreamer.api.load_scenario') as mock_load_scenario:
        # Setup mocks
        mock_state_store.load_state.return_value = mock_pipeline_state
        mock_load_scenario.return_value = simple_scenario

        # First step running, second pending
        step_statuses = {
            "step1": StepStatus.RUNNING,
            "step2": StepStatus.PENDING,
        }

        async def mock_get_step_status(pipeline_id, step_id):
            return step_statuses[step_id]

        mock_state_store.get_step_status.side_effect = mock_get_step_status

        result = await cancel_pipeline("test-pipeline", mock_state_store)

        assert result is True

        # Both steps should be marked as failed
        assert mock_state_store.set_step_status.call_count == 2

        # State should be saved
        mock_state_store.save_state.assert_called_once()


@pytest.mark.asyncio
async def test_cancel_pipeline_not_found(mock_state_store):
    """Test cancel_pipeline for non-existent pipeline."""
    mock_state_store.load_state.return_value = None

    result = await cancel_pipeline("nonexistent", mock_state_store)

    assert result is False


# Tests for resume_pipeline

@pytest.mark.asyncio
async def test_resume_pipeline_success(mock_state_store, mock_scheduler, mock_pipeline_state, simple_scenario, mock_cvm):
    """Test successful pipeline resume."""
    with patch('aim.dreamer.api.load_scenario') as mock_load_scenario:
        # Setup mocks
        mock_state_store.load_state.return_value = mock_pipeline_state
        mock_load_scenario.return_value = simple_scenario

        # Mock DAG entries - both steps exist in DAG
        mock_state_store.redis.hgetall.return_value = {
            b"step1": b"failed",
            b"step2": b"pending",
        }

        # First step failed, second pending (deps not satisfied)
        step_statuses = {
            "step1": StepStatus.FAILED,
            "step2": StepStatus.PENDING,
        }

        async def mock_get_step_status(pipeline_id, step_id):
            return step_statuses[step_id]

        mock_state_store.get_step_status.side_effect = mock_get_step_status

        # First step has no deps, so can be resumed
        mock_scheduler.all_deps_complete.return_value = True

        result = await resume_pipeline("test-pipeline", mock_state_store, mock_scheduler, mock_cvm)

        assert result.found is True

        # Step1 should be reset to pending and enqueued
        mock_state_store.set_step_status.assert_called()
        mock_scheduler.enqueue_step.assert_called()

        # State should be saved
        mock_state_store.save_state.assert_called_once()


@pytest.mark.asyncio
async def test_resume_pipeline_not_found(mock_state_store, mock_scheduler, mock_cvm):
    """Test resume_pipeline for non-existent pipeline."""
    mock_state_store.load_state.return_value = None

    result = await resume_pipeline("nonexistent", mock_state_store, mock_scheduler, mock_cvm)

    assert result.found is False


@pytest.mark.asyncio
async def test_resume_complete_pipeline_with_new_step(mock_state_store, mock_scheduler, mock_pipeline_state, mock_cvm):
    """Test resuming a complete pipeline with new steps added to scenario.

    This tests the Resume functionality where:
    1. Pipeline completed with steps A, B
    2. Scenario YAML updated to add step C (depends on B)
    3. User clicks Resume (force=True)
    4. New step C should be detected and enqueued
    """
    # Create scenario with 3 steps - original 2 + 1 new
    scenario_with_new_step = Scenario(
        name="test_scenario",
        version=2,
        description="Test scenario with new step",
        context=ScenarioContext(
            required_aspects=["coder"],
            core_documents=["summary"],
            location="Test location",
        ),
        seed=[],
        steps={
            "step1": StepDefinition(
                id="step1",
                prompt="Step 1 prompt",
                output=StepOutput(document_type="test", weight=1.0),
                next=["step2"],
            ),
            "step2": StepDefinition(
                id="step2",
                prompt="Step 2 prompt",
                output=StepOutput(document_type="test", weight=1.0),
                depends_on=["step1"],
                next=["step3"],  # Now points to new step
            ),
            "step3": StepDefinition(
                id="step3",
                prompt="Step 3 prompt - NEW",
                output=StepOutput(document_type="test", weight=1.0),
                depends_on=["step2"],
                next=[],
            ),
        },
    )

    with patch('aim.dreamer.api.load_scenario') as mock_load_scenario:
        # Setup mocks
        # Pipeline state shows only step1, step2 were completed
        mock_pipeline_state.completed_steps = ["step1", "step2"]
        mock_pipeline_state.step_doc_ids = {"step1": "doc-step1", "step2": "doc-step2"}
        mock_state_store.load_state.return_value = mock_pipeline_state
        mock_load_scenario.return_value = scenario_with_new_step

        # Mock DAG entries - only step1, step2 exist (step3 is new)
        mock_state_store.redis.hgetall.return_value = {
            b"step1": b"complete",
            b"step2": b"complete",
        }

        # DAG has step1, step2 as COMPLETE, step3 is NOT in DAG (returns PENDING by default)
        step_statuses = {
            "step1": StepStatus.COMPLETE,
            "step2": StepStatus.COMPLETE,
            "step3": StepStatus.PENDING,  # New step, not in original DAG
        }

        async def mock_get_step_status(pipeline_id, step_id):
            return step_statuses.get(step_id, StepStatus.PENDING)

        mock_state_store.get_step_status.side_effect = mock_get_step_status

        # Step3 depends on step2, which is COMPLETE, so all_deps_complete should return True
        # Let's use the real implementation logic instead of mocking
        async def mock_all_deps_complete(pipeline_id, step_def):
            # Simulate real logic: check if all depends_on steps are COMPLETE
            if not step_def.depends_on:
                return True
            for dep_id in step_def.depends_on:
                status = step_statuses.get(dep_id, StepStatus.PENDING)
                if status != StepStatus.COMPLETE:
                    return False
            return True

        mock_scheduler.all_deps_complete.side_effect = mock_all_deps_complete

        # All documents exist in CVM
        mock_cvm.index.get_document.return_value = {"doc_id": "test", "content": "test"}

        # Call resume_pipeline with force=True (Resume click)
        result = await resume_pipeline("test-pipeline", mock_state_store, mock_scheduler, mock_cvm, force=True)

        # Assertions
        assert result.found is True
        assert "step3" in result.steps_enqueued, "New step3 should be enqueued"
        assert "step3" in result.new_steps_added, "step3 should be in new_steps_added"

        # Verify step3 was added to DAG
        mock_state_store.redis.hset.assert_called()

        # Verify step3 was enqueued
        enqueue_calls = mock_scheduler.enqueue_step.call_args_list
        enqueued_steps = [call[0][1] for call in enqueue_calls]
        assert "step3" in enqueued_steps, "step3 should have been enqueued"

        # State should be saved
        mock_state_store.save_state.assert_called_once()


@pytest.mark.asyncio
async def test_resume_pipeline_resets_steps_with_missing_documents(mock_state_store, mock_scheduler, mock_pipeline_state, simple_scenario, mock_cvm):
    """Test that resume_pipeline resets COMPLETE steps when their documents are missing from CVM."""
    with patch('aim.dreamer.api.load_scenario') as mock_load_scenario:
        # Setup mocks
        mock_pipeline_state.completed_steps = ["step1"]
        mock_pipeline_state.step_doc_ids = {"step1": "doc-step1"}
        mock_state_store.load_state.return_value = mock_pipeline_state
        mock_load_scenario.return_value = simple_scenario

        # Mock DAG entries
        mock_state_store.redis.hgetall.return_value = {
            b"step1": b"complete",
            b"step2": b"pending",
        }

        step_statuses = {
            "step1": StepStatus.COMPLETE,
            "step2": StepStatus.PENDING,
        }

        async def mock_get_step_status(pipeline_id, step_id):
            return step_statuses.get(step_id, StepStatus.PENDING)

        mock_state_store.get_step_status.side_effect = mock_get_step_status
        mock_scheduler.all_deps_complete.return_value = True

        # Document for step1 is MISSING in CVM
        def get_document_mock(doc_id):
            if doc_id == "doc-step1":
                return None  # Document missing!
            return {"doc_id": doc_id, "content": "test"}

        mock_cvm.index.get_document.side_effect = get_document_mock

        result = await resume_pipeline("test-pipeline", mock_state_store, mock_scheduler, mock_cvm)

        assert result.found is True
        # step1 should be reset because its document is missing
        assert "step1" in result.steps_reset, "step1 should be reset due to missing document"

        # Verify document was checked
        mock_cvm.index.get_document.assert_called_with("doc-step1")


@pytest.mark.asyncio
async def test_resume_pipeline_enqueues_after_document_verification(mock_state_store, mock_scheduler, mock_pipeline_state, simple_scenario, mock_cvm):
    """Test that resume_pipeline DOES enqueue steps after verifying documents."""
    with patch('aim.dreamer.api.load_scenario') as mock_load_scenario:
        # Setup mocks
        mock_pipeline_state.completed_steps = []
        mock_pipeline_state.step_doc_ids = {}
        mock_state_store.load_state.return_value = mock_pipeline_state
        mock_load_scenario.return_value = simple_scenario

        # Mock DAG entries - step1 failed, step2 pending
        mock_state_store.redis.hgetall.return_value = {
            b"step1": b"failed",
            b"step2": b"pending",
        }

        step_statuses = {
            "step1": StepStatus.FAILED,
            "step2": StepStatus.PENDING,
        }

        async def mock_get_step_status(pipeline_id, step_id):
            return step_statuses.get(step_id, StepStatus.PENDING)

        mock_state_store.get_step_status.side_effect = mock_get_step_status
        mock_scheduler.all_deps_complete.return_value = True

        result = await resume_pipeline("test-pipeline", mock_state_store, mock_scheduler, mock_cvm)

        assert result.found is True
        # resume_pipeline SHOULD enqueue failed steps
        assert "step1" in result.steps_enqueued, "resume should enqueue failed step1"
        mock_scheduler.enqueue_step.assert_called()


# Tests for refresh_pipeline

@pytest.mark.asyncio
async def test_refresh_pipeline_does_not_enqueue(mock_state_store, mock_scheduler, mock_pipeline_state, simple_scenario, mock_cvm):
    """Test that refresh_pipeline does NOT enqueue any steps - it only syncs state."""
    with patch('aim.dreamer.api.load_scenario') as mock_load_scenario:
        # Setup mocks
        mock_pipeline_state.completed_steps = ["step1"]
        mock_pipeline_state.step_doc_ids = {"step1": "doc-step1"}
        mock_state_store.load_state.return_value = mock_pipeline_state
        mock_load_scenario.return_value = simple_scenario

        # Mock DAG entries
        mock_state_store.redis.hgetall.return_value = {
            b"step1": b"complete",
            b"step2": b"pending",
        }

        step_statuses = {
            "step1": StepStatus.COMPLETE,
            "step2": StepStatus.PENDING,
        }

        async def mock_get_step_status(pipeline_id, step_id):
            return step_statuses.get(step_id, StepStatus.PENDING)

        mock_state_store.get_step_status.side_effect = mock_get_step_status

        # All documents exist
        mock_cvm.index.get_document.return_value = {"doc_id": "test", "content": "test"}

        result = await refresh_pipeline("test-pipeline", mock_state_store, mock_scheduler, mock_cvm)

        assert result.found is True
        # refresh_pipeline should NOT enqueue anything
        assert result.steps_enqueued == [], "refresh should NOT enqueue any steps"
        mock_scheduler.enqueue_step.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_pipeline_resets_steps_with_missing_documents(mock_state_store, mock_scheduler, mock_pipeline_state, simple_scenario, mock_cvm):
    """Test that refresh_pipeline resets COMPLETE steps when their documents are missing from CVM."""
    with patch('aim.dreamer.api.load_scenario') as mock_load_scenario:
        # Setup mocks
        mock_pipeline_state.completed_steps = ["step1"]
        mock_pipeline_state.step_doc_ids = {"step1": "doc-step1"}
        mock_state_store.load_state.return_value = mock_pipeline_state
        mock_load_scenario.return_value = simple_scenario

        # Mock DAG entries
        mock_state_store.redis.hgetall.return_value = {
            b"step1": b"complete",
            b"step2": b"pending",
        }

        step_statuses = {
            "step1": StepStatus.COMPLETE,
            "step2": StepStatus.PENDING,
        }

        async def mock_get_step_status(pipeline_id, step_id):
            return step_statuses.get(step_id, StepStatus.PENDING)

        mock_state_store.get_step_status.side_effect = mock_get_step_status

        # Document for step1 is MISSING in CVM
        def get_document_mock(doc_id):
            if doc_id == "doc-step1":
                return None  # Document missing!
            return {"doc_id": doc_id, "content": "test"}

        mock_cvm.index.get_document.side_effect = get_document_mock

        result = await refresh_pipeline("test-pipeline", mock_state_store, mock_scheduler, mock_cvm)

        assert result.found is True
        # step1 should be reset because its document is missing
        assert "step1" in result.steps_reset, "step1 should be reset due to missing document"
        # But refresh should NOT enqueue
        assert result.steps_enqueued == [], "refresh should NOT enqueue any steps"
        mock_scheduler.enqueue_step.assert_not_called()

        # Verify document was checked
        mock_cvm.index.get_document.assert_called_with("doc-step1")


@pytest.mark.asyncio
async def test_refresh_pipeline_not_found(mock_state_store, mock_scheduler, mock_cvm):
    """Test refresh_pipeline for non-existent pipeline."""
    mock_state_store.load_state.return_value = None

    result = await refresh_pipeline("nonexistent", mock_state_store, mock_scheduler, mock_cvm)

    assert result.found is False
    assert result.steps_enqueued == []


@pytest.mark.asyncio
async def test_refresh_pipeline_adds_new_steps_without_enqueue(mock_state_store, mock_scheduler, mock_pipeline_state, mock_cvm):
    """Test that refresh_pipeline adds new scenario steps to DAG but does NOT enqueue them."""
    # Create scenario with 3 steps - original 2 + 1 new
    scenario_with_new_step = Scenario(
        name="test_scenario",
        version=2,
        description="Test scenario with new step",
        context=ScenarioContext(
            required_aspects=["coder"],
            core_documents=["summary"],
            location="Test location",
        ),
        seed=[],
        steps={
            "step1": StepDefinition(
                id="step1",
                prompt="Step 1 prompt",
                output=StepOutput(document_type="test", weight=1.0),
                next=["step2"],
            ),
            "step2": StepDefinition(
                id="step2",
                prompt="Step 2 prompt",
                output=StepOutput(document_type="test", weight=1.0),
                depends_on=["step1"],
                next=["step3"],
            ),
            "step3": StepDefinition(
                id="step3",
                prompt="Step 3 prompt - NEW",
                output=StepOutput(document_type="test", weight=1.0),
                depends_on=["step2"],
                next=[],
            ),
        },
    )

    with patch('aim.dreamer.api.load_scenario') as mock_load_scenario:
        # Setup mocks
        mock_pipeline_state.completed_steps = ["step1", "step2"]
        mock_pipeline_state.step_doc_ids = {"step1": "doc-step1", "step2": "doc-step2"}
        mock_state_store.load_state.return_value = mock_pipeline_state
        mock_load_scenario.return_value = scenario_with_new_step

        # Mock DAG entries - only step1, step2 exist (step3 is new)
        mock_state_store.redis.hgetall.return_value = {
            b"step1": b"complete",
            b"step2": b"complete",
        }

        step_statuses = {
            "step1": StepStatus.COMPLETE,
            "step2": StepStatus.COMPLETE,
            "step3": StepStatus.PENDING,
        }

        async def mock_get_step_status(pipeline_id, step_id):
            return step_statuses.get(step_id, StepStatus.PENDING)

        mock_state_store.get_step_status.side_effect = mock_get_step_status

        # All documents exist
        mock_cvm.index.get_document.return_value = {"doc_id": "test", "content": "test"}

        result = await refresh_pipeline("test-pipeline", mock_state_store, mock_scheduler, mock_cvm)

        assert result.found is True
        # New step should be detected
        assert "step3" in result.new_steps_added, "step3 should be in new_steps_added"
        # But refresh should NOT enqueue
        assert result.steps_enqueued == [], "refresh should NOT enqueue any steps"
        mock_scheduler.enqueue_step.assert_not_called()

        # Verify step3 was added to DAG
        mock_state_store.redis.hset.assert_called()


# Tests for list_pipelines

@pytest.mark.asyncio
async def test_list_pipelines_empty(mock_state_store):
    """Test list_pipelines with no pipelines."""
    # Mock empty scan result
    mock_state_store.redis.scan.return_value = (0, [])

    result = await list_pipelines(mock_state_store)

    assert result == []


@pytest.mark.asyncio
async def test_list_pipelines_with_results(mock_state_store, mock_pipeline_state, simple_scenario):
    """Test list_pipelines with results."""
    with patch('aim.dreamer.api.load_scenario') as mock_load_scenario, \
         patch('aim.dreamer.api.get_status') as mock_get_status:

        # Mock scan result
        keys = [
            b"dreamer:pipeline:test-1:state",
            b"dreamer:pipeline:test-2:state",
        ]
        mock_state_store.redis.scan.return_value = (0, keys)

        # Mock load_state
        mock_state_store.load_state.return_value = mock_pipeline_state

        # Mock load_scenario
        mock_load_scenario.return_value = simple_scenario

        # Mock get_status
        status1 = PipelineStatus(
            pipeline_id="test-1",
            scenario_name="test_scenario",
            status="complete",
            current_step=None,
            completed_steps=["step1", "step2"],
            failed_steps=[],
            step_errors={},
            progress_percent=100.0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        status2 = PipelineStatus(
            pipeline_id="test-2",
            scenario_name="test_scenario",
            status="running",
            current_step="step2",
            completed_steps=["step1"],
            failed_steps=[],
            step_errors={},
            progress_percent=50.0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        mock_get_status.side_effect = [status1, status2]

        result = await list_pipelines(mock_state_store)

        assert len(result) == 2
        assert result[0].pipeline_id == "test-1"
        assert result[1].pipeline_id == "test-2"


@pytest.mark.asyncio
async def test_list_pipelines_with_status_filter(mock_state_store, mock_pipeline_state, simple_scenario):
    """Test list_pipelines with status filter."""
    with patch('aim.dreamer.api.load_scenario') as mock_load_scenario, \
         patch('aim.dreamer.api.get_status') as mock_get_status:

        # Mock scan result
        keys = [
            b"dreamer:pipeline:test-1:state",
            b"dreamer:pipeline:test-2:state",
        ]
        mock_state_store.redis.scan.return_value = (0, keys)

        # Mock load_state
        mock_state_store.load_state.return_value = mock_pipeline_state

        # Mock load_scenario
        mock_load_scenario.return_value = simple_scenario

        # Mock get_status - one complete, one running
        status1 = PipelineStatus(
            pipeline_id="test-1",
            scenario_name="test_scenario",
            status="complete",
            current_step=None,
            completed_steps=["step1", "step2"],
            failed_steps=[],
            step_errors={},
            progress_percent=100.0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        status2 = PipelineStatus(
            pipeline_id="test-2",
            scenario_name="test_scenario",
            status="running",
            current_step="step2",
            completed_steps=["step1"],
            failed_steps=[],
            step_errors={},
            progress_percent=50.0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        mock_get_status.side_effect = [status1, status2]

        # Filter for only running pipelines
        result = await list_pipelines(mock_state_store, status="running")

        # Should only get the running one
        assert len(result) == 1
        assert result[0].status == "running"


# Tests for dialogue flow resume/refresh

@pytest.fixture
def mock_dialogue_state():
    """Create a mock DialogueState."""
    from aim.dreamer.dialogue.models import DialogueState, DialogueTurn
    from datetime import datetime, timezone

    return DialogueState(
        pipeline_id="test-dialogue-123",
        strategy_name="test_dialogue_scenario",
        conversation_id="test-conv",
        persona_id="test_persona",
        user_id="test_user",
        model="test-model",
        branch=1,
        turns=[
            DialogueTurn(
                speaker_id="aspect:coder",
                content="Test turn 1",
                step_id="step1",
                doc_id="doc-step1",
                document_type="dialogue-coder",
                timestamp=datetime.now(timezone.utc),
            ),
        ],
        completed_steps=["step1"],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_resume_dialogue_pipeline_success(mock_state_store, mock_scheduler, mock_dialogue_state, simple_scenario, mock_cvm):
    """Test successful dialogue pipeline resume."""
    from aim.conversation.message import ConversationMessage

    with patch('aim.dreamer.api.load_scenario') as mock_load_scenario:
        # Setup mocks
        mock_state_store.get_state_type.return_value = 'dialogue'
        mock_state_store.load_dialogue_state.return_value = mock_dialogue_state
        mock_load_scenario.return_value = simple_scenario

        # Mock DAG entries
        mock_state_store.redis.hgetall.return_value = {
            b"step1": b"complete",
            b"step2": b"failed",
        }

        step_statuses = {
            "step1": StepStatus.COMPLETE,
            "step2": StepStatus.FAILED,
        }

        async def mock_get_step_status(pipeline_id, step_id):
            return step_statuses.get(step_id, StepStatus.PENDING)

        mock_state_store.get_step_status.side_effect = mock_get_step_status
        mock_scheduler.all_deps_complete.return_value = True

        # Mock load_conversation to return messages with step_name and branch
        mock_msg = MagicMock(spec=ConversationMessage)
        mock_msg.scenario_name = "test_dialogue_scenario"
        mock_msg.step_name = "step1"
        mock_msg.branch = 1
        mock_cvm.load_conversation.return_value = [mock_msg]

        result = await resume_pipeline("test-dialogue", mock_state_store, mock_scheduler, mock_cvm)

        assert result.found is True
        # step2 should be enqueued since it's failed and deps are satisfied
        assert "step2" in result.steps_enqueued
        mock_state_store.save_dialogue_state.assert_called_once()


@pytest.mark.asyncio
async def test_resume_dialogue_pipeline_resets_missing_docs(mock_state_store, mock_scheduler, mock_dialogue_state, simple_scenario, mock_cvm):
    """Test dialogue pipeline resume resets steps with missing documents."""
    with patch('aim.dreamer.api.load_scenario') as mock_load_scenario:
        # Setup mocks
        mock_state_store.get_state_type.return_value = 'dialogue'
        mock_state_store.load_dialogue_state.return_value = mock_dialogue_state
        mock_load_scenario.return_value = simple_scenario

        # Mock DAG entries
        mock_state_store.redis.hgetall.return_value = {
            b"step1": b"complete",
            b"step2": b"pending",
        }

        step_statuses = {
            "step1": StepStatus.COMPLETE,
            "step2": StepStatus.PENDING,
        }

        async def mock_get_step_status(pipeline_id, step_id):
            return step_statuses.get(step_id, StepStatus.PENDING)

        mock_state_store.get_step_status.side_effect = mock_get_step_status
        mock_scheduler.all_deps_complete.return_value = True

        # step1's document is missing - return empty list (no completed steps in CVM)
        mock_cvm.load_conversation.return_value = []

        result = await resume_pipeline("test-dialogue", mock_state_store, mock_scheduler, mock_cvm)

        assert result.found is True
        assert "step1" in result.steps_reset


@pytest.mark.asyncio
async def test_refresh_dialogue_pipeline_success(mock_state_store, mock_scheduler, mock_dialogue_state, simple_scenario, mock_cvm):
    """Test successful dialogue pipeline refresh."""
    from aim.conversation.message import ConversationMessage

    with patch('aim.dreamer.api.load_scenario') as mock_load_scenario:
        # Setup mocks
        mock_state_store.get_state_type.return_value = 'dialogue'
        mock_state_store.load_dialogue_state.return_value = mock_dialogue_state
        mock_load_scenario.return_value = simple_scenario

        # Mock DAG entries
        mock_state_store.redis.hgetall.return_value = {
            b"step1": b"complete",
            b"step2": b"pending",
        }

        step_statuses = {
            "step1": StepStatus.COMPLETE,
            "step2": StepStatus.PENDING,
        }

        async def mock_get_step_status(pipeline_id, step_id):
            return step_statuses.get(step_id, StepStatus.PENDING)

        mock_state_store.get_step_status.side_effect = mock_get_step_status

        # All documents exist - return message with step1 completed
        mock_msg = MagicMock(spec=ConversationMessage)
        mock_msg.scenario_name = "test_dialogue_scenario"
        mock_msg.step_name = "step1"
        mock_msg.branch = 1
        mock_cvm.load_conversation.return_value = [mock_msg]

        result = await refresh_pipeline("test-dialogue", mock_state_store, mock_scheduler, mock_cvm)

        assert result.found is True
        # Refresh should NOT enqueue any steps
        assert result.steps_enqueued == []
        mock_scheduler.enqueue_step.assert_not_called()
        mock_state_store.save_dialogue_state.assert_called_once()


@pytest.mark.asyncio
async def test_refresh_dialogue_pipeline_adds_new_steps(mock_state_store, mock_scheduler, mock_dialogue_state, mock_cvm):
    """Test dialogue pipeline refresh detects new scenario steps."""
    from aim.conversation.message import ConversationMessage

    # Create scenario with 3 steps
    scenario_with_new_step = Scenario(
        name="test_dialogue_scenario",
        version=2,
        description="Test dialogue scenario with new step",
        context=ScenarioContext(
            required_aspects=["coder"],
            core_documents=["summary"],
            location="Test location",
        ),
        seed=[],
        steps={
            "step1": StepDefinition(
                id="step1",
                prompt="Step 1 prompt",
                output=StepOutput(document_type="test", weight=1.0),
                next=["step2"],
            ),
            "step2": StepDefinition(
                id="step2",
                prompt="Step 2 prompt",
                output=StepOutput(document_type="test", weight=1.0),
                depends_on=["step1"],
                next=["step3"],
            ),
            "step3": StepDefinition(
                id="step3",
                prompt="Step 3 prompt - NEW",
                output=StepOutput(document_type="test", weight=1.0),
                depends_on=["step2"],
                next=[],
            ),
        },
    )

    with patch('aim.dreamer.api.load_scenario') as mock_load_scenario:
        # Setup mocks
        mock_state_store.get_state_type.return_value = 'dialogue'
        mock_state_store.load_dialogue_state.return_value = mock_dialogue_state
        mock_load_scenario.return_value = scenario_with_new_step

        # Only step1, step2 in DAG (step3 is new)
        mock_state_store.redis.hgetall.return_value = {
            b"step1": b"complete",
            b"step2": b"complete",
        }

        step_statuses = {
            "step1": StepStatus.COMPLETE,
            "step2": StepStatus.COMPLETE,
            "step3": StepStatus.PENDING,
        }

        async def mock_get_step_status(pipeline_id, step_id):
            return step_statuses.get(step_id, StepStatus.PENDING)

        mock_state_store.get_step_status.side_effect = mock_get_step_status

        # Both step1 and step2 have documents in CVM
        mock_msg1 = MagicMock(spec=ConversationMessage)
        mock_msg1.scenario_name = "test_dialogue_scenario"
        mock_msg1.step_name = "step1"
        mock_msg1.branch = 1
        mock_msg2 = MagicMock(spec=ConversationMessage)
        mock_msg2.scenario_name = "test_dialogue_scenario"
        mock_msg2.step_name = "step2"
        mock_msg2.branch = 1
        mock_cvm.load_conversation.return_value = [mock_msg1, mock_msg2]

        result = await refresh_pipeline("test-dialogue", mock_state_store, mock_scheduler, mock_cvm)

        assert result.found is True
        assert "step3" in result.new_steps_added
        # Refresh should NOT enqueue
        assert result.steps_enqueued == []
