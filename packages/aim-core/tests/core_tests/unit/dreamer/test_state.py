# tests/unit/dreamer/test_state.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for StateStore Redis state management."""

import pytest
import pytest_asyncio
from datetime import datetime
from fakeredis import aioredis

from aim.dreamer.server.state import StateStore
from aim.dreamer.core.models import (
    PipelineState,
    StepStatus,
    StepResult,
    Scenario,
    ScenarioContext,
    StepDefinition,
    StepOutput,
)


@pytest_asyncio.fixture
async def redis_client():
    """Create a fake Redis client for testing."""
    client = aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.flushall()
    await client.aclose()


@pytest_asyncio.fixture
async def state_store(redis_client):
    """Create a StateStore instance with fake Redis."""
    return StateStore(redis_client, key_prefix="dreamer")


@pytest.fixture
def sample_state():
    """Create a sample PipelineState for testing."""
    return PipelineState(
        pipeline_id="test-pipeline-123",
        scenario_name="analyst",
        conversation_id="conv-456",
        persona_id="persona-789",
        user_id="user-001",
        model="claude-3-5-sonnet-20241022",
        thought_model="gpt-4-turbo",
        codex_model="gpt-4",
        guidance="Test guidance",
        query_text="Test query",
        persona_mood="focused",
        branch=1,
        step_counter=1,
        turns=[],
        extra=[],
        completed_steps=[],
        step_results={},
        created_at=datetime(2025, 1, 1, 12, 0, 0),
        updated_at=datetime(2025, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def sample_scenario():
    """Create a sample Scenario for testing."""
    return Scenario(
        name="test_scenario",
        version=2,
        description="Test scenario",
        context=ScenarioContext(
            required_aspects=["coder"],
            core_documents=["summary"],
            location="Test location",
        ),
        steps={
            "step1": StepDefinition(
                id="step1",
                prompt="Test prompt 1",
                output=StepOutput(document_type="test"),
                next=["step2"],
            ),
            "step2": StepDefinition(
                id="step2",
                prompt="Test prompt 2",
                output=StepOutput(document_type="test"),
                depends_on=["step1"],
            ),
        },
    )


class TestStateStore:
    """Test suite for StateStore."""

    @pytest.mark.asyncio
    async def test_save_and_load_state(self, state_store, sample_state):
        """Test save_state and load_state roundtrip."""
        # Save the state
        await state_store.save_state(sample_state)

        # Load it back
        loaded_state = await state_store.load_state(sample_state.pipeline_id)

        # Verify it matches
        assert loaded_state is not None
        assert loaded_state.pipeline_id == sample_state.pipeline_id
        assert loaded_state.scenario_name == sample_state.scenario_name
        assert loaded_state.conversation_id == sample_state.conversation_id
        assert loaded_state.persona_id == sample_state.persona_id
        assert loaded_state.user_id == sample_state.user_id
        assert loaded_state.model == sample_state.model
        assert loaded_state.thought_model == sample_state.thought_model
        assert loaded_state.codex_model == sample_state.codex_model
        assert loaded_state.guidance == sample_state.guidance
        assert loaded_state.query_text == sample_state.query_text
        assert loaded_state.persona_mood == sample_state.persona_mood
        assert loaded_state.branch == sample_state.branch
        assert loaded_state.step_counter == sample_state.step_counter

    @pytest.mark.asyncio
    async def test_load_state_missing_pipeline(self, state_store):
        """Test load_state returns None for missing pipeline."""
        loaded_state = await state_store.load_state("nonexistent-pipeline")
        assert loaded_state is None

    @pytest.mark.asyncio
    async def test_delete_state(self, state_store, sample_state, sample_scenario):
        """Test delete_state removes the state and DAG."""
        # Save state and initialize DAG
        await state_store.save_state(sample_state)
        await state_store.init_dag(sample_state.pipeline_id, sample_scenario)

        # Verify they exist
        assert await state_store.load_state(sample_state.pipeline_id) is not None
        step_status = await state_store.get_step_status(sample_state.pipeline_id, "step1")
        assert step_status == StepStatus.PENDING

        # Delete
        await state_store.delete_state(sample_state.pipeline_id)

        # Verify they're gone
        assert await state_store.load_state(sample_state.pipeline_id) is None
        # After deletion, step status should default to PENDING (key doesn't exist)
        step_status = await state_store.get_step_status(sample_state.pipeline_id, "step1")
        assert step_status == StepStatus.PENDING

    @pytest.mark.asyncio
    async def test_init_dag(self, state_store, sample_state, sample_scenario):
        """Test init_dag creates correct HASH entries."""
        await state_store.init_dag(sample_state.pipeline_id, sample_scenario)

        # Check that all steps are set to PENDING
        for step_id in sample_scenario.steps.keys():
            status = await state_store.get_step_status(sample_state.pipeline_id, step_id)
            assert status == StepStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_and_set_step_status(self, state_store, sample_state, sample_scenario):
        """Test get_step_status and set_step_status."""
        # Initialize DAG
        await state_store.init_dag(sample_state.pipeline_id, sample_scenario)

        # Verify initial status is PENDING
        status = await state_store.get_step_status(sample_state.pipeline_id, "step1")
        assert status == StepStatus.PENDING

        # Update to RUNNING
        await state_store.set_step_status(
            sample_state.pipeline_id, "step1", StepStatus.RUNNING
        )
        status = await state_store.get_step_status(sample_state.pipeline_id, "step1")
        assert status == StepStatus.RUNNING

        # Update to COMPLETE
        await state_store.set_step_status(
            sample_state.pipeline_id, "step1", StepStatus.COMPLETE
        )
        status = await state_store.get_step_status(sample_state.pipeline_id, "step1")
        assert status == StepStatus.COMPLETE

        # Update to FAILED
        await state_store.set_step_status(
            sample_state.pipeline_id, "step1", StepStatus.FAILED
        )
        status = await state_store.get_step_status(sample_state.pipeline_id, "step1")
        assert status == StepStatus.FAILED

    @pytest.mark.asyncio
    async def test_get_step_status_missing_step(self, state_store, sample_state):
        """Test get_step_status returns PENDING for missing step."""
        # Without initializing DAG
        status = await state_store.get_step_status(sample_state.pipeline_id, "nonexistent")
        assert status == StepStatus.PENDING

    @pytest.mark.asyncio
    async def test_acquire_lock_success(self, state_store, sample_state):
        """Test acquire_lock succeeds first time."""
        acquired = await state_store.acquire_lock(
            sample_state.pipeline_id, "step1", ttl=60
        )
        assert acquired is True

    @pytest.mark.asyncio
    async def test_acquire_lock_fails_when_held(self, state_store, sample_state):
        """Test acquire_lock fails second time when lock is held."""
        # First acquisition succeeds
        acquired1 = await state_store.acquire_lock(
            sample_state.pipeline_id, "step1", ttl=60
        )
        assert acquired1 is True

        # Second acquisition fails
        acquired2 = await state_store.acquire_lock(
            sample_state.pipeline_id, "step1", ttl=60
        )
        assert acquired2 is False

    @pytest.mark.asyncio
    async def test_release_lock_allows_reacquisition(self, state_store, sample_state):
        """Test release_lock allows re-acquisition."""
        # Acquire lock
        acquired1 = await state_store.acquire_lock(
            sample_state.pipeline_id, "step1", ttl=60
        )
        assert acquired1 is True

        # Release lock
        await state_store.release_lock(sample_state.pipeline_id, "step1")

        # Re-acquire should succeed
        acquired2 = await state_store.acquire_lock(
            sample_state.pipeline_id, "step1", ttl=60
        )
        assert acquired2 is True

    @pytest.mark.asyncio
    async def test_state_with_step_doc_ids(self, state_store, sample_state):
        """Test saving and loading state with step doc_id references."""
        # Add step doc_id references
        sample_state.step_doc_ids["step1"] = "doc-123"
        sample_state.step_doc_ids["step2"] = "doc-456"
        sample_state.completed_steps.append("step1")
        sample_state.completed_steps.append("step2")

        # Save and load
        await state_store.save_state(sample_state)
        loaded_state = await state_store.load_state(sample_state.pipeline_id)

        # Verify step doc_ids are preserved
        assert loaded_state is not None
        assert "step1" in loaded_state.step_doc_ids
        assert loaded_state.step_doc_ids["step1"] == "doc-123"
        assert "step2" in loaded_state.step_doc_ids
        assert loaded_state.step_doc_ids["step2"] == "doc-456"
        assert "step1" in loaded_state.completed_steps
        assert "step2" in loaded_state.completed_steps

    @pytest.mark.asyncio
    async def test_state_with_seed_doc_ids(self, state_store, sample_state):
        """Test saving and loading state with seed doc_id references."""
        # Add seed doc_ids for steps
        sample_state.seed_doc_ids["step1"] = ["seed-doc-1", "seed-doc-2"]
        sample_state.seed_doc_ids["step2"] = ["seed-doc-3"]

        # Save and load
        await state_store.save_state(sample_state)
        loaded_state = await state_store.load_state(sample_state.pipeline_id)

        # Verify seed doc_ids are preserved
        assert loaded_state is not None
        assert "step1" in loaded_state.seed_doc_ids
        assert len(loaded_state.seed_doc_ids["step1"]) == 2
        assert loaded_state.seed_doc_ids["step1"][0] == "seed-doc-1"
        assert loaded_state.seed_doc_ids["step1"][1] == "seed-doc-2"
        assert "step2" in loaded_state.seed_doc_ids
        assert len(loaded_state.seed_doc_ids["step2"]) == 1

    @pytest.mark.asyncio
    async def test_different_pipelines_isolated(self, state_store):
        """Test that different pipelines are isolated."""
        state1 = PipelineState(
            pipeline_id="pipeline-1",
            scenario_name="analyst",
            conversation_id="conv-1",
            persona_id="persona-1",
            user_id="user-1",
            model="model-1",
            branch=1,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        state2 = PipelineState(
            pipeline_id="pipeline-2",
            scenario_name="journaler",
            conversation_id="conv-2",
            persona_id="persona-2",
            user_id="user-2",
            model="model-2",
            branch=1,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Save both
        await state_store.save_state(state1)
        await state_store.save_state(state2)

        # Load and verify isolation
        loaded1 = await state_store.load_state("pipeline-1")
        loaded2 = await state_store.load_state("pipeline-2")

        assert loaded1.scenario_name == "analyst"
        assert loaded2.scenario_name == "journaler"
        assert loaded1.conversation_id != loaded2.conversation_id

    @pytest.mark.asyncio
    async def test_lock_different_steps_isolated(self, state_store, sample_state):
        """Test that locks for different steps are isolated."""
        # Acquire lock for step1
        acquired1 = await state_store.acquire_lock(
            sample_state.pipeline_id, "step1", ttl=60
        )
        assert acquired1 is True

        # Should be able to acquire lock for step2
        acquired2 = await state_store.acquire_lock(
            sample_state.pipeline_id, "step2", ttl=60
        )
        assert acquired2 is True

        # Should not be able to re-acquire step1
        acquired3 = await state_store.acquire_lock(
            sample_state.pipeline_id, "step1", ttl=60
        )
        assert acquired3 is False
