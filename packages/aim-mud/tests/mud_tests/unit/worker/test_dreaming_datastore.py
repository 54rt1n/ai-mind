# tests/mud_tests/unit/worker/test_dreaming_datastore.py
"""Tests for DreamingDatastoreMixin save/load round-trip."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import json


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client that behaves like fakeredis."""
    redis = AsyncMock()
    # Storage for hset/hgetall
    redis._storage = {}

    async def mock_hset(key, mapping=None, **kwargs):
        if key not in redis._storage:
            redis._storage[key] = {}
        if mapping:
            redis._storage[key].update(mapping)
        return len(mapping) if mapping else 0

    async def mock_hgetall(key):
        return redis._storage.get(key, {})

    async def mock_delete(key):
        if key in redis._storage:
            del redis._storage[key]
            return 1
        return 0

    redis.hset = mock_hset
    redis.hgetall = mock_hgetall
    redis.delete = mock_delete

    return redis


@pytest.fixture
def mock_worker(mock_redis):
    """Create a minimal worker mock with redis and config."""
    from andimud_worker.mixins.dreaming_datastore import DreamingDatastoreMixin
    from andimud_worker.config import MUDConfig

    class MockWorker(DreamingDatastoreMixin):
        def __init__(self):
            self.redis = mock_redis
            self.config = MUDConfig(agent_id="test_agent", persona_id="test_persona")

    return MockWorker()


@pytest.fixture
def sample_dreaming_state():
    """Create a sample DreamingState for testing."""
    from aim_mud_types.coordination import DreamingState, DreamStatus

    return DreamingState(
        pipeline_id="test-pipeline-123",
        agent_id="test_agent",
        status=DreamStatus.RUNNING,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        scenario_name="test_scenario",
        execution_order=["step1", "step2", "step3"],
        query="test query",
        guidance="test guidance",
        conversation_id="conv-123",
        base_model="test-model",
        step_index=0,
        completed_steps=[],
        step_doc_ids={},
        context_doc_ids=[],
        scenario_config={"name": "test"},
        persona_config={"persona_id": "test"},
    )


@pytest.mark.asyncio
async def test_save_dreaming_state_writes_all_fields(mock_worker, sample_dreaming_state, mock_redis):
    """Test that save_dreaming_state writes all required fields to Redis."""
    await mock_worker.save_dreaming_state(sample_dreaming_state)

    # Check what was saved
    from aim_mud_types.redis_keys import RedisKeys
    key = RedisKeys.agent_dreaming_state("test_agent")
    saved_data = mock_redis._storage.get(key, {})

    # Verify all required fields are present
    assert "pipeline_id" in saved_data
    assert "agent_id" in saved_data
    assert "status" in saved_data
    assert "created_at" in saved_data, "created_at must be saved"
    assert "updated_at" in saved_data, "updated_at must be saved"
    assert "scenario_name" in saved_data
    assert "execution_order" in saved_data
    assert "conversation_id" in saved_data
    assert "base_model" in saved_data
    assert "step_index" in saved_data
    assert "completed_steps" in saved_data
    assert "step_doc_ids" in saved_data
    assert "context_doc_ids" in saved_data
    assert "scenario_config" in saved_data
    assert "persona_config" in saved_data

    # Verify created_at is a valid Unix timestamp string
    assert saved_data["created_at"], "created_at should not be empty"
    # Should be an integer timestamp as string
    timestamp = int(saved_data["created_at"])
    # Verify it's a reasonable Unix timestamp (after 2020-01-01)
    assert timestamp > 1577836800, "created_at should be a valid Unix timestamp"


@pytest.mark.asyncio
async def test_save_load_round_trip(mock_worker, sample_dreaming_state, mock_redis):
    """Test that save followed by load returns equivalent state."""
    # Save
    await mock_worker.save_dreaming_state(sample_dreaming_state)

    # Load
    loaded_state = await mock_worker.load_dreaming_state("test_agent")

    assert loaded_state is not None, "load_dreaming_state should return a state"
    assert loaded_state.pipeline_id == sample_dreaming_state.pipeline_id
    assert loaded_state.agent_id == sample_dreaming_state.agent_id
    assert loaded_state.status == sample_dreaming_state.status
    assert loaded_state.scenario_name == sample_dreaming_state.scenario_name
    assert loaded_state.execution_order == sample_dreaming_state.execution_order
    assert loaded_state.conversation_id == sample_dreaming_state.conversation_id
    assert loaded_state.base_model == sample_dreaming_state.base_model
    assert loaded_state.step_index == sample_dreaming_state.step_index
    assert loaded_state.completed_steps == sample_dreaming_state.completed_steps


@pytest.mark.asyncio
async def test_load_with_strategy_fields(mock_worker, mock_redis):
    """Test loading a DreamingState that includes strategy-based fields (framework, state)."""
    from aim_mud_types.coordination import DreamingState, DreamStatus

    # Create state with strategy fields
    state = DreamingState(
        pipeline_id="strategy-test-123",
        agent_id="test_agent",
        status=DreamStatus.RUNNING,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        scenario_name="analysis_dialogue",
        execution_order=[],  # Not used for strategy-based
        conversation_id="conv-456",
        base_model="test-model",
        step_index=0,
        completed_steps=[],
        step_doc_ids={},
        context_doc_ids=[],
        scenario_config={},
        persona_config={},
        # Strategy-based fields
        framework='{"name": "test", "steps": {}}',
        state='{"current_step": "start", "step_results": {}}',
    )

    await mock_worker.save_dreaming_state(state)
    loaded = await mock_worker.load_dreaming_state("test_agent")

    assert loaded is not None
    assert loaded.framework == state.framework
    assert loaded.state == state.state


@pytest.mark.asyncio
async def test_load_nonexistent_state(mock_worker):
    """Test that loading a nonexistent state returns None."""
    result = await mock_worker.load_dreaming_state("nonexistent_agent")
    assert result is None


@pytest.mark.asyncio
async def test_delete_dreaming_state(mock_worker, sample_dreaming_state, mock_redis):
    """Test that delete removes the state from Redis."""
    # Save first
    await mock_worker.save_dreaming_state(sample_dreaming_state)

    # Verify it exists
    loaded = await mock_worker.load_dreaming_state("test_agent")
    assert loaded is not None

    # Delete
    await mock_worker.delete_dreaming_state("test_agent")

    # Verify it's gone
    loaded = await mock_worker.load_dreaming_state("test_agent")
    assert loaded is None
