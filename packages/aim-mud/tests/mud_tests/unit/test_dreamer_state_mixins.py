# packages/aim-mud/tests/mud_tests/unit/test_dreamer_state_mixins.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for DreamerStateMixin.has_running_dream() method.

Tests both sync and async implementations to ensure proper detection
of running dreams (PENDING or RUNNING status) vs completed/failed dreams.

IMPLEMENTATION DEFICIENCY DETECTED:
The has_running_dream() method in both async and sync mixins calls
redis.hgetall() directly and passes the result to DreamingState.model_validate()
without decoding bytes to strings. This works in tests with mocked string data,
but will fail in production where Redis returns dict[bytes, bytes] when
decode_responses=False (which is the configuration used by the system).

The implementation should either:
1. Use the existing _get_hash() helper which handles byte decoding
2. Add byte decoding logic before model_validate()

Files affected:
- packages/aim-mud/src/aim_mud_types/client/async_mixins/dreamer_state.py (line 72)
- packages/aim-mud/src/aim_mud_types/client/sync_mixins/dreamer_state.py (line 72)
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from aim_mud_types import RedisKeys
from aim_mud_types.client import RedisMUDClient, SyncRedisMUDClient
from aim_mud_types.models.coordination import DreamingState, DreamStatus


@pytest.fixture
def client():
    """Create RedisMUDClient with mocked Redis."""
    mock_redis = AsyncMock()
    return RedisMUDClient(mock_redis)


@pytest.fixture
def sync_client():
    """Create SyncRedisMUDClient with mocked Redis."""
    mock_redis = MagicMock()
    return SyncRedisMUDClient(mock_redis)


@pytest.fixture
def base_dreaming_state():
    """Create a base DreamingState for testing with different statuses.

    Returns a dict that can be modified for different test scenarios.

    NOTE: Using string keys/values because the implementation currently expects
    decoded data. The implementation has a deficiency where it doesn't decode
    bytes from Redis before calling model_validate(). See IMPLEMENTATION DEFICIENCY
    note in the test module docstring.
    """
    return {
        "pipeline_id": "dream-uuid-123",
        "agent_id": "andi",
        "status": "pending",  # Will be overridden in tests
        "created_at": "1704067200",
        "updated_at": "1704067200",
        "scenario_name": "test_scenario",
        "execution_order": '["step1", "step2"]',
        "conversation_id": "conv1",
        "base_model": "claude-opus-4",
        "step_index": "0",
        "completed_steps": "[]",
        "step_doc_ids": "{}",
        "context_doc_ids": "[]",
        "current_step_attempts": "0",
        "max_step_retries": "3",
        "heartbeat_timeout_seconds": "300",
        "scenario_config": "{}",
        "persona_config": "{}",
    }


class TestAsyncDreamerStateMixin:
    """Tests for AsyncDreamerStateMixin.has_running_dream()."""

    @pytest.mark.asyncio
    async def test_has_running_dream_with_pending_dream(self, client, base_dreaming_state):
        """Should return True when dream has PENDING status."""
        # Arrange
        base_dreaming_state["status"] = "pending"
        client.redis.hgetall.return_value = base_dreaming_state

        # Act
        result = await client.has_running_dream("andi")

        # Assert
        assert result is True
        client.redis.hgetall.assert_called_once_with(
            RedisKeys.agent_dreaming_state("andi")
        )

    @pytest.mark.asyncio
    async def test_has_running_dream_with_running_dream(self, client, base_dreaming_state):
        """Should return True when dream has RUNNING status."""
        # Arrange
        base_dreaming_state["status"] = "running"
        client.redis.hgetall.return_value = base_dreaming_state

        # Act
        result = await client.has_running_dream("andi")

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_has_running_dream_with_complete_dream(self, client, base_dreaming_state):
        """Should return False when dream has COMPLETE status."""
        # Arrange
        base_dreaming_state["status"] = "complete"
        base_dreaming_state["completed_at"] = "1704070800"
        client.redis.hgetall.return_value = base_dreaming_state

        # Act
        result = await client.has_running_dream("andi")

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_has_running_dream_with_failed_dream(self, client, base_dreaming_state):
        """Should return False when dream has FAILED status."""
        # Arrange
        base_dreaming_state["status"] = "failed"
        base_dreaming_state["last_error"] = "LLM timeout"
        client.redis.hgetall.return_value = base_dreaming_state

        # Act
        result = await client.has_running_dream("andi")

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_has_running_dream_with_aborted_dream(self, client, base_dreaming_state):
        """Should return False when dream has ABORTED status."""
        # Arrange
        base_dreaming_state["status"] = "aborted"
        client.redis.hgetall.return_value = base_dreaming_state

        # Act
        result = await client.has_running_dream("andi")

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_has_running_dream_no_dream(self, client):
        """Should return False when no dream exists."""
        # Arrange
        client.redis.hgetall.return_value = {}

        # Act
        result = await client.has_running_dream("andi")

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_has_running_dream_none_data(self, client):
        """Should return False when Redis returns None."""
        # Arrange
        client.redis.hgetall.return_value = None

        # Act
        result = await client.has_running_dream("andi")

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_has_running_dream_different_agent(self, client, base_dreaming_state):
        """Should check correct Redis key for different agent."""
        # Arrange
        base_dreaming_state["status"] = "running"
        base_dreaming_state["agent_id"] = "nova"
        client.redis.hgetall.return_value = base_dreaming_state

        # Act
        result = await client.has_running_dream("nova")

        # Assert
        assert result is True
        client.redis.hgetall.assert_called_once_with(
            RedisKeys.agent_dreaming_state("nova")
        )

    @pytest.mark.asyncio
    async def test_abort_running_dream_aborts_and_clears_state(self, client, base_dreaming_state):
        """abort_running_dream should archive and delete active PENDING/RUNNING dreams."""
        base_dreaming_state["status"] = "running"
        client.redis.hgetall.return_value = base_dreaming_state
        client.redis.lpush = AsyncMock(return_value=1)
        client.redis.ltrim = AsyncMock(return_value=True)
        client.redis.delete = AsyncMock(return_value=1)

        result = await client.abort_running_dream("andi", reason="manual abort")

        assert result is True
        client.redis.lpush.assert_called_once()
        client.redis.ltrim.assert_called_once_with(
            RedisKeys.agent_dreaming_history("andi"), 0, 99
        )
        client.redis.delete.assert_called_once_with(
            RedisKeys.agent_dreaming_state("andi")
        )

    @pytest.mark.asyncio
    async def test_abort_running_dream_returns_false_for_terminal_state(self, client, base_dreaming_state):
        """abort_running_dream should no-op for COMPLETE/FAILED/ABORTED dreams."""
        base_dreaming_state["status"] = "complete"
        base_dreaming_state["completed_at"] = "1704070800"
        client.redis.hgetall.return_value = base_dreaming_state
        client.redis.lpush = AsyncMock(return_value=1)
        client.redis.ltrim = AsyncMock(return_value=True)
        client.redis.delete = AsyncMock(return_value=1)

        result = await client.abort_running_dream("andi")

        assert result is False
        client.redis.lpush.assert_not_called()
        client.redis.ltrim.assert_not_called()
        client.redis.delete.assert_not_called()


class TestSyncDreamerStateMixin:
    """Tests for SyncDreamerStateMixin.has_running_dream()."""

    def test_has_running_dream_with_pending_dream(self, sync_client, base_dreaming_state):
        """Should return True when dream has PENDING status."""
        # Arrange
        base_dreaming_state["status"] = "pending"
        sync_client.redis.hgetall.return_value = base_dreaming_state

        # Act
        result = sync_client.has_running_dream("andi")

        # Assert
        assert result is True
        sync_client.redis.hgetall.assert_called_once_with(
            RedisKeys.agent_dreaming_state("andi")
        )

    def test_has_running_dream_with_running_dream(self, sync_client, base_dreaming_state):
        """Should return True when dream has RUNNING status."""
        # Arrange
        base_dreaming_state["status"] = "running"
        sync_client.redis.hgetall.return_value = base_dreaming_state

        # Act
        result = sync_client.has_running_dream("andi")

        # Assert
        assert result is True

    def test_has_running_dream_with_complete_dream(self, sync_client, base_dreaming_state):
        """Should return False when dream has COMPLETE status."""
        # Arrange
        base_dreaming_state["status"] = "complete"
        base_dreaming_state["completed_at"] = "1704070800"
        sync_client.redis.hgetall.return_value = base_dreaming_state

        # Act
        result = sync_client.has_running_dream("andi")

        # Assert
        assert result is False

    def test_has_running_dream_with_failed_dream(self, sync_client, base_dreaming_state):
        """Should return False when dream has FAILED status."""
        # Arrange
        base_dreaming_state["status"] = "failed"
        base_dreaming_state["last_error"] = "LLM timeout"
        sync_client.redis.hgetall.return_value = base_dreaming_state

        # Act
        result = sync_client.has_running_dream("andi")

        # Assert
        assert result is False

    def test_has_running_dream_with_aborted_dream(self, sync_client, base_dreaming_state):
        """Should return False when dream has ABORTED status."""
        # Arrange
        base_dreaming_state["status"] = "aborted"
        sync_client.redis.hgetall.return_value = base_dreaming_state

        # Act
        result = sync_client.has_running_dream("andi")

        # Assert
        assert result is False

    def test_has_running_dream_no_dream(self, sync_client):
        """Should return False when no dream exists."""
        # Arrange
        sync_client.redis.hgetall.return_value = {}

        # Act
        result = sync_client.has_running_dream("andi")

        # Assert
        assert result is False

    def test_has_running_dream_none_data(self, sync_client):
        """Should return False when Redis returns None."""
        # Arrange
        sync_client.redis.hgetall.return_value = None

        # Act
        result = sync_client.has_running_dream("andi")

        # Assert
        assert result is False

    def test_has_running_dream_different_agent(self, sync_client, base_dreaming_state):
        """Should check correct Redis key for different agent."""
        # Arrange
        base_dreaming_state["status"] = "running"
        base_dreaming_state["agent_id"] = "tiberius"
        sync_client.redis.hgetall.return_value = base_dreaming_state

        # Act
        result = sync_client.has_running_dream("tiberius")

        # Assert
        assert result is True
        sync_client.redis.hgetall.assert_called_once_with(
            RedisKeys.agent_dreaming_state("tiberius")
        )

    def test_abort_running_dream_aborts_and_clears_state(self, sync_client, base_dreaming_state):
        """abort_running_dream should archive and delete active PENDING/RUNNING dreams."""
        base_dreaming_state["status"] = "running"
        sync_client.redis.hgetall.return_value = base_dreaming_state

        result = sync_client.abort_running_dream("andi", reason="manual abort")

        assert result is True
        sync_client.redis.lpush.assert_called_once()
        sync_client.redis.ltrim.assert_called_once_with(
            RedisKeys.agent_dreaming_history("andi"), 0, 99
        )
        sync_client.redis.delete.assert_called_once_with(
            RedisKeys.agent_dreaming_state("andi")
        )

    def test_abort_running_dream_returns_false_for_terminal_state(self, sync_client, base_dreaming_state):
        """abort_running_dream should no-op for COMPLETE/FAILED/ABORTED dreams."""
        base_dreaming_state["status"] = "failed"
        base_dreaming_state["last_error"] = "timeout"
        sync_client.redis.hgetall.return_value = base_dreaming_state

        result = sync_client.abort_running_dream("andi")

        assert result is False
        sync_client.redis.lpush.assert_not_called()
        sync_client.redis.ltrim.assert_not_called()
        sync_client.redis.delete.assert_not_called()


class TestDreamingStateIntegration:
    """Integration tests verifying DreamingState deserialization."""

    @pytest.mark.asyncio
    async def test_deserialize_pending_state(self, client, base_dreaming_state):
        """Should properly deserialize PENDING DreamingState from Redis."""
        # Arrange
        base_dreaming_state["status"] = DreamStatus.PENDING.value
        client.redis.hgetall.return_value = base_dreaming_state

        # Act
        result = await client.has_running_dream("andi")

        # Assert - Validates DreamingState.model_validate() works correctly
        assert result is True

    @pytest.mark.asyncio
    async def test_deserialize_running_state(self, client, base_dreaming_state):
        """Should properly deserialize RUNNING DreamingState from Redis."""
        # Arrange
        base_dreaming_state["status"] = DreamStatus.RUNNING.value
        client.redis.hgetall.return_value = base_dreaming_state

        # Act
        result = await client.has_running_dream("andi")

        # Assert - Validates DreamingState.model_validate() works correctly
        assert result is True

    def test_sync_deserialize_complete_state(self, sync_client, base_dreaming_state):
        """Should properly deserialize COMPLETE DreamingState from Redis (sync)."""
        # Arrange
        base_dreaming_state["status"] = DreamStatus.COMPLETE.value
        base_dreaming_state["completed_at"] = "1704070800"
        sync_client.redis.hgetall.return_value = base_dreaming_state

        # Act
        result = sync_client.has_running_dream("andi")

        # Assert - Validates DreamingState.model_validate() works correctly
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
