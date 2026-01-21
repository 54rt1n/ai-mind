# packages/aim-mud/tests/mud_tests/unit/worker/test_think_command.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for ThinkCommand."""

import pytest
from unittest.mock import MagicMock, AsyncMock
import json

from aim_mud_types import RedisKeys, TurnRequestStatus
from andimud_worker.commands.think import ThinkCommand


@pytest.fixture
def mock_worker():
    """Mock worker with minimal required attributes."""
    worker = MagicMock()
    worker.config = MagicMock()
    worker.config.agent_id = "test_agent"
    worker.redis = AsyncMock()
    worker._response_strategy = MagicMock()
    worker._response_strategy.thought_content = None
    worker.pending_events = []
    return worker


class TestThinkCommand:
    """Tests for ThinkCommand execution."""

    @pytest.mark.asyncio
    async def test_think_command_reads_thought(self, mock_worker):
        """Test ThinkCommand reads thought from Redis and injects into strategy."""
        # Setup mock Redis to return thought data
        thought_data = json.dumps({
            "content": "Focus on emotional memories",
            "source": "manual",
            "timestamp": 1234567890
        })
        mock_worker.redis.get.return_value = thought_data.encode("utf-8")

        cmd = ThinkCommand()
        result = await cmd.execute(mock_worker, turn_id="test_turn", metadata={})

        # Verify Redis key was read
        expected_key = RedisKeys.agent_thought("test_agent")
        mock_worker.redis.get.assert_called_once_with(expected_key)

        # Verify thought was injected
        assert mock_worker._response_strategy.thought_content == "Focus on emotional memories"

        # Verify result
        assert result.complete is False
        assert result.status == TurnRequestStatus.DONE
        assert "Focus on emotional memories" in result.message

    @pytest.mark.asyncio
    async def test_think_command_no_thought(self, mock_worker):
        """Test ThinkCommand handles missing thought gracefully."""
        mock_worker.redis.get.return_value = None

        cmd = ThinkCommand()
        result = await cmd.execute(mock_worker, turn_id="test_turn", metadata={})

        # Verify thought was not set on strategy
        assert mock_worker._response_strategy.thought_content is None

        # Verify result
        assert result.complete is False
        assert result.status == TurnRequestStatus.DONE
        assert "Think turn ready" in result.message

    @pytest.mark.asyncio
    async def test_think_command_invalid_json(self, mock_worker):
        """Test ThinkCommand handles invalid JSON gracefully."""
        mock_worker.redis.get.return_value = b"invalid json {{{}"

        cmd = ThinkCommand()
        result = await cmd.execute(mock_worker, turn_id="test_turn", metadata={})

        # Verify thought was not set on strategy
        assert mock_worker._response_strategy.thought_content is None

        # Verify result still succeeds
        assert result.complete is False
        assert result.status == TurnRequestStatus.DONE

    @pytest.mark.asyncio
    async def test_think_command_with_guidance(self, mock_worker):
        """Test ThinkCommand preserves guidance from metadata."""
        thought_data = json.dumps({
            "content": "Test thought",
            "source": "manual",
            "timestamp": 1234567890
        })
        mock_worker.redis.get.return_value = thought_data.encode("utf-8")

        metadata = {"guidance": "User provided guidance"}
        cmd = ThinkCommand()
        result = await cmd.execute(mock_worker, turn_id="test_turn", metadata=metadata)

        # Verify guidance is preserved
        assert result.plan_guidance == "User provided guidance"

    @pytest.mark.asyncio
    async def test_think_command_no_response_strategy(self, mock_worker):
        """Test ThinkCommand handles missing response strategy gracefully."""
        mock_worker._response_strategy = None
        thought_data = json.dumps({
            "content": "Test thought",
            "source": "manual",
            "timestamp": 1234567890
        })
        mock_worker.redis.get.return_value = thought_data.encode("utf-8")

        cmd = ThinkCommand()
        result = await cmd.execute(mock_worker, turn_id="test_turn", metadata={})

        # Should not crash, just skip injection
        assert result.complete is False
        assert result.status == TurnRequestStatus.DONE

    @pytest.mark.asyncio
    async def test_think_command_name_property(self):
        """Test ThinkCommand has correct name."""
        cmd = ThinkCommand()
        assert cmd.name == "think"

    @pytest.mark.asyncio
    async def test_think_command_long_thought_truncation(self, mock_worker):
        """Test ThinkCommand truncates long thoughts in message."""
        long_thought = "a" * 100
        thought_data = json.dumps({
            "content": long_thought,
            "source": "manual",
            "timestamp": 1234567890
        })
        mock_worker.redis.get.return_value = thought_data.encode("utf-8")

        cmd = ThinkCommand()
        result = await cmd.execute(mock_worker, turn_id="test_turn", metadata={})

        # Verify thought was injected (full)
        assert mock_worker._response_strategy.thought_content == long_thought

        # Verify message contains truncated version
        assert "..." in result.message
        assert len(result.message) < len(long_thought) + 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
