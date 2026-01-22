# packages/aim-mud/tests/mud_tests/unit/worker/test_think_command.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for ThinkCommand."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from aim_mud_types import TurnRequestStatus, TurnReason
from andimud_worker.commands.think import ThinkCommand


@pytest.fixture
def mock_worker(mocker):
    """Mock worker with minimal required attributes."""
    worker = MagicMock()
    worker.config = MagicMock()
    worker.config.agent_id = "test_agent"
    worker.pending_events = []
    # Setup async methods that ThinkCommand calls
    worker._setup_turn_context = AsyncMock()

    # Mock ThinkingTurnProcessor to avoid LLM calls
    mock_processor = AsyncMock()
    mock_processor.execute = AsyncMock()
    mocker.patch(
        "andimud_worker.commands.think.ThinkingTurnProcessor",
        return_value=mock_processor
    )

    return worker


@pytest.fixture
def base_kwargs():
    """Base kwargs with all required fields for MUDTurnRequest validation."""
    return {
        "turn_id": "test_turn",
        "sequence_id": 1000,
        "reason": TurnReason.THINK,
        "status": TurnRequestStatus.IN_PROGRESS,
        "metadata": {},
    }


class TestThinkCommand:
    """Tests for ThinkCommand execution."""

    @pytest.mark.asyncio
    async def test_think_command_returns_complete(self, mock_worker, base_kwargs):
        """Test ThinkCommand processes turn and returns complete=True."""
        cmd = ThinkCommand()
        result = await cmd.execute(mock_worker, **base_kwargs)

        # Verify command completed processing
        assert result.complete is True
        assert result.status == TurnRequestStatus.DONE
        assert "Think turn processed" in result.message
        # Verify setup was called
        mock_worker._setup_turn_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_think_command_with_guidance(self, mock_worker, base_kwargs):
        """Test ThinkCommand passes guidance to processor."""
        base_kwargs["metadata"] = {"guidance": "Focus on emotional memories"}
        cmd = ThinkCommand()
        result = await cmd.execute(mock_worker, **base_kwargs)

        # Verify guidance appears in message
        assert result.complete is True
        assert "Focus on emotional memories" in result.message
        assert result.status == TurnRequestStatus.DONE

    @pytest.mark.asyncio
    async def test_think_command_truncates_long_guidance_in_message(self, mock_worker, base_kwargs):
        """Test ThinkCommand truncates long guidance in message."""
        long_guidance = "a" * 100
        base_kwargs["metadata"] = {"guidance": long_guidance}
        cmd = ThinkCommand()
        result = await cmd.execute(mock_worker, **base_kwargs)

        # Verify command completed
        assert result.complete is True

        # Verify message is truncated (50 char limit + "...")
        assert "..." in result.message
        assert "aaa" in result.message  # Some of the guidance should be present

    @pytest.mark.asyncio
    async def test_think_command_name_property(self):
        """Test ThinkCommand has correct name."""
        cmd = ThinkCommand()
        assert cmd.name == "think"

    @pytest.mark.asyncio
    async def test_think_command_without_metadata(self, mock_worker, base_kwargs):
        """Test ThinkCommand handles missing metadata."""
        base_kwargs["metadata"] = None
        cmd = ThinkCommand()
        result = await cmd.execute(mock_worker, **base_kwargs)

        # Should not crash and should complete
        assert result.complete is True
        assert result.status == TurnRequestStatus.DONE
        assert "Think turn processed" in result.message

    @pytest.mark.asyncio
    async def test_think_command_empty_guidance(self, mock_worker, base_kwargs):
        """Test ThinkCommand handles empty guidance."""
        base_kwargs["metadata"] = {"guidance": ""}
        cmd = ThinkCommand()
        result = await cmd.execute(mock_worker, **base_kwargs)

        # Empty guidance should be treated as no guidance
        assert result.complete is True
        assert "Think turn processed" in result.message
        # Message should not include truncated guidance
        assert "..." not in result.message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
