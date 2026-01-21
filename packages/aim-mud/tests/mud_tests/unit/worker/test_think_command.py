# packages/aim-mud/tests/mud_tests/unit/worker/test_think_command.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for ThinkCommand."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from aim_mud_types import TurnRequestStatus, TurnReason
from andimud_worker.commands.think import ThinkCommand


@pytest.fixture
def mock_worker():
    """Mock worker with minimal required attributes."""
    worker = MagicMock()
    worker.config = MagicMock()
    worker.config.agent_id = "test_agent"
    worker.pending_events = []
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
    async def test_think_command_returns_incomplete(self, mock_worker, base_kwargs):
        """Test ThinkCommand returns complete=False to fall through to processor."""
        cmd = ThinkCommand()
        result = await cmd.execute(mock_worker, **base_kwargs)

        # Verify result falls through
        assert result.complete is False
        assert result.status == TurnRequestStatus.DONE
        assert "Think turn ready" in result.message

    @pytest.mark.asyncio
    async def test_think_command_preserves_guidance(self, mock_worker, base_kwargs):
        """Test ThinkCommand preserves guidance from metadata."""
        base_kwargs["metadata"] = {"guidance": "Focus on emotional memories"}
        cmd = ThinkCommand()
        result = await cmd.execute(mock_worker, **base_kwargs)

        # Verify guidance is preserved
        assert result.plan_guidance == "Focus on emotional memories"
        assert "Focus on emotional memories" in result.message

    @pytest.mark.asyncio
    async def test_think_command_truncates_long_guidance_in_message(self, mock_worker, base_kwargs):
        """Test ThinkCommand truncates long guidance in message."""
        long_guidance = "a" * 100
        base_kwargs["metadata"] = {"guidance": long_guidance}
        cmd = ThinkCommand()
        result = await cmd.execute(mock_worker, **base_kwargs)

        # Verify guidance is preserved in full
        assert result.plan_guidance == long_guidance

        # Verify message is truncated
        assert "..." in result.message

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

        # Should not crash
        assert result.complete is False
        assert result.plan_guidance is None

    @pytest.mark.asyncio
    async def test_think_command_empty_guidance(self, mock_worker, base_kwargs):
        """Test ThinkCommand handles empty guidance."""
        base_kwargs["metadata"] = {"guidance": ""}
        cmd = ThinkCommand()
        result = await cmd.execute(mock_worker, **base_kwargs)

        assert result.plan_guidance is None
        assert "will generate reasoning" in result.message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
