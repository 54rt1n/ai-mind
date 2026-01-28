# packages/aim-mud/tests/mud_tests/unit/worker/test_focus_commands.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for FocusCommand and ClearFocusCommand."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from aim_mud_types import TurnRequestStatus, TurnReason
from andimud_worker.commands.focus import FocusCommand, ClearFocusCommand


@pytest.fixture
def mock_worker():
    """Mock worker with code strategies that have set_focus/clear_focus."""
    worker = MagicMock()
    worker.config = MagicMock()
    worker.config.agent_id = "test_agent"

    # Mock Redis client with async methods for profile persistence
    worker.redis = MagicMock()
    worker.redis.hset = AsyncMock(return_value=1)
    worker.redis.eval = AsyncMock(return_value=1)
    worker.redis.hdel = AsyncMock(return_value=1)

    # Mock decision strategy with focus methods
    worker._decision_strategy = MagicMock()
    worker._decision_strategy.set_focus = MagicMock()
    worker._decision_strategy.clear_focus = MagicMock()

    # Mock response strategy with focus methods
    worker._response_strategy = MagicMock()
    worker._response_strategy.set_focus = MagicMock()
    worker._response_strategy.clear_focus = MagicMock()

    return worker


@pytest.fixture
def mock_worker_no_focus_methods():
    """Mock worker with strategies that lack focus methods."""
    worker = MagicMock()
    worker.config = MagicMock()
    worker.config.agent_id = "test_agent"

    # Mock Redis client with async methods for profile persistence
    worker.redis = MagicMock()
    worker.redis.hset = AsyncMock(return_value=1)
    worker.redis.eval = AsyncMock(return_value=1)
    worker.redis.hdel = AsyncMock(return_value=1)

    # Strategies exist but don't have focus methods
    worker._decision_strategy = MagicMock(spec=[])  # Empty spec means no attributes
    worker._response_strategy = MagicMock(spec=[])

    return worker


@pytest.fixture
def base_kwargs():
    """Base kwargs with all required fields for MUDTurnRequest validation."""
    return {
        "turn_id": "test_turn",
        "sequence_id": 1000,
        "reason": TurnReason.FOCUS,
        "status": TurnRequestStatus.IN_PROGRESS,
        "metadata": {},
    }


class TestFocusCommand:
    """Tests for FocusCommand execution."""

    @pytest.mark.asyncio
    async def test_focus_command_sets_focus_on_strategies(self, mock_worker, base_kwargs):
        """Test FocusCommand sets focus on both strategies."""
        # New format: files is list[dict] with path, start, end
        base_kwargs["metadata"] = {
            "files": [
                {"path": "model.py", "start": 10, "end": 50},
                {"path": "utils.py", "start": 100, "end": 200},
            ],
            "height": 2,
            "depth": 1,
        }
        cmd = FocusCommand()
        result = await cmd.execute(mock_worker, **base_kwargs)

        # Verify command completed
        assert result.complete is True
        assert result.status == TurnRequestStatus.DONE

        # Verify set_focus was called on both strategies
        mock_worker._decision_strategy.set_focus.assert_called_once()
        mock_worker._response_strategy.set_focus.assert_called_once()

        # Verify the FocusRequest has correct values
        focus_arg = mock_worker._decision_strategy.set_focus.call_args[0][0]
        assert focus_arg.get_file_paths() == ["model.py", "utils.py"]
        assert focus_arg.get_line_range("model.py") == (10, 50)
        assert focus_arg.get_line_range("utils.py") == (100, 200)
        assert focus_arg.height == 2
        assert focus_arg.depth == 1

    @pytest.mark.asyncio
    async def test_focus_command_without_metadata(self, mock_worker, base_kwargs):
        """Test FocusCommand handles missing metadata gracefully."""
        base_kwargs["metadata"] = None
        cmd = FocusCommand()
        result = await cmd.execute(mock_worker, **base_kwargs)

        # Should complete without setting focus
        assert result.complete is True
        mock_worker._decision_strategy.set_focus.assert_not_called()
        mock_worker._response_strategy.set_focus.assert_not_called()

    @pytest.mark.asyncio
    async def test_focus_command_with_empty_files(self, mock_worker, base_kwargs):
        """Test FocusCommand handles empty files list gracefully."""
        base_kwargs["metadata"] = {"files": []}
        cmd = FocusCommand()
        result = await cmd.execute(mock_worker, **base_kwargs)

        # Should complete without setting focus
        assert result.complete is True
        mock_worker._decision_strategy.set_focus.assert_not_called()
        mock_worker._response_strategy.set_focus.assert_not_called()

    @pytest.mark.asyncio
    async def test_focus_command_with_minimal_metadata(self, mock_worker, base_kwargs):
        """Test FocusCommand with only files specified (no line ranges)."""
        # New format: files is list[dict] with just path
        base_kwargs["metadata"] = {"files": [{"path": "model.py"}]}
        cmd = FocusCommand()
        result = await cmd.execute(mock_worker, **base_kwargs)

        # Verify command completed
        assert result.complete is True

        # Verify set_focus was called
        mock_worker._decision_strategy.set_focus.assert_called_once()

        # Verify defaults for optional fields
        focus_arg = mock_worker._decision_strategy.set_focus.call_args[0][0]
        assert focus_arg.get_file_paths() == ["model.py"]
        assert focus_arg.get_line_range("model.py") == (None, None)
        assert focus_arg.height == 1
        assert focus_arg.depth == 1

    @pytest.mark.asyncio
    async def test_focus_command_with_strategy_lacking_method(
        self, mock_worker_no_focus_methods, base_kwargs
    ):
        """Test FocusCommand handles strategies without set_focus method."""
        base_kwargs["metadata"] = {"files": ["model.py"]}
        cmd = FocusCommand()
        result = await cmd.execute(mock_worker_no_focus_methods, **base_kwargs)

        # Should complete without error
        assert result.complete is True

    @pytest.mark.asyncio
    async def test_focus_command_name_property(self):
        """Test FocusCommand has correct name."""
        cmd = FocusCommand()
        assert cmd.name == "focus"


class TestClearFocusCommand:
    """Tests for ClearFocusCommand execution."""

    @pytest.mark.asyncio
    async def test_clear_focus_command_clears_focus_on_strategies(
        self, mock_worker, base_kwargs
    ):
        """Test ClearFocusCommand clears focus on both strategies."""
        base_kwargs["reason"] = TurnReason.CLEAR_FOCUS
        cmd = ClearFocusCommand()
        result = await cmd.execute(mock_worker, **base_kwargs)

        # Verify command completed
        assert result.complete is True
        assert result.status == TurnRequestStatus.DONE

        # Verify clear_focus was called on both strategies
        mock_worker._decision_strategy.clear_focus.assert_called_once()
        mock_worker._response_strategy.clear_focus.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_focus_command_with_strategy_lacking_method(
        self, mock_worker_no_focus_methods, base_kwargs
    ):
        """Test ClearFocusCommand handles strategies without clear_focus method."""
        base_kwargs["reason"] = TurnReason.CLEAR_FOCUS
        cmd = ClearFocusCommand()
        result = await cmd.execute(mock_worker_no_focus_methods, **base_kwargs)

        # Should complete without error
        assert result.complete is True

    @pytest.mark.asyncio
    async def test_clear_focus_command_name_property(self):
        """Test ClearFocusCommand has correct name."""
        cmd = ClearFocusCommand()
        assert cmd.name == "clear_focus"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
