# packages/aim-mud/tests/mud_tests/unit/worker/test_thinking_processor.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for ThinkingTurnProcessor."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from aim_mud_types import MUDTurnRequest, MUDEvent, TurnReason, TurnRequestStatus
from andimud_worker.turns.processor.thinking import ThinkingTurnProcessor


@pytest.fixture
def mock_worker():
    """Mock worker with minimal required attributes."""
    worker = MagicMock()
    worker._response_strategy = MagicMock()
    worker._response_strategy.thought_content = None
    worker._check_abort_requested = AsyncMock(return_value=False)
    return worker


@pytest.fixture
def sample_turn_request():
    """Sample turn request for testing."""
    return MUDTurnRequest(
        turn_id="test_turn_123",
        reason=TurnReason.THINK,
        status=TurnRequestStatus.IN_PROGRESS,
        sequence_id=1000,
    )


@pytest.fixture
def sample_events():
    """Sample events for testing."""
    return []


class TestThinkingTurnProcessorInit:
    """Tests for ThinkingTurnProcessor initialization."""

    def test_init_with_thought_content(self, mock_worker):
        """Test ThinkingTurnProcessor accepts thought_content."""
        processor = ThinkingTurnProcessor(mock_worker, thought_content="Test thought")

        assert processor.thought_content == "Test thought"
        assert processor.user_guidance == ""

    def test_init_without_thought_content(self, mock_worker):
        """Test ThinkingTurnProcessor with no thought."""
        processor = ThinkingTurnProcessor(mock_worker)

        assert processor.thought_content == ""
        assert processor.user_guidance == ""

    def test_init_defaults_to_empty_string(self, mock_worker):
        """Test ThinkingTurnProcessor defaults thought_content to empty string."""
        processor = ThinkingTurnProcessor(mock_worker)

        assert isinstance(processor.thought_content, str)
        assert processor.thought_content == ""


class TestThinkingTurnProcessorDecideAction:
    """Tests for ThinkingTurnProcessor._decide_action method."""

    @pytest.mark.asyncio
    async def test_decide_action_injects_thought_into_guidance(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test _decide_action injects thought into user_guidance."""
        processor = ThinkingTurnProcessor(mock_worker, thought_content="Focus on memories")

        # Mock the parent _decide_action
        with patch(
            "andimud_worker.turns.processor.phased.PhasedTurnProcessor._decide_action",
            new_callable=AsyncMock,
            return_value=([], "thinking text")
        ) as mock_parent:
            await processor._decide_action(sample_turn_request, sample_events)

            # Verify thought was injected into guidance
            assert "[Thought injection: Focus on memories]" in processor.user_guidance

    @pytest.mark.asyncio
    async def test_decide_action_sets_thought_on_strategy(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test _decide_action sets thought_content on response strategy."""
        processor = ThinkingTurnProcessor(mock_worker, thought_content="Test thought")

        # Mock the parent _decide_action
        with patch(
            "andimud_worker.turns.processor.phased.PhasedTurnProcessor._decide_action",
            new_callable=AsyncMock,
            return_value=([], "thinking text")
        ):
            await processor._decide_action(sample_turn_request, sample_events)

            # Verify thought was set on strategy
            assert mock_worker._response_strategy.thought_content == "Test thought"

    @pytest.mark.asyncio
    async def test_decide_action_with_existing_guidance(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test _decide_action prepends thought to existing guidance."""
        processor = ThinkingTurnProcessor(mock_worker, thought_content="New thought")
        processor.user_guidance = "Existing guidance"

        # Mock the parent _decide_action
        with patch(
            "andimud_worker.turns.processor.phased.PhasedTurnProcessor._decide_action",
            new_callable=AsyncMock,
            return_value=([], "thinking text")
        ):
            await processor._decide_action(sample_turn_request, sample_events)

            # Verify thought was prepended
            assert processor.user_guidance.startswith("[Thought injection: New thought]")
            assert "Existing guidance" in processor.user_guidance

    @pytest.mark.asyncio
    async def test_decide_action_without_thought(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test _decide_action with no thought content."""
        processor = ThinkingTurnProcessor(mock_worker)

        # Mock the parent _decide_action
        with patch(
            "andimud_worker.turns.processor.phased.PhasedTurnProcessor._decide_action",
            new_callable=AsyncMock,
            return_value=([], "thinking text")
        ):
            await processor._decide_action(sample_turn_request, sample_events)

            # Verify no thought was injected
            assert processor.user_guidance == ""
            assert mock_worker._response_strategy.thought_content is None

    @pytest.mark.asyncio
    async def test_decide_action_no_response_strategy(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test _decide_action handles missing response strategy gracefully."""
        mock_worker._response_strategy = None
        processor = ThinkingTurnProcessor(mock_worker, thought_content="Test thought")

        # Mock the parent _decide_action
        with patch(
            "andimud_worker.turns.processor.phased.PhasedTurnProcessor._decide_action",
            new_callable=AsyncMock,
            return_value=([], "thinking text")
        ):
            # Should not crash
            await processor._decide_action(sample_turn_request, sample_events)

            # Verify thought was still injected into guidance
            assert "[Thought injection: Test thought]" in processor.user_guidance

    @pytest.mark.asyncio
    async def test_decide_action_calls_parent(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test _decide_action calls parent implementation."""
        processor = ThinkingTurnProcessor(mock_worker, thought_content="Test thought")

        # Mock the parent _decide_action
        with patch(
            "andimud_worker.turns.processor.phased.PhasedTurnProcessor._decide_action",
            new_callable=AsyncMock,
            return_value=(["action1"], "thinking text")
        ) as mock_parent:
            result = await processor._decide_action(sample_turn_request, sample_events)

            # Verify parent was called
            mock_parent.assert_called_once_with(sample_turn_request, sample_events)

            # Verify result is returned from parent
            assert result == (["action1"], "thinking text")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
