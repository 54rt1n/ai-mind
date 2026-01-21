# packages/aim-mud/tests/mud_tests/unit/worker/test_thinking_processor.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for ThinkingTurnProcessor."""

import json
import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch

from aim_mud_types import MUDTurnRequest, MUDEvent, MUDAction, TurnReason, TurnRequestStatus
from andimud_worker.turns.processor.thinking import (
    ThinkingTurnProcessor,
    THOUGHT_TTL_SECONDS,
    REASONING_PROMPT,
)


@pytest.fixture
def mock_worker():
    """Mock worker with minimal required attributes."""
    worker = MagicMock()
    worker.config = MagicMock()
    worker.config.agent_id = "test_agent"
    worker.redis = AsyncMock()
    worker.persona = MagicMock()
    worker.session = MagicMock()
    worker.model = MagicMock()
    worker.model.max_tokens = 128000
    worker.chat_config = MagicMock()
    worker.chat_config.max_tokens = 4096
    worker._response_strategy = MagicMock()
    worker._response_strategy.thought_content = None
    worker._response_strategy.build_turns = AsyncMock(return_value=[
        {"role": "user", "content": "test context"}
    ])
    worker._is_fresh_session = AsyncMock(return_value=False)
    worker._call_llm = AsyncMock(return_value="<reasoning>\n    <inspiration>Test observation.</inspiration>\n</reasoning>")
    worker._emit_actions = AsyncMock()
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

    def test_init_creates_processor(self, mock_worker):
        """Test ThinkingTurnProcessor initializes correctly."""
        processor = ThinkingTurnProcessor(mock_worker)

        assert processor.worker == mock_worker
        assert processor.user_guidance == ""


class TestThinkingTurnProcessorDecideAction:
    """Tests for ThinkingTurnProcessor._decide_action method."""

    @pytest.mark.asyncio
    async def test_decide_action_generates_reasoning(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test _decide_action generates reasoning via LLM call."""
        # No previous thought
        mock_worker.redis.get.return_value = None

        processor = ThinkingTurnProcessor(mock_worker)
        actions, thinking = await processor._decide_action(sample_turn_request, sample_events)

        # Verify LLM was called
        mock_worker._call_llm.assert_called_once()

        # Verify emote action was emitted
        assert len(actions) == 1
        assert actions[0].tool == "emote"
        assert actions[0].args["action"] == "pauses thoughtfully."

        # Verify reasoning was extracted and included in thinking
        assert "<reasoning>" in thinking

    @pytest.mark.asyncio
    async def test_decide_action_stores_reasoning_to_redis(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test _decide_action stores generated reasoning to Redis."""
        mock_worker.redis.get.return_value = None

        processor = ThinkingTurnProcessor(mock_worker)
        await processor._decide_action(sample_turn_request, sample_events)

        # Verify Redis set was called with correct key and TTL
        mock_worker.redis.set.assert_called_once()
        call_args = mock_worker.redis.set.call_args
        key = call_args[0][0]
        data = json.loads(call_args[0][1])

        assert key == "agent:test_agent:thought"
        assert data["source"] == "reasoning"
        assert "content" in data
        assert "timestamp" in data
        assert call_args[1]["ex"] == THOUGHT_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_decide_action_loads_previous_thought_within_ttl(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test _decide_action loads previous thought if within TTL."""
        # Setup previous thought within TTL
        previous_thought = {
            "content": "<reasoning>Previous reasoning</reasoning>",
            "source": "reasoning",
            "timestamp": int(time.time()) - 100,  # 100 seconds ago
        }
        mock_worker.redis.get.return_value = json.dumps(previous_thought).encode("utf-8")

        processor = ThinkingTurnProcessor(mock_worker)
        await processor._decide_action(sample_turn_request, sample_events)

        # Verify previous thought was set on response strategy
        assert mock_worker._response_strategy.thought_content == "<reasoning>Previous reasoning</reasoning>"

    @pytest.mark.asyncio
    async def test_decide_action_ignores_expired_thought(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test _decide_action ignores thought outside TTL."""
        # Setup expired thought
        expired_thought = {
            "content": "<reasoning>Expired reasoning</reasoning>",
            "source": "reasoning",
            "timestamp": int(time.time()) - THOUGHT_TTL_SECONDS - 100,  # Expired
        }
        mock_worker.redis.get.return_value = json.dumps(expired_thought).encode("utf-8")

        processor = ThinkingTurnProcessor(mock_worker)
        await processor._decide_action(sample_turn_request, sample_events)

        # Verify expired thought was NOT set on response strategy
        assert mock_worker._response_strategy.thought_content is None

    @pytest.mark.asyncio
    async def test_decide_action_includes_user_guidance(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test _decide_action includes user guidance in prompt."""
        mock_worker.redis.get.return_value = None

        processor = ThinkingTurnProcessor(mock_worker)
        processor.user_guidance = "Focus on recent memories"
        await processor._decide_action(sample_turn_request, sample_events)

        # Verify build_turns was called with guidance in user_input
        call_args = mock_worker._response_strategy.build_turns.call_args
        user_input = call_args[1]["user_input"]
        assert "[User Guidance: Focus on recent memories]" in user_input
        assert REASONING_PROMPT in user_input

    @pytest.mark.asyncio
    async def test_decide_action_handles_no_reasoning_block(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test _decide_action handles LLM response without reasoning block."""
        mock_worker.redis.get.return_value = None
        mock_worker._call_llm.return_value = "Just some text without reasoning block."

        processor = ThinkingTurnProcessor(mock_worker)
        actions, thinking = await processor._decide_action(sample_turn_request, sample_events)

        # Verify emote was still emitted
        assert len(actions) == 1
        assert actions[0].tool == "emote"

        # Verify empty reasoning was stored
        call_args = mock_worker.redis.set.call_args
        data = json.loads(call_args[0][1])
        assert data["content"] == ""

    @pytest.mark.asyncio
    async def test_decide_action_extracts_think_tags_first(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test _decide_action extracts think tags before reasoning block."""
        mock_worker.redis.get.return_value = None
        mock_worker._call_llm.return_value = (
            "<think>Internal thinking</think>"
            "<reasoning>\n    <inspiration>Test</inspiration>\n</reasoning>"
        )

        processor = ThinkingTurnProcessor(mock_worker)
        actions, thinking = await processor._decide_action(sample_turn_request, sample_events)

        # Verify both think content and reasoning are in thinking output
        assert "Internal thinking" in thinking
        assert "<reasoning>" in thinking

    @pytest.mark.asyncio
    async def test_decide_action_handles_invalid_previous_thought_json(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test _decide_action handles invalid JSON in previous thought."""
        mock_worker.redis.get.return_value = b"invalid json {{{"

        processor = ThinkingTurnProcessor(mock_worker)
        # Should not raise
        actions, thinking = await processor._decide_action(sample_turn_request, sample_events)

        # Verify processing continued
        assert len(actions) == 1
        mock_worker._call_llm.assert_called_once()


class TestThinkingTurnProcessorHelpers:
    """Tests for ThinkingTurnProcessor helper methods."""

    @pytest.mark.asyncio
    async def test_store_thought_sets_correct_data(self, mock_worker):
        """Test _store_thought stores correct JSON structure."""
        processor = ThinkingTurnProcessor(mock_worker)

        with patch("time.time", return_value=1234567890):
            await processor._store_thought("<reasoning>Test</reasoning>")

        call_args = mock_worker.redis.set.call_args
        key = call_args[0][0]
        data = json.loads(call_args[0][1])

        assert key == "agent:test_agent:thought"
        assert data["content"] == "<reasoning>Test</reasoning>"
        assert data["source"] == "reasoning"
        assert data["timestamp"] == 1234567890
        assert call_args[1]["ex"] == THOUGHT_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_load_previous_thought_handles_no_thought(self, mock_worker):
        """Test _load_previous_thought handles missing thought."""
        mock_worker.redis.get.return_value = None

        processor = ThinkingTurnProcessor(mock_worker)
        await processor._load_previous_thought()

        # Verify thought was not set
        assert mock_worker._response_strategy.thought_content is None

    @pytest.mark.asyncio
    async def test_decide_action_with_extremely_large_reasoning(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test _decide_action handles very large reasoning blocks."""
        # Generate large reasoning block (10KB)
        large_reasoning = "<reasoning>\n"
        for i in range(300):
            large_reasoning += f"    <inspiration>Observation number {i} with lots of detail.</inspiration>\n"
        large_reasoning += "</reasoning>"

        mock_worker.redis.get.return_value = None
        mock_worker._call_llm.return_value = large_reasoning

        processor = ThinkingTurnProcessor(mock_worker)
        actions, thinking = await processor._decide_action(sample_turn_request, sample_events)

        # Verify reasoning was stored
        call_args = mock_worker.redis.set.call_args
        data = json.loads(call_args[0][1])
        assert len(data["content"]) > 5000  # Should be large
        assert data["source"] == "reasoning"

    @pytest.mark.asyncio
    async def test_decide_action_empty_guidance_not_included(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test _decide_action does not include empty user guidance."""
        mock_worker.redis.get.return_value = None

        processor = ThinkingTurnProcessor(mock_worker)
        processor.user_guidance = ""  # Empty string
        await processor._decide_action(sample_turn_request, sample_events)

        # Verify build_turns was called WITHOUT guidance prefix
        call_args = mock_worker._response_strategy.build_turns.call_args
        user_input = call_args[1]["user_input"]
        assert "[User Guidance:" not in user_input
        assert REASONING_PROMPT in user_input


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
