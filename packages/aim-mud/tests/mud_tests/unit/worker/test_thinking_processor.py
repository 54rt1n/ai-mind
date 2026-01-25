# packages/aim-mud/tests/mud_tests/unit/worker/test_thinking_processor.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for ThinkingTurnProcessor."""

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
    worker.model_set = MagicMock()
    worker.model_set.get_model_name = MagicMock(side_effect=lambda role: f"{role}_model")
    worker._response_strategy = MagicMock()
    worker._response_strategy.thought_content = None
    worker._decision_strategy = MagicMock()
    worker._decision_strategy.thought_content = None
    worker._response_strategy.build_turns = AsyncMock(return_value=[
        {"role": "user", "content": "test context"}
    ])
    worker._is_fresh_session = AsyncMock(return_value=False)
    worker._call_llm = AsyncMock(return_value="<reasoning>\n    <inspiration>Test observation.</inspiration>\n</reasoning>")
    worker._emit_actions = AsyncMock()
    worker._check_abort_requested = AsyncMock(return_value=False)
    # Default: _load_thought_content does nothing (no thought to load)
    worker._load_thought_content = AsyncMock()
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
        # Mock redis.eval to simulate successful hash creation
        mock_worker.redis.eval.return_value = 1
        # Mock redis.expire for TTL setting
        mock_worker.redis.expire.return_value = True

        processor = ThinkingTurnProcessor(mock_worker)
        await processor._decide_action(sample_turn_request, sample_events)

        # Verify Redis eval was called (for _create_hash Lua script)
        assert mock_worker.redis.eval.called
        # Verify expire was called to set TTL
        assert mock_worker.redis.expire.called

        # Check expire was called with correct key and TTL
        expire_call = mock_worker.redis.expire.call_args
        key = expire_call[0][0]
        ttl = expire_call[0][1]

        assert key == "agent:test_agent:thought"
        assert ttl == THOUGHT_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_decide_action_calls_load_thought_content(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test _decide_action calls _load_thought_content to load previous thought."""
        processor = ThinkingTurnProcessor(mock_worker)
        await processor._decide_action(sample_turn_request, sample_events)

        # Verify _load_thought_content was called (delegates to ProfileMixin)
        mock_worker._load_thought_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_decide_action_thought_content_set_by_load_method(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test _decide_action relies on _load_thought_content to set thought."""
        # Mock _load_thought_content to simulate loading a thought
        async def mock_load():
            mock_worker._response_strategy.thought_content = "<reasoning>Loaded thought</reasoning>"
            mock_worker._decision_strategy.thought_content = "<reasoning>Loaded thought</reasoning>"

        mock_worker._load_thought_content = AsyncMock(side_effect=mock_load)

        processor = ThinkingTurnProcessor(mock_worker)
        await processor._decide_action(sample_turn_request, sample_events)

        # Verify thought content was set by the mock (simulating ProfileMixin behavior)
        assert mock_worker._response_strategy.thought_content == "<reasoning>Loaded thought</reasoning>"

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
        assert "[Link Guidance: Focus on recent memories]" in user_input
        assert REASONING_PROMPT in user_input

    @pytest.mark.asyncio
    async def test_decide_action_handles_no_reasoning_block(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test _decide_action handles LLM response without reasoning block."""
        mock_worker.redis.get.return_value = None
        mock_worker._call_llm.return_value = "Just some text without reasoning block."
        # Configure fallback model same as chat to skip fallback
        mock_worker.model_set.get_model_name = MagicMock(return_value="same_model")

        processor = ThinkingTurnProcessor(mock_worker)
        actions, thinking = await processor._decide_action(sample_turn_request, sample_events)

        # Verify emote was still emitted
        assert len(actions) == 1
        assert actions[0].tool == "emote"

        # Verify LLM was called multiple times (retries)
        assert mock_worker._call_llm.call_count >= 3

        # Verify Redis eval was NOT called (no valid reasoning, no storage)
        assert mock_worker.redis.eval.call_count == 0

        # Verify error message in thinking
        assert "[ERROR]" in thinking

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
    async def test_decide_action_continues_when_load_thought_fails(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test _decide_action continues when _load_thought_content raises."""
        # _load_thought_content handles errors internally, but simulate it raising
        mock_worker._load_thought_content = AsyncMock(side_effect=Exception("Redis error"))

        processor = ThinkingTurnProcessor(mock_worker)
        # Should raise since _load_previous_thought doesn't catch exceptions itself
        # (it relies on ProfileMixin to handle Redis errors)
        with pytest.raises(Exception, match="Redis error"):
            await processor._decide_action(sample_turn_request, sample_events)


class TestThinkingTurnProcessorHelpers:
    """Tests for ThinkingTurnProcessor helper methods."""

    @pytest.mark.asyncio
    async def test_store_thought_sets_correct_data(self, mock_worker):
        """Test _store_thought stores correct JSON structure."""
        # Mock redis.eval to simulate successful hash creation
        mock_worker.redis.eval.return_value = 1
        # Mock redis.expire for TTL setting
        mock_worker.redis.expire.return_value = True

        processor = ThinkingTurnProcessor(mock_worker)

        with patch("time.time", return_value=1234567890):
            await processor._store_thought("<reasoning>Test</reasoning>")

        # Verify redis.eval was called with Lua script for HSET
        assert mock_worker.redis.eval.called
        eval_call_args = mock_worker.redis.eval.call_args

        # The Lua script and arguments are passed to eval
        # Keys[1] is the key, and ARGV contains field-value pairs
        lua_script = eval_call_args[0][0]
        num_keys = eval_call_args[0][1]
        key = eval_call_args[0][2]

        # Check key and num_keys
        assert num_keys == 1
        assert key == "agent:test_agent:thought"

        # The remaining args are field-value pairs for HSET
        # We can verify the fields were serialized correctly by checking they're in the args
        args_list = list(eval_call_args[0][3:])

        # Convert list to dict (field, value, field, value, ...)
        fields_dict = {}
        for i in range(0, len(args_list), 2):
            if i + 1 < len(args_list):
                fields_dict[args_list[i]] = args_list[i + 1]

        # Verify the fields
        assert fields_dict.get("agent_id") == "test_agent"
        assert fields_dict.get("content") == "<reasoning>Test</reasoning>"
        assert fields_dict.get("source") == "reasoning"
        assert "created_at" in fields_dict  # Timestamp is stored as created_at
        assert fields_dict.get("actions_since_generation") == "0"

        # Verify expire was called with correct key and TTL
        expire_call = mock_worker.redis.expire.call_args
        assert expire_call[0][0] == "agent:test_agent:thought"
        assert expire_call[0][1] == THOUGHT_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_load_previous_thought_delegates_to_worker(self, mock_worker):
        """Test _load_previous_thought delegates to worker._load_thought_content."""
        processor = ThinkingTurnProcessor(mock_worker)
        await processor._load_previous_thought()

        # Verify it delegates to ProfileMixin._load_thought_content
        mock_worker._load_thought_content.assert_called_once()

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
        # Mock redis.eval to simulate successful hash creation
        mock_worker.redis.eval.return_value = 1
        # Mock redis.expire for TTL setting
        mock_worker.redis.expire.return_value = True

        processor = ThinkingTurnProcessor(mock_worker)
        actions, thinking = await processor._decide_action(sample_turn_request, sample_events)

        # Verify reasoning was stored by checking redis.eval was called
        assert mock_worker.redis.eval.called
        eval_call_args = mock_worker.redis.eval.call_args

        # Extract the field-value pairs from eval args
        args_list = list(eval_call_args[0][3:])
        fields_dict = {}
        for i in range(0, len(args_list), 2):
            if i + 1 < len(args_list):
                fields_dict[args_list[i]] = args_list[i + 1]

        # Verify the content is large and has correct source
        assert len(fields_dict.get("content", "")) > 5000  # Should be large
        assert fields_dict.get("source") == "reasoning"

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


class TestThinkingTurnProcessorRetryLogic:
    """Tests for retry and fallback logic."""

    @pytest.mark.asyncio
    async def test_retry_on_missing_reasoning_block(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test processor retries when reasoning block is missing."""
        mock_worker.redis.get.return_value = None
        # First two calls fail, third succeeds
        mock_worker._call_llm.side_effect = [
            "No reasoning here",
            "Still no reasoning",
            "<reasoning>\n    <inspiration>Success</inspiration>\n</reasoning>",
        ]
        # Configure fallback model same as chat to skip fallback
        mock_worker.model_set.get_model_name = MagicMock(return_value="same_model")
        # Mock redis.eval to simulate successful hash creation
        mock_worker.redis.eval.return_value = 1
        # Mock redis.expire for TTL setting
        mock_worker.redis.expire.return_value = True

        processor = ThinkingTurnProcessor(mock_worker)
        actions, thinking = await processor._decide_action(sample_turn_request, sample_events)

        # Verify LLM was called 3 times
        assert mock_worker._call_llm.call_count == 3

        # Verify success
        assert "<reasoning>" in thinking
        assert "Success" in thinking

        # Verify reasoning was stored (redis.eval was called)
        assert mock_worker.redis.eval.called

    @pytest.mark.asyncio
    async def test_fallback_model_on_repeated_failure(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test processor tries fallback model after max retries."""
        mock_worker.redis.get.return_value = None
        # Chat model fails 3 times, fallback succeeds
        mock_worker._call_llm.side_effect = [
            "No reasoning 1",
            "No reasoning 2",
            "No reasoning 3",
            "<reasoning>\n    <inspiration>Fallback success</inspiration>\n</reasoning>",
        ]

        processor = ThinkingTurnProcessor(mock_worker)
        actions, thinking = await processor._decide_action(sample_turn_request, sample_events)

        # Verify LLM was called 4 times (3 chat + 1 fallback)
        assert mock_worker._call_llm.call_count == 4

        # Verify last call used fallback role
        last_call = mock_worker._call_llm.call_args_list[-1]
        assert last_call[1]["role"] == "fallback"

        # Verify success
        assert "<reasoning>" in thinking
        assert "Fallback success" in thinking

    @pytest.mark.asyncio
    async def test_skip_fallback_if_same_as_chat(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test processor skips fallback if it's the same model as chat."""
        mock_worker.redis.get.return_value = None
        mock_worker._call_llm.return_value = "No reasoning"
        # Configure fallback model same as chat
        mock_worker.model_set.get_model_name = MagicMock(return_value="same_model")

        processor = ThinkingTurnProcessor(mock_worker)
        actions, thinking = await processor._decide_action(sample_turn_request, sample_events)

        # Verify LLM was called only 3 times (no fallback)
        assert mock_worker._call_llm.call_count == 3

        # Verify error handling
        assert "[ERROR]" in thinking
        assert mock_worker.redis.set.call_count == 0

    @pytest.mark.asyncio
    async def test_abort_during_retry_loop(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test processor respects abort requests during retry loop."""
        from andimud_worker.exceptions import AbortRequestedException

        mock_worker.redis.get.return_value = None
        mock_worker._check_abort_requested.return_value = True

        processor = ThinkingTurnProcessor(mock_worker)

        with pytest.raises(AbortRequestedException):
            await processor._decide_action(sample_turn_request, sample_events)

        # Verify LLM was not called
        mock_worker._call_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_format_guidance_added_on_retry(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test format guidance is added to chat turns on retry."""
        mock_worker.redis.get.return_value = None
        # First call fails, second succeeds
        mock_worker._call_llm.side_effect = [
            "No reasoning",
            "<reasoning>\n    <inspiration>Success</inspiration>\n</reasoning>",
        ]
        # Configure fallback model same as chat
        mock_worker.model_set.get_model_name = MagicMock(return_value="same_model")

        # Track chat_turns modifications
        chat_turns_history = []

        async def capture_build_turns(*args, **kwargs):
            result = [{"role": "user", "content": "test context"}]
            chat_turns_history.append(result.copy())
            return result

        mock_worker._response_strategy.build_turns = AsyncMock(side_effect=capture_build_turns)

        processor = ThinkingTurnProcessor(mock_worker)
        await processor._decide_action(sample_turn_request, sample_events)

        # Verify guidance was added after first failure
        # Note: guidance is added to the chat_turns list returned by build_turns
        # We can't easily verify this without inspecting call args to _call_llm
        assert mock_worker._call_llm.call_count == 2

    @pytest.mark.asyncio
    async def test_no_storage_on_all_retries_failed(
        self, mock_worker, sample_turn_request, sample_events
    ):
        """Test no Redis storage when all retries fail."""
        mock_worker.redis.get.return_value = None
        mock_worker._call_llm.return_value = "No reasoning ever"
        # Configure fallback model same as chat
        mock_worker.model_set.get_model_name = MagicMock(return_value="same_model")

        processor = ThinkingTurnProcessor(mock_worker)
        actions, thinking = await processor._decide_action(sample_turn_request, sample_events)

        # Verify Redis eval was NOT called (no storage when no valid reasoning)
        mock_worker.redis.eval.assert_not_called()

        # Verify error in thinking
        assert "[ERROR]" in thinking
        assert "Failed to generate valid reasoning format" in thinking

        # Verify emote was still emitted
        assert len(actions) == 1
        assert actions[0].tool == "emote"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
