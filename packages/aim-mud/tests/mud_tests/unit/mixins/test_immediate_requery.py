# tests/unit/mixins/test_immediate_requery.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for immediate requery overflow handling in TurnsMixin.

Tests the overflow detection and retry mechanism for CODE_RAG agents.
When focus tool selects too much code, the system should:
1. Detect overflow BEFORE sending to LLM
2. Clear focus and inject error message
3. Retry up to IMMEDIATE_REQUERY_MAX_RETRIES times
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from andimud_worker.exceptions import ContextOverflowError
from andimud_worker.mixins.turns import TurnsMixin


class MockToolResult:
    """Mock tool result for process_response."""

    def __init__(self, is_valid: bool = True, function_name: str = "speak"):
        self.is_valid = is_valid
        self.function_name = function_name
        self.arguments = {}
        self.error = None


class MockDecisionStrategy:
    """Mock decision strategy that can simulate code or non-code behavior."""

    def __init__(self, is_code_strategy: bool = True, tokens_per_turn: int = 1000):
        self.is_code_strategy = is_code_strategy
        self.tokens_per_turn = tokens_per_turn
        self.focus_cleared_count = 0
        self._cached_system_message = None

        # Setup tool_user mock
        self.tool_user = MagicMock()
        self.tool_user.process_response.return_value = MockToolResult()

    async def build_turns(self, **kwargs) -> list[dict]:
        """Build mock turns with configurable token size."""
        return [
            {"role": "user", "content": "x" * self.tokens_per_turn},
            {"role": "assistant", "content": "y" * 100},
        ]

    def get_system_message(self, persona) -> str:
        """Return a mock system message."""
        return "System message for testing" * 100

    def clear_focus(self) -> None:
        """Track focus clears for verification."""
        self.focus_cleared_count += 1

    def is_aura_tool(self, tool_name: str) -> bool:
        """Always return False for mock."""
        return False


class MockNonCodeDecisionStrategy:
    """Mock non-code decision strategy (no clear_focus method)."""

    def __init__(self, tokens_per_turn: int = 1000):
        self.tokens_per_turn = tokens_per_turn

        # Setup tool_user mock
        self.tool_user = MagicMock()
        self.tool_user.process_response.return_value = MockToolResult()

    async def build_turns(self, **kwargs) -> list[dict]:
        """Build mock turns."""
        return [
            {"role": "user", "content": "x" * self.tokens_per_turn},
        ]

    def get_system_message(self, persona) -> str:
        """Return a mock system message."""
        return "System message"

    def is_aura_tool(self, tool_name: str) -> bool:
        """Always return False for mock."""
        return False


class MockWorkerWithMixin(TurnsMixin):
    """Mock worker with TurnsMixin for testing overflow handling."""

    def __init__(
        self,
        decision_strategy,
        model_max_tokens: int = 32768,
        max_output_tokens: int = 4096,
    ):
        self._decision_strategy = decision_strategy
        self.persona = MagicMock()
        self.session = MagicMock()
        self.model = MagicMock()
        self.model.max_tokens = model_max_tokens
        self.chat_config = MagicMock()
        self.chat_config.max_tokens = max_output_tokens
        self.config = MagicMock()
        self.config.decision_max_retries = 3

    async def _call_llm(self, turns: list[dict], role: str = "tool") -> str:
        """Mock LLM call that returns a valid tool response."""
        return '{"speak": {}}'


class TestImmediateRequery:
    """Tests for immediate requery overflow handling."""

    def test_is_code_strategy_true(self):
        """Test _is_code_strategy returns True for strategies with clear_focus."""
        strategy = MockDecisionStrategy(is_code_strategy=True)
        worker = MockWorkerWithMixin(strategy)
        assert worker._is_code_strategy() is True

    def test_is_code_strategy_false(self):
        """Test _is_code_strategy returns False for strategies without clear_focus."""
        strategy = MockNonCodeDecisionStrategy()
        worker = MockWorkerWithMixin(strategy)
        assert worker._is_code_strategy() is False

    def test_measure_total_tokens(self):
        """Test _measure_total_tokens counts tokens correctly."""
        strategy = MockDecisionStrategy()
        worker = MockWorkerWithMixin(strategy)

        turns = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
        ]
        system_message = "You are a helpful assistant."

        total = worker._measure_total_tokens(turns, system_message)
        # Should be > 0 and reasonable
        assert total > 0
        assert total < 1000  # Short messages

    def test_get_model_context_limit_from_model(self):
        """Test _get_model_context_limit uses model.max_tokens."""
        strategy = MockDecisionStrategy()
        worker = MockWorkerWithMixin(strategy, model_max_tokens=65536)

        assert worker._get_model_context_limit() == 65536

    def test_get_model_context_limit_fallback(self):
        """Test _get_model_context_limit falls back when model unavailable."""
        strategy = MockDecisionStrategy()
        worker = MockWorkerWithMixin(strategy)
        worker.model = None

        assert worker._get_model_context_limit() == 32768  # Default fallback

    def test_format_overflow_error(self):
        """Test _format_overflow_error formats message correctly."""
        strategy = MockDecisionStrategy()
        worker = MockWorkerWithMixin(strategy)

        error = worker._format_overflow_error(
            total_tokens=40000,
            model_limit=32768,
            output_tokens=4096,
        )

        assert "[OVERFLOW ERROR]" in error
        assert "40000" in error  # input tokens
        assert "32768" in error  # model limit
        assert "4096" in error   # output tokens
        assert "Narrower line range" in error  # Guidance
        assert "Fewer files" in error  # Guidance


class TestImmediateRequeryIntegration:
    """Integration tests for the overflow retry loop."""

    @pytest.mark.asyncio
    async def test_no_overflow_direct_success(self):
        """Test that no overflow leads to direct LLM call without retries."""
        strategy = MockDecisionStrategy(tokens_per_turn=100)  # Small tokens
        worker = MockWorkerWithMixin(strategy, model_max_tokens=100000)

        result = await worker._decide_action(idle_mode=False)

        # Should succeed on first try with no focus clears
        assert strategy.focus_cleared_count == 0
        assert result[0] == "speak"  # Tool name from mock

    @pytest.mark.asyncio
    async def test_single_retry_success(self):
        """Test that overflow triggers retry after clearing focus."""
        # First call has large tokens, strategy will reduce on rebuild
        # Note: tiktoken encodes ~8 chars per token, so we need large content
        # For a 8192 token limit with 4096 output reserve, we have ~4096 usable
        # System message adds ~1500 tokens, so we need content > 2500 tokens
        call_count = 0

        async def mock_build_turns(self, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: ~5000 tokens of content to trigger overflow
                # Using varied content for better token efficiency
                large_content = " ".join(["word"] * 5000)  # ~5000 tokens
                return [{"role": "user", "content": large_content}]
            else:
                # After focus cleared: smaller content
                return [{"role": "user", "content": "small content"}]

        strategy = MockDecisionStrategy()
        strategy.build_turns = mock_build_turns.__get__(strategy, MockDecisionStrategy)
        # Use small model limit to trigger overflow easily
        worker = MockWorkerWithMixin(strategy, model_max_tokens=8192, max_output_tokens=4096)

        result = await worker._decide_action(idle_mode=False)

        # Focus should have been cleared once
        assert strategy.focus_cleared_count == 1
        assert result[0] == "speak"

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test that max retries exceeded raises ContextOverflowError."""
        # Always return large content that triggers overflow
        # Using varied content for better token counting

        async def mock_build_turns(self, **kwargs):
            # ~5000 tokens of content to always exceed limit
            large_content = " ".join(["word"] * 5000)
            return [{"role": "user", "content": large_content}]

        strategy = MockDecisionStrategy()
        strategy.build_turns = mock_build_turns.__get__(strategy, MockDecisionStrategy)
        # Use small model limit to trigger overflow
        worker = MockWorkerWithMixin(strategy, model_max_tokens=8192, max_output_tokens=4096)

        with pytest.raises(ContextOverflowError) as exc_info:
            await worker._decide_action(idle_mode=False)

        assert "Focus too large after" in str(exc_info.value)
        assert f"{TurnsMixin.IMMEDIATE_REQUERY_MAX_RETRIES}" in str(exc_info.value)
        # Focus cleared on each retry
        assert strategy.focus_cleared_count == TurnsMixin.IMMEDIATE_REQUERY_MAX_RETRIES

    @pytest.mark.asyncio
    async def test_non_code_strategy_skips_overflow_check(self):
        """Test that non-code strategies skip overflow handling entirely."""
        # Use large content but non-code strategy
        # Even with content that would trigger overflow for code strategy,
        # non-code strategy should proceed without checking
        async def mock_build_turns(self, **kwargs):
            large_content = " ".join(["word"] * 5000)
            return [{"role": "user", "content": large_content}]

        strategy = MockNonCodeDecisionStrategy()
        strategy.build_turns = mock_build_turns.__get__(strategy, MockNonCodeDecisionStrategy)
        # Small limit that would trigger overflow for code strategy
        worker = MockWorkerWithMixin(strategy, model_max_tokens=8192, max_output_tokens=4096)

        # Should proceed without overflow check (will fail at LLM call normally)
        # For this test we just verify it doesn't raise ContextOverflowError
        result = await worker._decide_action(idle_mode=False)
        assert result[0] == "speak"

    @pytest.mark.asyncio
    async def test_overflow_error_injected_into_turns(self):
        """Test that overflow error message is injected into turns on retry."""
        call_count = 0
        injected_error = None

        async def mock_build_turns(self, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: large content that triggers overflow
                large_content = " ".join(["word"] * 5000)
                return [{"role": "user", "content": large_content}]
            else:
                # After focus cleared: smaller content
                return [{"role": "user", "content": "small content"}]

        original_call_llm = MockWorkerWithMixin._call_llm

        async def capturing_call_llm(self, turns, role="tool"):
            nonlocal injected_error
            # Check if error was injected
            for turn in turns:
                if "[OVERFLOW ERROR]" in turn.get("content", ""):
                    injected_error = turn["content"]
            return await original_call_llm(self, turns, role)

        strategy = MockDecisionStrategy()
        strategy.build_turns = mock_build_turns.__get__(strategy, MockDecisionStrategy)
        # Small limit to trigger overflow
        worker = MockWorkerWithMixin(strategy, model_max_tokens=8192, max_output_tokens=4096)
        worker._call_llm = capturing_call_llm.__get__(worker, MockWorkerWithMixin)

        await worker._decide_action(idle_mode=False)

        # Verify error was injected
        assert injected_error is not None
        assert "[OVERFLOW ERROR]" in injected_error


class TestOverflowErrorMessage:
    """Tests for the overflow error message formatting."""

    def test_overflow_calculation(self):
        """Test that overflow is calculated correctly."""
        strategy = MockDecisionStrategy()
        worker = MockWorkerWithMixin(strategy)

        # 40000 input + 4096 output = 44096 total
        # 44096 - 32768 limit = 11328 overflow
        error = worker._format_overflow_error(
            total_tokens=40000,
            model_limit=32768,
            output_tokens=4096,
        )

        assert "11328" in error  # overflow amount

    def test_error_contains_guidance(self):
        """Test that error message contains actionable guidance."""
        strategy = MockDecisionStrategy()
        worker = MockWorkerWithMixin(strategy)

        error = worker._format_overflow_error(40000, 32768, 4096)

        # Check all guidance points are present
        assert "Narrower line range" in error
        assert "focus on specific methods" in error
        assert "Fewer files" in error
        assert "1-2 files" in error
        assert "Smaller height/depth" in error
        assert "call graph traversal" in error
        assert "Current focus will be cleared" in error
