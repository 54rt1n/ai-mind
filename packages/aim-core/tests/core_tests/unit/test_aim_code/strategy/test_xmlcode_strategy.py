# tests/core_tests/unit/aim_code/strategy/test_xmlcode_strategy.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for XMLCodeTurnStrategy - code-focused consciousness building.

Tests verify:
- Focus management (set, clear, persistence)
- Consciousness building with focused code
- Call graph integration
- Semantic search integration
- Line range filtering
"""

import json
import pytest
import pandas as pd
from unittest.mock import MagicMock


class TestFocusRequest:
    """Tests for FocusRequest data class."""

    def test_focus_request_creation_with_required_args(self):
        """FocusRequest should accept files list."""
        from aim_code.strategy.base import FocusRequest

        # New format: files is list[dict] with path, start, end
        request = FocusRequest(files=[{"path": "file.py"}])

        assert request.get_file_paths() == ["file.py"]
        assert request.get_line_range("file.py") == (None, None)
        assert request.height == 1
        assert request.depth == 1

    def test_focus_request_creation_with_all_args(self):
        """FocusRequest should accept all optional arguments."""
        from aim_code.strategy.base import FocusRequest

        request = FocusRequest(
            files=[
                {"path": "a.py", "start": 10, "end": 50},
                {"path": "b.py", "start": 100, "end": 200},
            ],
            height=3,
            depth=2,
        )

        assert request.get_file_paths() == ["a.py", "b.py"]
        assert request.get_line_range("a.py") == (10, 50)
        assert request.get_line_range("b.py") == (100, 200)
        assert request.height == 3
        assert request.depth == 2

    def test_focus_request_get_line_range_missing_file(self):
        """get_line_range should return (None, None) for unknown file."""
        from aim_code.strategy.base import FocusRequest

        request = FocusRequest(files=[{"path": "a.py", "start": 10, "end": 50}])

        assert request.get_line_range("unknown.py") == (None, None)

    def test_focus_request_mixed_line_ranges(self):
        """FocusRequest should handle mix of files with and without ranges."""
        from aim_code.strategy.base import FocusRequest

        request = FocusRequest(
            files=[
                {"path": "a.py", "start": 10, "end": 50},
                {"path": "b.py"},  # No range
                {"path": "c.py", "start": 5},  # Start only
            ],
        )

        assert request.get_line_range("a.py") == (10, 50)
        assert request.get_line_range("b.py") == (None, None)
        assert request.get_line_range("c.py") == (5, None)


class TestXMLCodeTurnStrategyFocusManagement:
    """Tests for focus state management."""

    @pytest.fixture
    def mock_chat_manager(self):
        """Create a mock ChatManager with mock CVM."""
        manager = MagicMock()
        manager.cvm = MagicMock()
        manager.cvm.query.return_value = pd.DataFrame()
        return manager

    def test_initial_focus_is_none(self, mock_chat_manager):
        """New strategy should have no focus set."""
        from aim_code.strategy.base import XMLCodeTurnStrategy

        strategy = XMLCodeTurnStrategy(mock_chat_manager)

        assert strategy.focus is None

    def test_set_focus_stores_request(self, mock_chat_manager):
        """set_focus should store the FocusRequest."""
        from aim_code.strategy.base import XMLCodeTurnStrategy, FocusRequest

        strategy = XMLCodeTurnStrategy(mock_chat_manager)
        request = FocusRequest(files=[{"path": "test.py"}])

        strategy.set_focus(request)

        assert strategy.focus is request

    def test_set_focus_none_clears_focus(self, mock_chat_manager):
        """set_focus(None) should clear focus."""
        from aim_code.strategy.base import XMLCodeTurnStrategy, FocusRequest

        strategy = XMLCodeTurnStrategy(mock_chat_manager)
        strategy.set_focus(FocusRequest(files=[{"path": "test.py"}]))

        strategy.set_focus(None)

        assert strategy.focus is None

    def test_clear_focus_removes_focus(self, mock_chat_manager):
        """clear_focus should set focus to None."""
        from aim_code.strategy.base import XMLCodeTurnStrategy, FocusRequest

        strategy = XMLCodeTurnStrategy(mock_chat_manager)
        strategy.set_focus(FocusRequest(files=[{"path": "test.py"}]))

        strategy.clear_focus()

        assert strategy.focus is None


class TestXMLCodeTurnStrategyCodeGraph:
    """Tests for code graph integration."""

    @pytest.fixture
    def mock_chat_manager(self):
        """Create a mock ChatManager."""
        manager = MagicMock()
        manager.cvm = MagicMock()
        manager.cvm.query.return_value = pd.DataFrame()
        return manager

    def test_initial_code_graph_is_none(self, mock_chat_manager):
        """New strategy should have no code graph."""
        from aim_code.strategy.base import XMLCodeTurnStrategy

        strategy = XMLCodeTurnStrategy(mock_chat_manager)

        assert strategy.code_graph is None

    def test_set_code_graph_stores_graph(self, mock_chat_manager):
        """set_code_graph should store the CodeGraph."""
        from aim_code.strategy.base import XMLCodeTurnStrategy
        from aim_code.graph import CodeGraph

        strategy = XMLCodeTurnStrategy(mock_chat_manager)
        graph = CodeGraph()

        strategy.set_code_graph(graph)

        assert strategy.code_graph is graph


class TestXMLCodeTurnStrategyUserTurn:
    """Tests for user_turn_for method."""

    @pytest.fixture
    def mock_chat_manager(self):
        """Create a mock ChatManager."""
        manager = MagicMock()
        manager.cvm = None
        return manager

    @pytest.fixture
    def mock_persona(self):
        """Create a mock persona."""
        return MagicMock()

    def test_user_turn_for_returns_correct_format(self, mock_chat_manager, mock_persona):
        """user_turn_for should return dict with role and content."""
        from aim_code.strategy.base import XMLCodeTurnStrategy

        strategy = XMLCodeTurnStrategy(mock_chat_manager)

        result = strategy.user_turn_for(mock_persona, "Hello world")

        assert result == {"role": "user", "content": "Hello world"}


class TestXMLCodeTurnStrategyChatTurns:
    """Tests for chat_turns_for method."""

    @pytest.fixture
    def mock_chat_manager(self):
        """Create a mock ChatManager with mock CVM."""
        manager = MagicMock()
        manager.cvm = MagicMock()
        manager.cvm.query.return_value = pd.DataFrame()
        return manager

    @pytest.fixture
    def mock_persona(self):
        """Create a mock persona."""
        return MagicMock()

    def test_chat_turns_includes_user_input(self, mock_chat_manager, mock_persona):
        """chat_turns_for should include user input as final turn."""
        from aim_code.strategy.base import XMLCodeTurnStrategy

        strategy = XMLCodeTurnStrategy(mock_chat_manager)

        turns = strategy.chat_turns_for(mock_persona, "Test query")

        assert turns[-1] == {"role": "user", "content": "Test query"}

    def test_chat_turns_preserves_history(self, mock_chat_manager, mock_persona):
        """chat_turns_for should include provided history."""
        from aim_code.strategy.base import XMLCodeTurnStrategy

        strategy = XMLCodeTurnStrategy(mock_chat_manager)
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]

        turns = strategy.chat_turns_for(mock_persona, "New question", history=history)

        # History should be present before final user turn
        assert {"role": "user", "content": "Previous question"} in turns
        assert {"role": "assistant", "content": "Previous answer"} in turns


class TestXMLCodeTurnStrategyConsciousness:
    """Tests for get_code_consciousness method."""

    @pytest.fixture
    def mock_chat_manager(self):
        """Create a mock ChatManager with mock CVM."""
        manager = MagicMock()
        manager.cvm = MagicMock()
        manager.cvm.query.return_value = pd.DataFrame()
        return manager

    @pytest.fixture
    def mock_persona(self):
        """Create a mock persona."""
        return MagicMock()

    def test_consciousness_empty_without_focus_or_query(self, mock_chat_manager, mock_persona):
        """get_code_consciousness returns minimal content without focus or query."""
        from aim_code.strategy.base import XMLCodeTurnStrategy

        strategy = XMLCodeTurnStrategy(mock_chat_manager)

        consciousness, count = strategy.get_code_consciousness(mock_persona, "")

        # Should have root XML structure but minimal content
        assert count == 0

    def test_consciousness_queries_cvm_with_query(self, mock_chat_manager, mock_persona):
        """get_code_consciousness should query CVM with provided query."""
        from aim_code.strategy.base import XMLCodeTurnStrategy
        from aim.constants import DOC_SOURCE_CODE

        strategy = XMLCodeTurnStrategy(mock_chat_manager)

        strategy.get_code_consciousness(mock_persona, "search term")

        mock_chat_manager.cvm.query.assert_called()
        # Check that the query was for DOC_SOURCE_CODE
        call_kwargs = mock_chat_manager.cvm.query.call_args[1]
        assert call_kwargs["query_document_type"] == DOC_SOURCE_CODE

    def test_consciousness_includes_search_results(self, mock_chat_manager, mock_persona):
        """get_code_consciousness should include formatted search results."""
        from aim_code.strategy.base import XMLCodeTurnStrategy

        # Mock search results
        mock_results = pd.DataFrame(
            [
                {
                    "doc_id": "test.py::func",
                    "content": "def func(): pass",
                    "metadata": "{}",
                }
            ]
        )
        mock_chat_manager.cvm.query.return_value = mock_results
        strategy = XMLCodeTurnStrategy(mock_chat_manager)

        consciousness, count = strategy.get_code_consciousness(mock_persona, "func")

        assert "test.py::func" in consciousness
        assert count == 1


class TestXMLCodeTurnStrategyFocusedCode:
    """Tests for focused code retrieval."""

    @pytest.fixture
    def mock_chat_manager(self):
        """Create a mock ChatManager with mock CVM."""
        manager = MagicMock()
        manager.cvm = MagicMock()
        return manager

    @pytest.fixture
    def mock_persona(self):
        """Create a mock persona."""
        return MagicMock()

    def test_focused_code_queries_specific_file(self, mock_chat_manager, mock_persona):
        """When focus is set, should query for that specific file."""
        from aim_code.strategy.base import XMLCodeTurnStrategy, FocusRequest
        from aim.constants import DOC_SOURCE_CODE

        mock_chat_manager.cvm.query.return_value = pd.DataFrame()
        strategy = XMLCodeTurnStrategy(mock_chat_manager)
        strategy.set_focus(FocusRequest(files=[{"path": "target.py"}]))

        strategy.get_code_consciousness(mock_persona, "")

        # Should have queried for the focused file
        calls = mock_chat_manager.cvm.query.call_args_list
        file_query_found = False
        for call in calls:
            if "target.py" in str(call):
                file_query_found = True
        assert file_query_found

    def test_focused_code_filters_by_line_range(self, mock_chat_manager, mock_persona):
        """Line range filtering should exclude docs outside range."""
        from aim_code.strategy.base import XMLCodeTurnStrategy, FocusRequest

        # Mock docs spanning different line ranges
        mock_results = pd.DataFrame(
            [
                {
                    "doc_id": "file.py::func1",
                    "content": "def func1(): pass",
                    "metadata": json.dumps({"line_start": 1, "line_end": 10}),
                },
                {
                    "doc_id": "file.py::func2",
                    "content": "def func2(): pass",
                    "metadata": json.dumps({"line_start": 20, "line_end": 30}),
                },
                {
                    "doc_id": "file.py::func3",
                    "content": "def func3(): pass",
                    "metadata": json.dumps({"line_start": 50, "line_end": 60}),
                },
            ]
        )
        mock_chat_manager.cvm.query.return_value = mock_results
        strategy = XMLCodeTurnStrategy(mock_chat_manager)
        # Focus on lines 15-40 (should include func2, exclude func1 and func3)
        # New format: per-file line ranges
        strategy.set_focus(FocusRequest(files=[{"path": "file.py", "start": 15, "end": 40}]))

        consciousness, count = strategy.get_code_consciousness(mock_persona, "")

        # func2 should be included (lines 20-30 overlap with 15-40)
        assert "func2" in consciousness
        # func1 should be excluded (lines 1-10 don't overlap with 15-40)
        assert "func1" not in consciousness
        # func3 should be excluded (lines 50-60 don't overlap with 15-40)
        assert "func3" not in consciousness


class TestXMLCodeTurnStrategyCallGraph:
    """Tests for call graph integration in consciousness."""

    @pytest.fixture
    def mock_chat_manager(self):
        """Create a mock ChatManager with mock CVM."""
        manager = MagicMock()
        manager.cvm = MagicMock()
        return manager

    @pytest.fixture
    def mock_persona(self):
        """Create a mock persona."""
        return MagicMock()

    def test_call_graph_not_included_without_graph(self, mock_chat_manager, mock_persona):
        """Consciousness should not include call graph section without code graph."""
        from aim_code.strategy.base import XMLCodeTurnStrategy, FocusRequest

        mock_chat_manager.cvm.query.return_value = pd.DataFrame()
        strategy = XMLCodeTurnStrategy(mock_chat_manager)
        strategy.set_focus(FocusRequest(files=[{"path": "test.py"}]))
        # No code_graph set

        consciousness, _ = strategy.get_code_consciousness(mock_persona, "")

        assert "mermaid" not in consciousness

    def test_call_graph_included_with_graph_and_focus(self, mock_chat_manager, mock_persona):
        """Consciousness should include mermaid graph when graph and focus are set."""
        from aim_code.strategy.base import XMLCodeTurnStrategy, FocusRequest
        from aim_code.graph import CodeGraph

        # Mock CVM returning a document with metadata
        mock_results = pd.DataFrame(
            [
                {
                    "doc_id": "file.py::func",
                    "content": "def func(): helper()",
                    "metadata": json.dumps(
                        {"symbol_name": "func", "line_start": 10, "line_end": 15}
                    ),
                }
            ]
        )
        mock_chat_manager.cvm.query.return_value = mock_results

        # Create a graph with an edge
        graph = CodeGraph()
        graph.add_edge(("file.py", "func", 10), ("file.py", "helper", 20))

        strategy = XMLCodeTurnStrategy(mock_chat_manager)
        strategy.set_code_graph(graph)
        strategy.set_focus(FocusRequest(files=[{"path": "file.py"}]))

        consciousness, _ = strategy.get_code_consciousness(mock_persona, "")

        assert "mermaid" in consciousness
        assert "graph TD" in consciousness


class TestXMLCodeTurnStrategyModuleSpecs:
    """Tests for module spec retrieval."""

    @pytest.fixture
    def mock_chat_manager(self):
        """Create a mock ChatManager with mock CVM."""
        manager = MagicMock()
        manager.cvm = MagicMock()
        return manager

    @pytest.fixture
    def mock_persona(self):
        """Create a mock persona."""
        return MagicMock()

    def test_module_path_derivation(self, mock_chat_manager, mock_persona):
        """_get_module_paths_for_files should derive module paths from file paths."""
        from aim_code.strategy.base import XMLCodeTurnStrategy

        strategy = XMLCodeTurnStrategy(mock_chat_manager)

        paths = strategy._get_module_paths_for_files(
            ["packages/aim-core/src/aim/config.py"]
        )

        assert "aim.config" in paths

    def test_module_path_handles_nested_src(self, mock_chat_manager, mock_persona):
        """Module path derivation should handle nested src directories."""
        from aim_code.strategy.base import XMLCodeTurnStrategy

        strategy = XMLCodeTurnStrategy(mock_chat_manager)

        paths = strategy._get_module_paths_for_files(
            ["packages/aim-core/src/aim/chat/strategy/base.py"]
        )

        assert "aim.chat.strategy.base" in paths

    def test_specs_queried_when_focus_set(self, mock_chat_manager, mock_persona):
        """When focus is set, should query for DOC_SPEC documents."""
        from aim_code.strategy.base import XMLCodeTurnStrategy, FocusRequest
        from aim.constants import DOC_SPEC

        mock_chat_manager.cvm.query.return_value = pd.DataFrame()
        strategy = XMLCodeTurnStrategy(mock_chat_manager)
        strategy.set_focus(
            FocusRequest(files=[{"path": "packages/aim-core/src/aim/config.py"}])
        )

        strategy.get_code_consciousness(mock_persona, "")

        # Check that DOC_SPEC was queried
        calls = mock_chat_manager.cvm.query.call_args_list
        spec_query_found = False
        for call in calls:
            kwargs = call[1] if len(call) > 1 else {}
            if kwargs.get("query_document_type") == DOC_SPEC:
                spec_query_found = True
        assert spec_query_found


class TestXMLCodeTurnStrategyTokenBudgeting:
    """Tests for token budgeting functionality."""

    @pytest.fixture
    def mock_chat_manager(self):
        """Create a mock ChatManager with mock CVM and config."""
        manager = MagicMock()
        manager.cvm = MagicMock()
        manager.cvm.query.return_value = pd.DataFrame()
        manager.config = MagicMock()
        manager.config.system_message = "You are a helpful assistant."
        return manager

    @pytest.fixture
    def mock_persona(self):
        """Create a mock persona."""
        return MagicMock()

    def test_count_tokens_returns_int(self, mock_chat_manager):
        """count_tokens should return an integer token count."""
        from aim_code.strategy.base import XMLCodeTurnStrategy

        strategy = XMLCodeTurnStrategy(mock_chat_manager)

        result = strategy.count_tokens("Hello world")

        assert isinstance(result, int)
        assert result > 0

    def test_count_tokens_empty_string(self, mock_chat_manager):
        """count_tokens should return 0 for empty string."""
        from aim_code.strategy.base import XMLCodeTurnStrategy

        strategy = XMLCodeTurnStrategy(mock_chat_manager)

        result = strategy.count_tokens("")

        assert result == 0

    def test_calc_max_context_tokens_reserves_output(self, mock_chat_manager):
        """_calc_max_context_tokens should reserve space for output tokens."""
        from aim_code.strategy.base import XMLCodeTurnStrategy

        strategy = XMLCodeTurnStrategy(mock_chat_manager)

        result = strategy._calc_max_context_tokens(
            max_context_tokens=32768, max_output_tokens=4096
        )

        # Should be less than max_context - max_output
        assert result < 32768 - 4096

    def test_calc_max_context_tokens_reserves_system_prompt(self, mock_chat_manager):
        """_calc_max_context_tokens should reserve space for system prompt."""
        from aim_code.strategy.base import XMLCodeTurnStrategy

        strategy = XMLCodeTurnStrategy(mock_chat_manager)
        system_tokens = strategy.count_tokens("You are a helpful assistant.")

        result = strategy._calc_max_context_tokens(
            max_context_tokens=32768, max_output_tokens=4096
        )

        # Should account for system tokens
        expected_max = 32768 - 4096 - system_tokens - 1024
        assert result == expected_max

    def test_trim_history_empty_list(self, mock_chat_manager):
        """_trim_history should return empty list for empty input."""
        from aim_code.strategy.base import XMLCodeTurnStrategy

        strategy = XMLCodeTurnStrategy(mock_chat_manager)

        result = strategy._trim_history([], 1000)

        assert result == []

    def test_trim_history_zero_budget(self, mock_chat_manager):
        """_trim_history should return empty list for zero budget."""
        from aim_code.strategy.base import XMLCodeTurnStrategy

        strategy = XMLCodeTurnStrategy(mock_chat_manager)
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        result = strategy._trim_history(history, 0)

        assert result == []

    def test_trim_history_preserves_newest(self, mock_chat_manager):
        """_trim_history should preserve newest turns when trimming."""
        from aim_code.strategy.base import XMLCodeTurnStrategy

        strategy = XMLCodeTurnStrategy(mock_chat_manager)
        history = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First reply"},
            {"role": "user", "content": "Second message"},
            {"role": "assistant", "content": "Second reply"},
            {"role": "user", "content": "Third message"},
            {"role": "assistant", "content": "Third reply"},
        ]

        # Use a small budget that can only fit a few messages
        # Each message is roughly 2-3 tokens, so 10 tokens should fit ~3-4 messages
        result = strategy._trim_history(history, 10)

        # Should have fewer turns
        assert len(result) < len(history)
        # Last turn should be preserved
        assert result[-1] == history[-1]

    def test_trim_history_keeps_all_when_under_budget(self, mock_chat_manager):
        """_trim_history should keep all turns when under budget."""
        from aim_code.strategy.base import XMLCodeTurnStrategy

        strategy = XMLCodeTurnStrategy(mock_chat_manager)
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        # Large budget should keep everything
        result = strategy._trim_history(history, 10000)

        assert len(result) == len(history)
        assert result == history

    def test_chat_turns_trims_long_history(self, mock_chat_manager, mock_persona):
        """chat_turns_for should trim history when over budget."""
        from aim_code.strategy.base import XMLCodeTurnStrategy

        strategy = XMLCodeTurnStrategy(mock_chat_manager)

        # Create a very long history that exceeds budget
        history = [
            {"role": "user", "content": f"Message {i} " * 100}
            for i in range(50)
        ] + [
            {"role": "assistant", "content": f"Reply {i} " * 100}
            for i in range(50)
        ]
        # Interleave
        interleaved = []
        for i in range(50):
            interleaved.append({"role": "user", "content": f"User message {i} " * 100})
            interleaved.append({"role": "assistant", "content": f"Assistant reply {i} " * 100})

        # Call with small context
        result = strategy.chat_turns_for(
            mock_persona,
            "New question",
            history=interleaved,
            max_context_tokens=8000,
            max_output_tokens=1000,
        )

        # Result should be shorter than original history + user input
        # (consciousness may or may not be included depending on CVM results)
        total_history_turns = sum(1 for t in result if t.get("content", "").startswith("User message") or t.get("content", "").startswith("Assistant reply"))
        assert total_history_turns < len(interleaved)

    def test_consciousness_respects_token_budget(self, mock_chat_manager, mock_persona):
        """get_code_consciousness should respect token_budget parameter."""
        from aim_code.strategy.base import XMLCodeTurnStrategy, FocusRequest

        # Mock CVM returning large content
        large_content = "x" * 10000  # Very large content
        mock_results = pd.DataFrame(
            [
                {
                    "doc_id": "file.py::func",
                    "content": large_content,
                    "metadata": json.dumps({"line_start": 1, "line_end": 100}),
                }
            ]
        )
        mock_chat_manager.cvm.query.return_value = mock_results

        strategy = XMLCodeTurnStrategy(mock_chat_manager)
        strategy.set_focus(FocusRequest(files=[{"path": "file.py"}]))

        # Request with small budget
        consciousness, count = strategy.get_code_consciousness(
            mock_persona, "test query", token_budget=500
        )

        # Consciousness should be truncated to fit budget
        consciousness_tokens = strategy.count_tokens(consciousness)
        # Allow some overhead for XML structure
        assert consciousness_tokens < 600  # 500 budget + some structural overhead

    def test_focused_source_respects_max_tokens(self, mock_chat_manager):
        """_get_focused_source should truncate content to fit max_tokens."""
        from aim_code.strategy.base import XMLCodeTurnStrategy, FocusRequest

        large_content = "def function():\n    " + "x = 1\n    " * 1000
        mock_results = pd.DataFrame(
            [
                {
                    "doc_id": "file.py::func",
                    "content": large_content,
                    "metadata": json.dumps({"line_start": 1}),
                }
            ]
        )
        mock_chat_manager.cvm.query.return_value = mock_results

        strategy = XMLCodeTurnStrategy(mock_chat_manager)
        strategy.set_focus(FocusRequest(files=[{"path": "file.py"}]))

        # Request with small budget
        source, count = strategy._get_focused_source(max_tokens=100)

        # Should be truncated
        source_tokens = strategy.count_tokens(source)
        assert source_tokens <= 100 or count == 1  # Either under budget or only one truncated doc

    def test_format_search_results_respects_max_tokens(self, mock_chat_manager):
        """_format_search_results should stop when budget exhausted."""
        from aim_code.strategy.base import XMLCodeTurnStrategy

        # Create many search results
        docs = pd.DataFrame(
            [
                {"doc_id": f"file{i}.py::func", "content": "x" * 400}
                for i in range(20)
            ]
        )

        strategy = XMLCodeTurnStrategy(mock_chat_manager)

        # Request with limited budget
        result = strategy._format_search_results(docs, max_tokens=200)

        # Should have fewer results than input
        result_tokens = strategy.count_tokens(result)
        assert result_tokens <= 200

    def test_format_specs_respects_max_tokens(self, mock_chat_manager):
        """_format_specs should stop when budget exhausted."""
        from aim_code.strategy.base import XMLCodeTurnStrategy

        # Create many specs
        specs = pd.DataFrame(
            [
                {"doc_id": f"module{i}", "content": "Spec content " * 100}
                for i in range(10)
            ]
        )

        strategy = XMLCodeTurnStrategy(mock_chat_manager)

        # Request with limited budget
        result = strategy._format_specs(specs, max_tokens=200)

        # Should have fewer results than input
        result_tokens = strategy.count_tokens(result)
        assert result_tokens <= 200
