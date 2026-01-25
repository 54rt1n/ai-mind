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

        request = FocusRequest(files=["file.py"])

        assert request.files == ["file.py"]
        assert request.start_line is None
        assert request.end_line is None
        assert request.height == 1
        assert request.depth == 1

    def test_focus_request_creation_with_all_args(self):
        """FocusRequest should accept all optional arguments."""
        from aim_code.strategy.base import FocusRequest

        request = FocusRequest(
            files=["a.py", "b.py"],
            start_line=10,
            end_line=50,
            height=3,
            depth=2,
        )

        assert request.files == ["a.py", "b.py"]
        assert request.start_line == 10
        assert request.end_line == 50
        assert request.height == 3
        assert request.depth == 2


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
        request = FocusRequest(files=["test.py"])

        strategy.set_focus(request)

        assert strategy.focus is request

    def test_set_focus_none_clears_focus(self, mock_chat_manager):
        """set_focus(None) should clear focus."""
        from aim_code.strategy.base import XMLCodeTurnStrategy, FocusRequest

        strategy = XMLCodeTurnStrategy(mock_chat_manager)
        strategy.set_focus(FocusRequest(files=["test.py"]))

        strategy.set_focus(None)

        assert strategy.focus is None

    def test_clear_focus_removes_focus(self, mock_chat_manager):
        """clear_focus should set focus to None."""
        from aim_code.strategy.base import XMLCodeTurnStrategy, FocusRequest

        strategy = XMLCodeTurnStrategy(mock_chat_manager)
        strategy.set_focus(FocusRequest(files=["test.py"]))

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
        strategy.set_focus(FocusRequest(files=["target.py"]))

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
        strategy.set_focus(FocusRequest(files=["file.py"], start_line=15, end_line=40))

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
        strategy.set_focus(FocusRequest(files=["test.py"]))
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
        strategy.set_focus(FocusRequest(files=["file.py"]))

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
            FocusRequest(files=["packages/aim-core/src/aim/config.py"])
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
