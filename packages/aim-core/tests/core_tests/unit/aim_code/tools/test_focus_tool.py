# tests/core_tests/unit/aim_code/tools/test_focus_tool.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for FocusTool - code focus context setting.

Tests verify:
- Focus setting and clearing
- Symbol preview generation
- Line range filtering in preview
- Return value structure
"""

import json
import pytest
import pandas as pd
from unittest.mock import MagicMock


class TestFocusToolFocusSetting:
    """Tests for focus() method."""

    @pytest.fixture
    def mock_strategy(self):
        """Create a mock XMLCodeTurnStrategy with mock chat manager."""
        strategy = MagicMock()
        strategy.chat = MagicMock()
        strategy.chat.cvm = MagicMock()
        strategy.chat.cvm.query.return_value = pd.DataFrame()
        return strategy

    def test_focus_sets_request_on_strategy(self, mock_strategy):
        """focus() should call set_focus on strategy."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        tool.focus(files=["test.py"])

        mock_strategy.set_focus.assert_called_once()
        request = mock_strategy.set_focus.call_args[0][0]
        # Files are now list[dict] with path, start, end
        assert request.get_file_paths() == ["test.py"]

    def test_focus_passes_all_parameters(self, mock_strategy):
        """focus() should pass all parameters to FocusRequest."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        tool.focus(
            files=["a.py", "b.py"],
            start_line=10,
            end_line=50,
            height=3,
            depth=2,
        )

        request = mock_strategy.set_focus.call_args[0][0]
        # Files are now list[dict] with per-file line ranges
        assert request.get_file_paths() == ["a.py", "b.py"]
        # Global start/end are applied to all files
        assert request.get_line_range("a.py") == (10, 50)
        assert request.get_line_range("b.py") == (10, 50)
        assert request.height == 3
        assert request.depth == 2

    def test_focus_returns_success_dict(self, mock_strategy):
        """focus() should return dict with success=True."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        result = tool.focus(files=["test.py"])

        assert result["success"] is True

    def test_focus_returns_focused_files(self, mock_strategy):
        """focus() should return the files in response."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        result = tool.focus(files=["file1.py", "file2.py"])

        assert result["focused_files"] == ["file1.py", "file2.py"]

    def test_focus_returns_line_range(self, mock_strategy):
        """focus() should return formatted line range."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        result = tool.focus(files=["test.py"], start_line=10, end_line=50)

        assert result["line_range"] == "10-50"

    def test_focus_returns_default_line_range(self, mock_strategy):
        """focus() should return 'start-end' when no lines specified."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        result = tool.focus(files=["test.py"])

        assert result["line_range"] == "start-end"

    def test_focus_returns_graph_parameters(self, mock_strategy):
        """focus() should return graph height and depth."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        result = tool.focus(files=["test.py"], height=5, depth=3)

        assert result["graph_height"] == 5
        assert result["graph_depth"] == 3


class TestFocusToolSymbolPreview:
    """Tests for symbol preview in focus() response."""

    @pytest.fixture
    def mock_strategy(self):
        """Create a mock strategy with CVM returning symbols."""
        strategy = MagicMock()
        strategy.chat = MagicMock()
        strategy.chat.cvm = MagicMock()
        return strategy

    def test_symbols_in_focus_count(self, mock_strategy):
        """focus() should return count of symbols found."""
        from aim_code.tools.focus import FocusTool

        # Mock CVM returning 3 documents
        mock_results = pd.DataFrame(
            [
                {"metadata": json.dumps({"symbol_name": "func1", "line_start": 1, "line_end": 10})},
                {"metadata": json.dumps({"symbol_name": "func2", "line_start": 15, "line_end": 20})},
                {"metadata": json.dumps({"symbol_name": "func3", "line_start": 25, "line_end": 30})},
            ]
        )
        mock_strategy.chat.cvm.query.return_value = mock_results
        tool = FocusTool(mock_strategy)

        result = tool.focus(files=["test.py"])

        assert result["symbols_in_focus"] == 3

    def test_symbols_filtered_by_line_range(self, mock_strategy):
        """focus() symbol count should reflect line range filtering."""
        from aim_code.tools.focus import FocusTool

        # Mock CVM returning 3 documents
        mock_results = pd.DataFrame(
            [
                {"metadata": json.dumps({"symbol_name": "func1", "line_start": 1, "line_end": 10})},
                {"metadata": json.dumps({"symbol_name": "func2", "line_start": 15, "line_end": 20})},
                {"metadata": json.dumps({"symbol_name": "func3", "line_start": 25, "line_end": 30})},
            ]
        )
        mock_strategy.chat.cvm.query.return_value = mock_results
        tool = FocusTool(mock_strategy)

        # Focus on lines 12-22 (should include only func2)
        result = tool.focus(files=["test.py"], start_line=12, end_line=22)

        assert result["symbols_in_focus"] == 1

    def test_symbols_zero_when_cvm_none(self, mock_strategy):
        """focus() should return 0 symbols when CVM is None."""
        from aim_code.tools.focus import FocusTool

        mock_strategy.chat.cvm = None
        tool = FocusTool(mock_strategy)

        result = tool.focus(files=["test.py"])

        assert result["symbols_in_focus"] == 0


class TestFocusToolClearFocus:
    """Tests for clear_focus() method."""

    @pytest.fixture
    def mock_strategy(self):
        """Create a mock strategy."""
        strategy = MagicMock()
        strategy.chat = MagicMock()
        strategy.chat.cvm = None
        return strategy

    def test_clear_focus_calls_strategy_clear(self, mock_strategy):
        """clear_focus() should call clear_focus on strategy."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        tool.clear_focus()

        mock_strategy.clear_focus.assert_called_once()

    def test_clear_focus_returns_success(self, mock_strategy):
        """clear_focus() should return success dict."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        result = tool.clear_focus()

        assert result["success"] is True
        assert "message" in result


class TestFocusToolEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def mock_strategy(self):
        """Create a mock strategy."""
        strategy = MagicMock()
        strategy.chat = MagicMock()
        strategy.chat.cvm = MagicMock()
        strategy.chat.cvm.query.return_value = pd.DataFrame()
        return strategy

    def test_focus_empty_files_list(self, mock_strategy):
        """focus() should handle empty files list."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        result = tool.focus(files=[])

        assert result["success"] is True
        assert result["focused_files"] == []
        assert result["symbols_in_focus"] == 0

    def test_focus_multiple_files(self, mock_strategy):
        """focus() should query CVM for each file."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        tool.focus(files=["file1.py", "file2.py", "file3.py"])

        # CVM should be queried once per file for symbol preview
        assert mock_strategy.chat.cvm.query.call_count >= 3

    def test_focus_partial_line_range_start_only(self, mock_strategy):
        """focus() should handle start_line without end_line."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        result = tool.focus(files=["test.py"], start_line=10)

        assert result["line_range"] == "10-end"

    def test_focus_partial_line_range_end_only(self, mock_strategy):
        """focus() should handle end_line without start_line."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        result = tool.focus(files=["test.py"], end_line=50)

        assert result["line_range"] == "start-50"


class TestFocusToolEntityResolution:
    """Tests for entity name to file path resolution."""

    @pytest.fixture
    def mock_strategy(self):
        """Create a mock strategy."""
        strategy = MagicMock()
        strategy.chat = MagicMock()
        strategy.chat.cvm = MagicMock()
        strategy.chat.cvm.query.return_value = pd.DataFrame()
        return strategy

    @pytest.fixture
    def sample_entities_dict(self):
        """Sample entities as dicts (like raw JSON from Redis)."""
        return [
            {
                "entity_id": "#1",
                "name": "model.py",
                "entity_type": "object",
                "metadata": {"file_path": "/repo/src/model.py", "rel_path": "src/model.py"},
            },
            {
                "entity_id": "#2",
                "name": "utils.py",
                "entity_type": "object",
                "metadata": {"file_path": "/repo/src/utils.py", "rel_path": "src/utils.py"},
            },
            {
                "entity_id": "#3",
                "name": "Andi",
                "entity_type": "ai",
                "metadata": {},
            },
        ]

    @pytest.fixture
    def sample_entities_objects(self):
        """Sample entities as objects (like EntityState instances)."""
        entity1 = MagicMock()
        entity1.name = "model.py"
        entity1.metadata = {"file_path": "/repo/src/model.py", "rel_path": "src/model.py"}

        entity2 = MagicMock()
        entity2.name = "utils.py"
        entity2.metadata = {"file_path": "/repo/src/utils.py", "rel_path": "src/utils.py"}

        entity3 = MagicMock()
        entity3.name = "Andi"
        entity3.metadata = {}

        return [entity1, entity2, entity3]

    def test_resolve_entity_name_to_file_path_dict(self, mock_strategy, sample_entities_dict):
        """Entity names without / should resolve to file paths via dict metadata."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        result = tool.focus(files=["model.py"], entities=sample_entities_dict)

        # Should resolve to full path
        assert result["focused_files"] == ["/repo/src/model.py"]
        # Strategy should receive resolved path in file specs
        request = mock_strategy.set_focus.call_args[0][0]
        assert request.get_file_paths() == ["/repo/src/model.py"]

    def test_resolve_entity_name_to_file_path_object(self, mock_strategy, sample_entities_objects):
        """Entity names without / should resolve to file paths via object metadata."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        result = tool.focus(files=["model.py"], entities=sample_entities_objects)

        # Should resolve to full path
        assert result["focused_files"] == ["/repo/src/model.py"]

    def test_paths_with_slash_pass_through(self, mock_strategy, sample_entities_dict):
        """File paths containing / should not be resolved."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        result = tool.focus(files=["/some/other/path.py"], entities=sample_entities_dict)

        # Path with / should not be changed
        assert result["focused_files"] == ["/some/other/path.py"]

    def test_unresolved_names_pass_through(self, mock_strategy, sample_entities_dict):
        """Entity names not found should pass through unchanged."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        result = tool.focus(files=["unknown.py"], entities=sample_entities_dict)

        # Unknown name should pass through unchanged
        assert result["focused_files"] == ["unknown.py"]

    def test_mixed_paths_and_names(self, mock_strategy, sample_entities_dict):
        """Should handle mix of paths and entity names."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        result = tool.focus(
            files=["model.py", "/direct/path.py", "utils.py"],
            entities=sample_entities_dict,
        )

        # Should resolve names but not paths
        assert result["focused_files"] == [
            "/repo/src/model.py",
            "/direct/path.py",
            "/repo/src/utils.py",
        ]

    def test_no_entities_passes_files_through(self, mock_strategy):
        """When no entities provided, files should pass through unchanged."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        result = tool.focus(files=["model.py", "utils.py"])

        # No resolution without entities
        assert result["focused_files"] == ["model.py", "utils.py"]

    def test_empty_entities_passes_files_through(self, mock_strategy):
        """When entities list is empty, files should pass through unchanged."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        result = tool.focus(files=["model.py"], entities=[])

        assert result["focused_files"] == ["model.py"]

    def test_entity_without_file_path_passes_name_through(self, mock_strategy):
        """Entity with empty metadata should not resolve."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        # Andi has empty metadata, so "Andi" should pass through
        entities = [{"name": "Andi", "metadata": {}}]
        result = tool.focus(files=["Andi"], entities=entities)

        assert result["focused_files"] == ["Andi"]

    def test_first_matching_entity_wins(self, mock_strategy):
        """When multiple entities have same name, first one is used."""
        from aim_code.tools.focus import FocusTool

        tool = FocusTool(mock_strategy)

        entities = [
            {"name": "model.py", "metadata": {"file_path": "/first/model.py"}},
            {"name": "model.py", "metadata": {"file_path": "/second/model.py"}},
        ]
        result = tool.focus(files=["model.py"], entities=entities)

        # First match should win
        assert result["focused_files"] == ["/first/model.py"]
