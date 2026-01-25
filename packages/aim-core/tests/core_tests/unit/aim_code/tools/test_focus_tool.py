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
        assert request.files == ["test.py"]

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
        assert request.files == ["a.py", "b.py"]
        assert request.start_line == 10
        assert request.end_line == 50
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
