# tests/andimud_tests/test_terminals.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for Terminal typeclasses and commands."""

import json
import pytest
from unittest.mock import MagicMock, patch, call


class TestTerminalBase:
    """Tests for the base Terminal class."""

    def test_terminal_has_display_output_method(self):
        """Test that Terminal class has display_output method."""
        from andimud.typeclasses.terminals import Terminal

        # The method should exist on the class
        assert hasattr(Terminal, "display_output")
        assert callable(getattr(Terminal, "display_output"))


class TestTerminalTypeclass:
    """Tests for Terminal subclasses."""

    def test_webterminal_inherits_from_terminal(self):
        """Test WebTerminal inherits from Terminal."""
        from andimud.typeclasses.terminals import Terminal, WebTerminal

        assert issubclass(WebTerminal, Terminal)

    def test_codeterminal_inherits_from_terminal(self):
        """Test CodeTerminal inherits from Terminal."""
        from andimud.typeclasses.terminals import Terminal, CodeTerminal

        assert issubclass(CodeTerminal, Terminal)

    def test_marketterminal_inherits_from_terminal(self):
        """Test MarketTerminal inherits from Terminal."""
        from andimud.typeclasses.terminals import Terminal, MarketTerminal

        assert issubclass(MarketTerminal, Terminal)

    def test_newsterminal_inherits_from_terminal(self):
        """Test NewsTerminal inherits from Terminal."""
        from andimud.typeclasses.terminals import Terminal, NewsTerminal

        assert issubclass(NewsTerminal, Terminal)

    def test_researchterminal_inherits_from_terminal(self):
        """Test ResearchTerminal inherits from Terminal."""
        from andimud.typeclasses.terminals import Terminal, ResearchTerminal

        assert issubclass(ResearchTerminal, Terminal)

    def test_listterminal_inherits_from_terminal(self):
        """Test ListTerminal inherits from Terminal."""
        from andimud.typeclasses.terminals import Terminal, ListTerminal

        assert issubclass(ListTerminal, Terminal)


class TestTerminalCommands:
    """Tests for terminal command classes."""

    def test_visit_webpage_cmd_exists(self):
        """Test CmdVisitWebpage exists and has correct key."""
        from andimud.commands.mud.object_commands.terminal import CmdVisitWebpage

        cmd = CmdVisitWebpage()
        assert cmd.key == "visit_webpage"

    def test_get_feed_cmd_exists(self):
        """Test CmdGetFeed exists and has correct key."""
        from andimud.commands.mud.object_commands.terminal import CmdGetFeed

        cmd = CmdGetFeed()
        assert cmd.key == "get_feed"

    def test_research_cmd_exists(self):
        """Test CmdResearch exists and has correct key."""
        from andimud.commands.mud.object_commands.terminal import CmdResearch

        cmd = CmdResearch()
        assert cmd.key == "research"

    def test_read_doc_cmd_exists(self):
        """Test CmdReadDoc exists and has correct key."""
        from andimud.commands.mud.object_commands.terminal import CmdReadDoc

        cmd = CmdReadDoc()
        assert cmd.key == "read_doc"

    def test_show_list_cmd_exists(self):
        """Test CmdShowList exists and has correct key."""
        from andimud.commands.mud.object_commands.terminal import CmdShowList

        cmd = CmdShowList()
        assert cmd.key == "show_list"

    def test_add_item_cmd_exists(self):
        """Test CmdAddItem exists and has correct key."""
        from andimud.commands.mud.object_commands.terminal import CmdAddItem

        cmd = CmdAddItem()
        assert cmd.key == "add_item"

    def test_check_item_cmd_exists(self):
        """Test CmdCheckItem exists and has correct key."""
        from andimud.commands.mud.object_commands.terminal import CmdCheckItem

        cmd = CmdCheckItem()
        assert cmd.key == "check_item"

    def test_stock_quote_cmd_exists(self):
        """Test CmdStockQuote exists and has correct key."""
        from andimud.commands.mud.object_commands.terminal import CmdStockQuote

        cmd = CmdStockQuote()
        assert cmd.key == "stock_quote"


class TestContainerMCPToolExecute:
    """Tests for ContainerMCPTool execute dispatcher."""

    @pytest.fixture
    def tool(self):
        """Create a ContainerMCPTool instance."""
        from aim.tool.impl.container_mcp import ContainerMCPTool

        return ContainerMCPTool(endpoint="http://localhost:3000")

    def test_visit_webpage_requires_url(self, tool):
        """Test visit_webpage requires url parameter."""
        with pytest.raises(ValueError, match="URL parameter is required"):
            tool.execute("visit_webpage", {})

    def test_get_feed_requires_url(self, tool):
        """Test get_feed requires url parameter."""
        with pytest.raises(ValueError, match="URL parameter is required"):
            tool.execute("get_feed", {})

    def test_research_requires_query(self, tool):
        """Test research requires query parameter."""
        with pytest.raises(ValueError, match="Query parameter is required"):
            tool.execute("research", {})

    def test_read_doc_requires_doc_id(self, tool):
        """Test read_doc requires doc_id parameter."""
        with pytest.raises(ValueError, match="doc_id parameter is required"):
            tool.execute("read_doc", {})

    def test_show_list_requires_list_id(self, tool):
        """Test show_list requires list_id parameter."""
        with pytest.raises(ValueError, match="list_id parameter is required"):
            tool.execute("show_list", {})

    def test_add_item_requires_list_id(self, tool):
        """Test add_item requires list_id parameter."""
        with pytest.raises(ValueError, match="list_id parameter is required"):
            tool.execute("add_item", {})

    def test_add_item_requires_text(self, tool):
        """Test add_item requires text parameter."""
        with pytest.raises(ValueError, match="text parameter is required"):
            tool.execute("add_item", {"list_id": "test"})

    def test_check_item_requires_list_id(self, tool):
        """Test check_item requires list_id parameter."""
        with pytest.raises(ValueError, match="list_id parameter is required"):
            tool.execute("check_item", {})

    def test_check_item_requires_item_id(self, tool):
        """Test check_item requires item_id parameter."""
        with pytest.raises(ValueError, match="item_id parameter is required"):
            tool.execute("check_item", {"list_id": "test"})

    def test_stock_quote_requires_symbol(self, tool):
        """Test stock_quote requires symbol parameter."""
        with pytest.raises(ValueError, match="Symbol parameter is required"):
            tool.execute("stock_quote", {})

    def test_unknown_function_returns_error(self, tool):
        """Test unknown function returns error dict."""
        result = tool.execute("unknown_function", {})
        assert result["success"] is False
        assert "Unknown container-mcp tool" in result["error"]


class TestTerminalEventPublishing:
    """Tests for terminal event publishing to Redis."""

    @pytest.fixture
    def mock_terminal(self):
        """Create a mock terminal instance with necessary attributes."""
        from andimud.typeclasses.terminals import CodeTerminal
        from evennia.objects.objects import DefaultRoom

        terminal = MagicMock(spec=CodeTerminal)
        terminal.key = "test_terminal"
        terminal.dbref = "#123"
        terminal.__class__.__name__ = "CodeTerminal"

        # Mock location (room) - make isinstance check pass
        terminal.location = MagicMock(spec=DefaultRoom)
        terminal.location.dbref = "#42"
        terminal.location.key = "Test Room"
        terminal.location.msg_contents = MagicMock()

        # Mock db attributes
        terminal.db = MagicMock()
        terminal.db.max_output_len = 4096
        terminal.db.aura_name = "CODE_ACCESS"

        return terminal

    @pytest.fixture
    def mock_caller(self):
        """Create a mock caller (player or AI character)."""
        caller = MagicMock()
        caller.key = "TestPlayer"
        caller.dbref = "#99"
        caller.msg = MagicMock()
        caller.get_actor_type = MagicMock(return_value="player")
        return caller

    def test_display_output_publishes_event(self, mock_terminal, mock_caller):
        """Test that display_output calls _publish_terminal_event."""
        from andimud.typeclasses.terminals import Terminal

        # Mock _publish_terminal_event on the terminal instance
        mock_terminal._publish_terminal_event = MagicMock()

        # Call display_output
        Terminal.display_output(mock_terminal, "py_exec", ">>> print('hello')\nhello", caller=mock_caller)

        # Verify _publish_terminal_event was called with correct arguments
        assert mock_terminal._publish_terminal_event.call_count == 1
        call_args = mock_terminal._publish_terminal_event.call_args
        assert call_args[0][0] == "py_exec"
        assert call_args[0][1] == ">>> print('hello')\nhello"
        assert call_args[0][2] == mock_caller or call_args[1].get('caller') == mock_caller

        # Verify room message was sent twice (narrative + output)
        assert mock_terminal.location.msg_contents.call_count == 2
        # First call: narrative action
        assert "types away at" in mock_terminal.location.msg_contents.call_args_list[0][0][0]
        # Second call: actual output
        assert "py_exec:" in mock_terminal.location.msg_contents.call_args_list[1][0][0]

    @patch("andimud.typeclasses.terminals.redis.from_url")
    @patch("andimud.typeclasses.terminals.append_mud_event")
    @patch("andimud.typeclasses.terminals._utc_now")
    def test_publish_terminal_event_structure(self, mock_utc, mock_append, mock_redis, mock_terminal, mock_caller):
        """Test that _publish_terminal_event creates correct event structure."""
        from andimud.typeclasses.terminals import Terminal
        from datetime import datetime

        # Setup mocks
        mock_utc.return_value = datetime(2026, 1, 15, 12, 0, 0)
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client

        # Call the method directly
        Terminal._publish_terminal_event(mock_terminal, "py_exec", ">>> print('hello')\nhello", caller=mock_caller)

        # Verify Redis was accessed
        mock_redis.assert_called_once_with("redis://localhost:6379")

        # Verify append_mud_event was called
        assert mock_append.called
        call_args = mock_append.call_args[0]
        assert len(call_args) == 2

        # Extract the payload
        payload_wrapper = call_args[1]
        assert "data" in payload_wrapper
        payload = json.loads(payload_wrapper["data"])

        # Verify event structure
        assert payload["type"] == "terminal"
        assert payload["actor"] == "TestPlayer"
        assert payload["actor_id"] == "#99"
        assert payload["actor_type"] == "player"
        assert payload["room_id"] == "#42"
        assert payload["room_name"] == "Test Room"
        assert "[test_terminal] py_exec:" in payload["content"]
        assert "hello" in payload["content"]

    @patch("andimud.typeclasses.terminals.redis.from_url")
    @patch("andimud.typeclasses.terminals.append_mud_event")
    def test_terminal_event_contains_metadata(self, mock_append, mock_redis, mock_terminal, mock_caller):
        """Test that terminal events include proper metadata."""
        from andimud.typeclasses.terminals import Terminal

        Terminal._publish_terminal_event(mock_terminal, "bash_exec", "$ ls\nfile1.txt", caller=mock_caller)

        # Get the event payload
        call_args = mock_append.call_args[0]
        payload_wrapper = call_args[1]
        payload = json.loads(payload_wrapper["data"])

        # Verify metadata
        assert "metadata" in payload
        metadata = payload["metadata"]
        assert metadata["terminal_id"] == "#123"
        assert metadata["terminal_name"] == "test_terminal"
        assert metadata["tool_name"] == "bash_exec"
        assert metadata["terminal_type"] == "CodeTerminal"
        assert metadata["aura_name"] == "CODE_ACCESS"

    @patch("andimud.typeclasses.terminals.redis.from_url")
    @patch("andimud.typeclasses.terminals.append_mud_event")
    def test_terminal_event_truncates_long_output(self, mock_append, mock_redis, mock_terminal, mock_caller):
        """Test that terminal events truncate long output for events."""
        from andimud.typeclasses.terminals import Terminal

        # Create output longer than 2048 chars
        long_output = "x" * 3000

        Terminal._publish_terminal_event(mock_terminal, "py_exec", long_output, caller=mock_caller)

        # Get the event payload
        call_args = mock_append.call_args[0]
        payload_wrapper = call_args[1]
        payload = json.loads(payload_wrapper["data"])

        # Verify content is truncated
        assert len(payload["content"]) < 3000
        assert "(output truncated)" in payload["content"]

    @patch("andimud.typeclasses.terminals.redis.from_url")
    @patch("andimud.typeclasses.terminals.append_mud_event")
    def test_terminal_event_handles_no_caller(self, mock_append, mock_redis, mock_terminal):
        """Test that terminal events work when caller is None (system execution)."""
        from andimud.typeclasses.terminals import Terminal

        Terminal._publish_terminal_event(mock_terminal, "py_exec", "output", caller=None)

        # Get the event payload
        call_args = mock_append.call_args[0]
        payload_wrapper = call_args[1]
        payload = json.loads(payload_wrapper["data"])

        # Verify system actor
        assert payload["actor"] == "system"
        assert payload["actor_id"] == ""
        assert payload["actor_type"] == "system"

    @patch("andimud.typeclasses.terminals.redis.from_url")
    def test_terminal_event_fails_silently_on_redis_error(self, mock_redis, mock_terminal, mock_caller):
        """Test that Redis errors don't break terminal display."""
        from andimud.typeclasses.terminals import Terminal

        # Make Redis raise an exception
        mock_redis.side_effect = Exception("Redis connection failed")

        # This should not raise an exception
        Terminal.display_output(mock_terminal, "py_exec", "output", caller=mock_caller)

        # Verify room message was still sent twice (narrative + output)
        assert mock_terminal.location.msg_contents.call_count == 2
        # First call: narrative action
        assert "types away at" in mock_terminal.location.msg_contents.call_args_list[0][0][0]
        # Second call: actual output
        assert "py_exec:" in mock_terminal.location.msg_contents.call_args_list[1][0][0]

    @patch("andimud.typeclasses.terminals.append_mud_event")
    def test_terminal_event_not_published_without_location(self, mock_append):
        """Test that events are not published if terminal has no location."""
        from andimud.typeclasses.terminals import Terminal

        # Create a mock terminal without a location
        terminal = MagicMock()
        terminal.key = "test_terminal"
        terminal.dbref = "#123"
        terminal.location = None  # No location
        terminal.db = MagicMock()
        terminal.db.max_output_len = 4096

        Terminal.display_output(terminal, "py_exec", "output", caller=None)

        # Event should not be published
        mock_append.assert_not_called()


class TestTerminalCommandMetadataArgs:
    """Tests for terminal commands reading args from metadata.

    When ActionConsumer executes a command, it passes structured args
    in pending_action_metadata. This bypasses Evennia's command parser
    which mangles special characters like "/" (treated as switch separator).
    """

    @pytest.fixture
    def mock_caller(self):
        """Create a mock caller with ndb for metadata."""
        caller = MagicMock()
        caller.msg = MagicMock()
        caller.ndb = MagicMock()
        caller.ndb.pending_action_metadata = None
        caller.location = MagicMock()
        caller.location.contents = []
        return caller

    @pytest.fixture
    def mock_terminal(self):
        """Create a mock terminal for finding in room."""
        terminal = MagicMock()
        terminal.display_output = MagicMock()
        terminal.db = MagicMock()
        terminal.db.aura_name = "market"
        return terminal

    def test_stock_quote_reads_symbol_from_metadata(self, mock_caller, mock_terminal):
        """Test CmdStockQuote reads symbol from metadata args."""
        from andimud.commands.mud.object_commands.terminal import CmdStockQuote

        # Set up metadata with symbol containing /
        mock_caller.ndb.pending_action_metadata = {
            "args": {"symbol": "USD/ZAR"},
            "tool": "stock_quote",
        }
        mock_caller.location.contents = [mock_terminal]

        cmd = CmdStockQuote()
        cmd.caller = mock_caller
        cmd.args = "USD"  # This would be mangled by Evennia parser

        with patch("aim.tool.impl.container_mcp.ContainerMCPTool") as mock_tool_cls:
            mock_tool = MagicMock()
            mock_tool.execute.return_value = {
                "symbol": "USD/ZAR",
                "name": "US Dollar / South African Rand",
                "price": 18.5,
                "change": 0.1,
                "change_percent": 0.5,
            }
            mock_tool_cls.return_value = mock_tool

            cmd.func()

            # Verify execute was called with full symbol from metadata, not mangled args
            mock_tool.execute.assert_called_once_with("stock_quote", {"symbol": "USD/ZAR"})

    def test_stock_quote_fallback_to_self_args(self, mock_caller, mock_terminal):
        """Test CmdStockQuote falls back to self.args when no metadata."""
        from andimud.commands.mud.object_commands.terminal import CmdStockQuote

        # No metadata (manual command invocation)
        mock_caller.ndb.pending_action_metadata = None
        mock_caller.location.contents = [mock_terminal]

        cmd = CmdStockQuote()
        cmd.caller = mock_caller
        cmd.args = "AAPL"

        with patch("aim.tool.impl.container_mcp.ContainerMCPTool") as mock_tool_cls:
            mock_tool = MagicMock()
            mock_tool.execute.return_value = {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "price": 175.0,
                "change": 1.5,
                "change_percent": 0.9,
            }
            mock_tool_cls.return_value = mock_tool

            cmd.func()

            # Verify execute was called with self.args as fallback
            mock_tool.execute.assert_called_once_with("stock_quote", {"symbol": "AAPL"})

    def test_visit_webpage_reads_url_from_metadata(self, mock_caller):
        """Test CmdVisitWebpage reads url from metadata args."""
        from andimud.commands.mud.object_commands.terminal import CmdVisitWebpage

        # Set up web terminal
        mock_terminal = MagicMock()
        mock_terminal.display_output = MagicMock()
        mock_terminal.db = MagicMock()
        mock_terminal.db.aura_name = "web"

        # Set up metadata with URL containing /
        mock_caller.ndb.pending_action_metadata = {
            "args": {"url": "https://example.com/path/to/page"},
            "tool": "visit_webpage",
        }
        mock_caller.location.contents = [mock_terminal]

        cmd = CmdVisitWebpage()
        cmd.caller = mock_caller
        cmd.args = "https:"  # Would be mangled

        with patch("aim.tool.impl.container_mcp.ContainerMCPTool") as mock_tool_cls:
            mock_tool = MagicMock()
            mock_tool.execute.return_value = {
                "title": "Example Page",
                "content": "Page content here",
            }
            mock_tool_cls.return_value = mock_tool

            cmd.func()

            # Verify execute was called with full URL from metadata
            mock_tool.execute.assert_called_once_with(
                "visit_webpage",
                {"url": "https://example.com/path/to/page"}
            )

    def test_py_exec_reads_code_from_metadata(self, mock_caller):
        """Test CmdPyExec reads code from metadata args."""
        from andimud.commands.mud.object_commands.terminal import CmdPyExec

        # Set up code terminal
        mock_terminal = MagicMock()
        mock_terminal.display_output = MagicMock()
        mock_terminal.db = MagicMock()
        mock_terminal.db.aura_name = "code"

        # Set up metadata with code containing /
        mock_caller.ndb.pending_action_metadata = {
            "args": {"code": "print(10/2)"},
            "tool": "py_exec",
        }
        mock_caller.location.contents = [mock_terminal]

        cmd = CmdPyExec()
        cmd.caller = mock_caller
        cmd.args = "print(10"  # Would be mangled

        with patch("aim.tool.impl.container_mcp.ContainerMCPTool") as mock_tool_cls:
            mock_tool = MagicMock()
            mock_tool.execute.return_value = {
                "output": "5.0",
                "result": 5.0,
            }
            mock_tool_cls.return_value = mock_tool

            cmd.func()

            # Verify execute was called with full code from metadata
            mock_tool.execute.assert_called_once_with("py_exec", {"code": "print(10/2)"})

    def test_bash_exec_reads_command_from_metadata(self, mock_caller):
        """Test CmdBashExec reads command from metadata args."""
        from andimud.commands.mud.object_commands.terminal import CmdBashExec

        # Set up code terminal
        mock_terminal = MagicMock()
        mock_terminal.display_output = MagicMock()
        mock_terminal.db = MagicMock()
        mock_terminal.db.aura_name = "code"

        # Set up metadata with command containing /
        mock_caller.ndb.pending_action_metadata = {
            "args": {"command": "ls /home/user"},
            "tool": "bash_exec",
        }
        mock_caller.location.contents = [mock_terminal]

        cmd = CmdBashExec()
        cmd.caller = mock_caller
        cmd.args = "ls"  # Would be mangled

        with patch("aim.tool.impl.container_mcp.ContainerMCPTool") as mock_tool_cls:
            mock_tool = MagicMock()
            mock_tool.execute.return_value = {
                "stdout": "file1.txt\nfile2.txt",
                "stderr": "",
                "exit_code": 0,
            }
            mock_tool_cls.return_value = mock_tool

            cmd.func()

            # Verify execute was called with full command from metadata
            mock_tool.execute.assert_called_once_with("bash_exec", {"command": "ls /home/user"})

    def test_web_search_reads_query_from_metadata(self, mock_caller):
        """Test CmdWebSearch reads query from metadata args."""
        from andimud.commands.mud.object_commands.terminal import CmdWebSearch

        # Set up web terminal
        mock_terminal = MagicMock()
        mock_terminal.display_output = MagicMock()
        mock_terminal.db = MagicMock()
        mock_terminal.db.aura_name = "web"

        # Set up metadata with query containing /
        mock_caller.ndb.pending_action_metadata = {
            "args": {"query": "site:reddit.com/r/python asyncio"},
            "tool": "web_search",
        }
        mock_caller.location.contents = [mock_terminal]

        cmd = CmdWebSearch()
        cmd.caller = mock_caller
        cmd.args = "site:reddit.com"  # Would be mangled

        with patch("aim.tool.impl.container_mcp.ContainerMCPTool") as mock_tool_cls:
            mock_tool = MagicMock()
            mock_tool.execute.return_value = {
                "results": [{"title": "Python Asyncio", "url": "https://example.com"}],
            }
            mock_tool_cls.return_value = mock_tool

            cmd.func()

            # Verify execute was called with full query from metadata
            mock_tool.execute.assert_called_once_with(
                "web_search",
                {"query": "site:reddit.com/r/python asyncio"}
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
