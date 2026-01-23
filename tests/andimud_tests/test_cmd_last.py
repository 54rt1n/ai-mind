# tests/unit/mud/test_cmd_last.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for CmdLast command."""

import json
import pytest
from unittest.mock import MagicMock, patch, Mock
from aim_mud_types import RedisKeys


class TestCmdLastParsing:
    """Tests for argument parsing in CmdLast."""

    def test_parse_no_args(self):
        """Test command with no arguments shows usage."""
        from commands.mud.actions import CmdLast

        cmd = CmdLast()
        cmd.caller = MagicMock()
        cmd.args = ""

        cmd.func()

        cmd.caller.msg.assert_called_once_with("Usage: @last <agent_id> [= <count>]")

    def test_parse_agent_only(self):
        """Test command with just agent_id defaults to count=1."""
        from commands.mud.actions import CmdLast

        cmd = CmdLast()
        cmd.caller = MagicMock()
        cmd.args = "  andi  "

        with patch('redis.from_url') as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.llen.return_value = 0

            cmd.func()

            # Should call llen with agent_conversation key
            mock_r.llen.assert_called_once()
            call_args = mock_r.llen.call_args[0][0]
            assert 'andi' in call_args

    def test_parse_agent_with_count(self):
        """Test command with agent_id = count."""
        from commands.mud.actions import CmdLast

        cmd = CmdLast()
        cmd.caller = MagicMock()
        cmd.args = "andi = 5"

        with patch('redis.from_url') as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.llen.return_value = 0

            cmd.func()

            # Should request count=5 entries
            mock_r.llen.assert_called_once()

    def test_parse_agent_with_all(self):
        """Test command with agent_id = all."""
        from commands.mud.actions import CmdLast

        cmd = CmdLast()
        cmd.caller = MagicMock()
        cmd.args = "andi = all"

        with patch('redis.from_url') as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.llen.return_value = 0

            cmd.func()

            # Should handle all entries
            mock_r.llen.assert_called_once()

    def test_parse_invalid_count_zero(self):
        """Test command with zero count shows error."""
        from commands.mud.actions import CmdLast

        cmd = CmdLast()
        cmd.caller = MagicMock()
        cmd.args = "andi = 0"

        cmd.func()

        cmd.caller.msg.assert_called_once_with("|rCount must be a positive number|n")

    def test_parse_invalid_count_negative(self):
        """Test command with negative count shows error."""
        from commands.mud.actions import CmdLast

        cmd = CmdLast()
        cmd.caller = MagicMock()
        cmd.args = "andi = -5"

        cmd.func()

        cmd.caller.msg.assert_called_once_with("|rCount must be a positive number|n")

    def test_parse_invalid_count_text(self):
        """Test command with invalid text count shows error."""
        from commands.mud.actions import CmdLast

        cmd = CmdLast()
        cmd.caller = MagicMock()
        cmd.args = "andi = xyz"

        cmd.func()

        cmd.caller.msg.assert_called_once_with("|rInvalid count. Use a number or 'all'|n")


class TestCmdLastRedisInteraction:
    """Tests for Redis interaction in CmdLast."""

    def test_no_conversation_history(self):
        """Test command when agent has no conversation history."""
        from commands.mud.actions import CmdLast

        cmd = CmdLast()
        cmd.caller = MagicMock()
        cmd.args = "andi"

        with patch('redis.from_url') as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.llen.return_value = 0

            cmd.func()

            cmd.caller.msg.assert_called_once_with("No conversation history found for andi.")

    def test_fetch_single_entry(self):
        """Test fetching single entry uses correct Redis commands."""
        from commands.mud.actions import CmdLast

        cmd = CmdLast()
        cmd.caller = MagicMock()
        cmd.args = "andi"

        entry = {
            "role": "assistant",
            "content": "Hello world",
            "tokens": 10,
            "saved": True,
            "speaker_id": "andi",
            "timestamp": "2026-01-09T12:00:00"
        }

        with patch('redis.from_url') as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.llen.return_value = 5
            mock_r.lrange.return_value = [json.dumps(entry).encode('utf-8')]

            cmd.func()

            # Should call lrange for last 1 entry
            mock_r.lrange.assert_called_once()
            call_args = mock_r.lrange.call_args[0]
            assert call_args[1] == -1  # Start at -1
            assert call_args[2] == -1  # End at -1

    def test_fetch_multiple_entries(self):
        """Test fetching multiple entries uses correct Redis commands."""
        from commands.mud.actions import CmdLast

        cmd = CmdLast()
        cmd.caller = MagicMock()
        cmd.args = "andi = 3"

        entries = [
            {
                "role": "assistant",
                "content": f"Message {i}",
                "tokens": 10,
                "saved": True,
                "speaker_id": "andi",
                "timestamp": f"2026-01-09T12:00:0{i}"
            }
            for i in range(3)
        ]

        with patch('redis.from_url') as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.llen.return_value = 10
            mock_r.lrange.return_value = [json.dumps(e).encode('utf-8') for e in entries]

            cmd.func()

            # Should call lrange for last 3 entries
            mock_r.lrange.assert_called_once()
            call_args = mock_r.lrange.call_args[0]
            assert call_args[1] == -3  # Start at -3
            assert call_args[2] == -1  # End at -1

    def test_fetch_all_entries(self):
        """Test fetching all entries uses correct Redis commands."""
        from commands.mud.actions import CmdLast

        cmd = CmdLast()
        cmd.caller = MagicMock()
        cmd.args = "andi = all"

        entries = [
            {
                "role": "assistant",
                "content": f"Message {i}",
                "tokens": 10,
                "saved": True,
                "speaker_id": "andi",
                "timestamp": f"2026-01-09T12:00:0{i}"
            }
            for i in range(5)
        ]

        with patch('redis.from_url') as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.llen.return_value = 5
            mock_r.lrange.return_value = [json.dumps(e).encode('utf-8') for e in entries]

            cmd.func()

            # Should call lrange for all entries (0, -1)
            mock_r.lrange.assert_called_once()
            call_args = mock_r.lrange.call_args[0]
            assert call_args[1] == 0  # Start at 0
            assert call_args[2] == -1  # End at -1

    def test_count_exceeds_total(self):
        """Test requesting more entries than exist returns all available."""
        from commands.mud.actions import CmdLast

        cmd = CmdLast()
        cmd.caller = MagicMock()
        cmd.args = "andi = 100"

        entries = [
            {
                "role": "assistant",
                "content": f"Message {i}",
                "tokens": 10,
                "saved": True,
                "speaker_id": "andi",
                "timestamp": f"2026-01-09T12:00:0{i}"
            }
            for i in range(5)
        ]

        with patch('redis.from_url') as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.llen.return_value = 5
            mock_r.lrange.return_value = [json.dumps(e).encode('utf-8') for e in entries]

            cmd.func()

            # Should only fetch 5 entries (total available)
            # When entries_to_fetch == total_entries, uses lrange(key, 0, -1)
            mock_r.lrange.assert_called_once()
            call_args = mock_r.lrange.call_args[0]
            assert call_args[1] == 0  # Start at 0 (all entries)
            assert call_args[2] == -1  # End at -1


class TestCmdLastOutputFormatting:
    """Tests for output formatting in CmdLast."""

    def test_single_entry_full_format(self):
        """Test single entry uses full format (backward compatible)."""
        from commands.mud.actions import CmdLast

        cmd = CmdLast()
        cmd.caller = MagicMock()
        cmd.args = "andi"

        entry = {
            "role": "assistant",
            "content": "Hello world",
            "tokens": 10,
            "saved": True,
            "speaker_id": "andi",
            "timestamp": "2026-01-09T12:00:00"
        }

        with patch('redis.from_url') as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.llen.return_value = 1
            mock_r.lrange.return_value = [json.dumps(entry).encode('utf-8')]

            cmd.func()

            # Should use full format
            call_arg = cmd.caller.msg.call_args[0][0]
            assert "=== Last Message for andi ===" in call_arg
            assert "|cRole:|n assistant" in call_arg
            assert "|cSpeaker:|n andi" in call_arg
            assert "|cTimestamp:|n 2026-01-09T12:00:00" in call_arg
            assert "|cTokens:|n 10" in call_arg
            assert "|cSaved to CVM:|n True" in call_arg
            assert "|gContent:|n" in call_arg
            assert "Hello world" in call_arg

    def test_single_entry_with_think_tags(self):
        """Test single entry includes think tags when present."""
        from commands.mud.actions import CmdLast

        cmd = CmdLast()
        cmd.caller = MagicMock()
        cmd.args = "andi"

        entry = {
            "role": "assistant",
            "content": "Hello world",
            "tokens": 10,
            "saved": True,
            "speaker_id": "andi",
            "timestamp": "2026-01-09T12:00:00",
            "think": "I should greet them warmly"
        }

        with patch('redis.from_url') as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.llen.return_value = 1
            mock_r.lrange.return_value = [json.dumps(entry).encode('utf-8')]

            cmd.func()

            call_arg = cmd.caller.msg.call_args[0][0]
            assert "|y<think>|n" in call_arg
            assert "I should greet them warmly" in call_arg
            assert "|y</think>|n" in call_arg

    def test_multiple_entries_condensed_format(self):
        """Test multiple entries use condensed format."""
        from commands.mud.actions import CmdLast

        cmd = CmdLast()
        cmd.caller = MagicMock()
        cmd.args = "andi = 3"

        entries = [
            {
                "role": "assistant",
                "content": f"Message {i}",
                "tokens": 10,
                "saved": True,
                "speaker_id": "andi",
                "timestamp": f"2026-01-09T12:00:0{i}"
            }
            for i in range(3)
        ]

        with patch('redis.from_url') as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.llen.return_value = 10
            # Return in chronological order (Redis behavior)
            mock_r.lrange.return_value = [json.dumps(e).encode('utf-8') for e in entries]

            cmd.func()

            call_arg = cmd.caller.msg.call_args[0][0]
            # Should use condensed format
            assert "=== Last 3 Messages for andi ===" in call_arg
            assert "[1/3]" in call_arg
            assert "[2/3]" in call_arg
            assert "[3/3]" in call_arg
            # Should show footer
            assert "Showing 3 of 10 total entries" in call_arg

    def test_multiple_entries_oldest_first(self):
        """Test multiple entries are shown in chronological order (oldest first)."""
        from commands.mud.actions import CmdLast

        cmd = CmdLast()
        cmd.caller = MagicMock()
        cmd.args = "andi = 3"

        # Create entries with timestamps showing order
        entries = [
            {
                "role": "assistant",
                "content": "First message (oldest)",
                "tokens": 10,
                "saved": True,
                "speaker_id": "andi",
                "timestamp": "2026-01-09T12:00:00"
            },
            {
                "role": "assistant",
                "content": "Second message",
                "tokens": 10,
                "saved": True,
                "speaker_id": "andi",
                "timestamp": "2026-01-09T12:01:00"
            },
            {
                "role": "assistant",
                "content": "Third message (newest)",
                "tokens": 10,
                "saved": True,
                "speaker_id": "andi",
                "timestamp": "2026-01-09T12:02:00"
            }
        ]

        with patch('redis.from_url') as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.llen.return_value = 3
            # Redis returns in chronological order (oldest to newest)
            mock_r.lrange.return_value = [json.dumps(e).encode('utf-8') for e in entries]

            cmd.func()

            call_arg = cmd.caller.msg.call_args[0][0]
            # Find positions of each message
            pos_first = call_arg.find("First message (oldest)")
            pos_second = call_arg.find("Second message")
            pos_newest = call_arg.find("Third message (newest)")

            # Messages should appear in chronological order (oldest first)
            assert pos_first < pos_second < pos_newest

    def test_long_content_shown_fully_in_condensed(self):
        """Test long content is shown fully in condensed format (no truncation)."""
        from commands.mud.actions import CmdLast

        cmd = CmdLast()
        cmd.caller = MagicMock()
        cmd.args = "andi = 2"

        long_content = "x" * 500  # 500 chars, now shown in full

        entries = [
            {
                "role": "assistant",
                "content": long_content,
                "tokens": 100,
                "saved": True,
                "speaker_id": "andi",
                "timestamp": "2026-01-09T12:00:00"
            },
            {
                "role": "assistant",
                "content": "Short message",
                "tokens": 10,
                "saved": True,
                "speaker_id": "andi",
                "timestamp": "2026-01-09T12:01:00"
            }
        ]

        with patch('redis.from_url') as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.llen.return_value = 2
            mock_r.lrange.return_value = [json.dumps(e).encode('utf-8') for e in entries]

            cmd.func()

            call_arg = cmd.caller.msg.call_args[0][0]
            # Should NOT show truncation notice (no truncation anymore)
            assert "[500 chars total]" not in call_arg
            # Should contain full content
            assert long_content in call_arg

    def test_long_think_shown_fully_in_condensed(self):
        """Test long think tags are shown fully in condensed format (no truncation)."""
        from commands.mud.actions import CmdLast

        cmd = CmdLast()
        cmd.caller = MagicMock()
        cmd.args = "andi = 2"

        long_think = "y" * 300  # 300 chars, now shown in full

        entries = [
            {
                "role": "assistant",
                "content": "Message",
                "tokens": 10,
                "saved": True,
                "speaker_id": "andi",
                "timestamp": "2026-01-09T12:00:00",
                "think": long_think
            },
            {
                "role": "assistant",
                "content": "Another message",
                "tokens": 10,
                "saved": True,
                "speaker_id": "andi",
                "timestamp": "2026-01-09T12:01:00"
            }
        ]

        with patch('redis.from_url') as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.llen.return_value = 2
            mock_r.lrange.return_value = [json.dumps(e).encode('utf-8') for e in entries]

            cmd.func()

            call_arg = cmd.caller.msg.call_args[0][0]
            # Should NOT show truncation notice (no truncation anymore)
            assert "[truncated]" not in call_arg
            # Should contain full think content
            assert long_think in call_arg

    def test_warning_for_large_requests(self):
        """Test warning shown when fetching more than 20 entries."""
        from commands.mud.actions import CmdLast

        cmd = CmdLast()
        cmd.caller = MagicMock()
        cmd.args = "andi = 30"

        entries = [
            {
                "role": "assistant",
                "content": f"Message {i}",
                "tokens": 10,
                "saved": True,
                "speaker_id": "andi",
                "timestamp": f"2026-01-09T12:00:{i:02d}"
            }
            for i in range(30)
        ]

        with patch('redis.from_url') as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.llen.return_value = 30
            mock_r.lrange.return_value = [json.dumps(e).encode('utf-8') for e in entries]

            cmd.func()

            # Should have called msg twice: once for warning, once for output
            assert cmd.caller.msg.call_count == 2
            warning_call = cmd.caller.msg.call_args_list[0][0][0]
            assert "30 entries" in warning_call
            assert "output may be long" in warning_call.lower()


class TestCmdLastErrorHandling:
    """Tests for error handling in CmdLast."""

    def test_json_decode_error(self):
        """Test handling of JSON decode errors."""
        from commands.mud.actions import CmdLast

        cmd = CmdLast()
        cmd.caller = MagicMock()
        cmd.args = "andi"

        with patch('redis.from_url') as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.llen.return_value = 1
            # Return invalid JSON
            mock_r.lrange.return_value = [b"invalid json {"]

            with patch('evennia.utils.logger.log_err') as mock_log:
                cmd.func()

                # Should have logged error
                mock_log.assert_called()

    def test_redis_connection_error(self):
        """Test handling of Redis connection errors."""
        from commands.mud.actions import CmdLast

        cmd = CmdLast()
        cmd.caller = MagicMock()
        cmd.args = "andi"

        with patch('redis.from_url') as mock_redis:
            mock_redis.side_effect = Exception("Connection failed")

            with patch('evennia.utils.logger.log_err') as mock_log:
                cmd.func()

                # Should show error to user
                cmd.caller.msg.assert_called()
                call_arg = cmd.caller.msg.call_args[0][0]
                assert "failed" in call_arg.lower()

                # Should log error
                mock_log.assert_called()
