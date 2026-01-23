# tests/andimud_tests/test_status_commands.py
"""Tests for @turns and @events commands.

These tests verify the new status commands that provide visibility into
turn coordination and event queues for AI agents in ANDIMUD.
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call
import redis


# Import commands and helper functions from the Evennia codebase
import sys
from pathlib import Path

# Add andimud to path for imports
andimud_path = Path(__file__).parent.parent / "andimud" / "andimud"
if andimud_path.exists():
    sys.path.insert(0, str(andimud_path.parent))

# Try to import commands - skip all tests if not available or Django not configured
EVENNIA_AVAILABLE = False
CmdTurns = None
CmdEvents = None
format_time_ago = None
is_agent_online = None
parse_stream_timestamp = None
get_hash_field = None

try:
    from commands.mud.status import (
        CmdTurns,
        CmdEvents,
    )
    from aim_mud_types import (
        format_time_ago,
        is_agent_online,
        parse_stream_timestamp,
        get_hash_field,
    )
    EVENNIA_AVAILABLE = True
except Exception:
    # Skip all tests if Evennia is not available or Django is not configured
    pass

if not EVENNIA_AVAILABLE:
    pytestmark = pytest.mark.skip(
        reason="Evennia status commands not available or Django not configured"
    )


@pytest.mark.skipif(not EVENNIA_AVAILABLE, reason="Evennia not available")
class TestHelperFunctions:
    """Tests for helper functions used by @turns and @events commands."""

    # Test _get_field function

    def test_get_field_bytes_key(self):
        """Test _get_field handles bytes keys from Redis."""
        data = {b"status": b"ready", b"sequence_id": b"42"}
        assert get_hash_field(data, "status") == "ready"
        assert get_hash_field(data, "sequence_id") == "42"

    def test_get_field_string_key(self):
        """Test _get_field handles string keys."""
        data = {"status": "ready", "sequence_id": "42"}
        assert get_hash_field(data, "status") == "ready"
        assert get_hash_field(data, "sequence_id") == "42"

    def test_get_field_mixed_keys(self):
        """Test _get_field handles mixed bytes/string keys."""
        data = {b"status": b"ready", "sequence_id": "42"}
        assert get_hash_field(data, "status") == "ready"
        assert get_hash_field(data, "sequence_id") == "42"

    def test_get_field_missing_key(self):
        """Test _get_field returns empty string for missing key."""
        data = {b"status": b"ready"}
        assert get_hash_field(data, "missing") == ""

    def test_get_field_empty_dict(self):
        """Test _get_field handles empty dict."""
        data = {}
        assert get_hash_field(data, "anything") == ""

    # Test _format_time_ago function

    def test_format_time_ago_seconds(self):
        """Test format for less than 1 minute ago."""
        now = datetime.now()
        timestamp_45s_ago = (now - timedelta(seconds=45)).isoformat()
        result = format_time_ago(timestamp_45s_ago)
        assert result == "45s ago"

    def test_format_time_ago_zero_seconds(self):
        """Test format for zero seconds ago."""
        now = datetime.now()
        timestamp_now = now.isoformat()
        result = format_time_ago(timestamp_now)
        # Should be 0s ago or 1s ago depending on timing
        assert "s ago" in result

    def test_format_time_ago_minutes(self):
        """Test format for minutes and seconds."""
        now = datetime.now()
        timestamp_3m_25s_ago = (now - timedelta(minutes=3, seconds=25)).isoformat()
        result = format_time_ago(timestamp_3m_25s_ago)
        assert result == "3m 25s ago"

    def test_format_time_ago_minutes_no_seconds(self):
        """Test format for exact minutes (no seconds)."""
        now = datetime.now()
        timestamp_5m_ago = (now - timedelta(minutes=5)).isoformat()
        result = format_time_ago(timestamp_5m_ago)
        # Should be either "5m ago" or "5m 0s ago" depending on exact timing
        assert "5m" in result

    def test_format_time_ago_hours(self):
        """Test format for hours and minutes."""
        now = datetime.now()
        timestamp_2h_30m_ago = (now - timedelta(hours=2, minutes=30)).isoformat()
        result = format_time_ago(timestamp_2h_30m_ago)
        assert result == "2h 30m ago"

    def test_format_time_ago_hours_no_minutes(self):
        """Test format for exact hours (no minutes)."""
        now = datetime.now()
        timestamp_3h_ago = (now - timedelta(hours=3)).isoformat()
        result = format_time_ago(timestamp_3h_ago)
        # Should be "3h ago" or "3h 0m ago" depending on exact timing
        assert "3h" in result

    def test_format_time_ago_days(self):
        """Test format for days and hours."""
        now = datetime.now()
        timestamp_2d_5h_ago = (now - timedelta(days=2, hours=5)).isoformat()
        result = format_time_ago(timestamp_2d_5h_ago)
        assert result == "2d 5h ago"

    def test_format_time_ago_days_no_hours(self):
        """Test format for exact days (no hours)."""
        now = datetime.now()
        timestamp_7d_ago = (now - timedelta(days=7)).isoformat()
        result = format_time_ago(timestamp_7d_ago)
        # Should be "7d ago" or "7d 0h ago" depending on exact timing
        assert "7d" in result

    def test_format_time_ago_future_timestamp(self):
        """Test format for future timestamp."""
        now = datetime.now()
        timestamp_future = (now + timedelta(hours=1)).isoformat()
        result = format_time_ago(timestamp_future)
        assert result == "in the future"

    def test_format_time_ago_invalid_timestamp(self):
        """Test format for invalid timestamp string."""
        result = format_time_ago("not-a-timestamp")
        assert result == "unknown"

    def test_format_time_ago_empty_string(self):
        """Test format for empty string."""
        result = format_time_ago("")
        assert result == "unknown"

    # Test _is_agent_online function

    def test_is_agent_online_fresh_heartbeat(self):
        """Test agent is online with fresh heartbeat."""
        now = datetime.now()
        heartbeat_2m_ago = (now - timedelta(minutes=2)).isoformat()
        assert is_agent_online(heartbeat_2m_ago, threshold_minutes=5) is True

    def test_is_agent_online_at_threshold(self):
        """Test agent at exact threshold boundary."""
        now = datetime.now()
        heartbeat_5m_ago = (now - timedelta(minutes=5)).isoformat()
        # Should be False (>= threshold is offline)
        assert is_agent_online(heartbeat_5m_ago, threshold_minutes=5) is False

    def test_is_agent_online_stale_heartbeat(self):
        """Test agent is offline with stale heartbeat."""
        now = datetime.now()
        heartbeat_10m_ago = (now - timedelta(minutes=10)).isoformat()
        assert is_agent_online(heartbeat_10m_ago, threshold_minutes=5) is False

    def test_is_agent_online_custom_threshold(self):
        """Test agent online status with custom threshold."""
        now = datetime.now()
        heartbeat_8m_ago = (now - timedelta(minutes=8)).isoformat()
        assert is_agent_online(heartbeat_8m_ago, threshold_minutes=10) is True
        assert is_agent_online(heartbeat_8m_ago, threshold_minutes=5) is False

    def test_is_agent_online_invalid_timestamp(self):
        """Test invalid timestamp returns False."""
        assert is_agent_online("not-a-timestamp") is False

    def test_is_agent_online_empty_string(self):
        """Test empty string returns False."""
        assert is_agent_online("") is False

    # Test _parse_stream_timestamp function

    def test_parse_stream_timestamp_valid_id(self):
        """Test parsing valid Redis stream ID."""
        # 1736423400000 = 2025-01-09 10:30:00 UTC
        stream_id = "1736423400000-0"
        result = parse_stream_timestamp(stream_id)
        assert isinstance(result, datetime)
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 9

    def test_parse_stream_timestamp_with_sequence(self):
        """Test parsing stream ID with non-zero sequence."""
        stream_id = "1736423400000-42"
        result = parse_stream_timestamp(stream_id)
        # Sequence number shouldn't affect timestamp
        assert isinstance(result, datetime)
        assert result.year == 2025

    def test_parse_stream_timestamp_invalid_format(self):
        """Test parsing invalid stream ID format."""
        stream_id = "invalid-id"
        result = parse_stream_timestamp(stream_id)
        # Should return current time on error
        assert isinstance(result, datetime)
        assert abs((datetime.now() - result).total_seconds()) < 2

    def test_parse_stream_timestamp_missing_dash(self):
        """Test parsing stream ID without dash separator still works.

        Note: Even without a dash, split('-')[0] succeeds and returns the
        full string, so this actually parses correctly.
        """
        stream_id = "1736423400000"
        result = parse_stream_timestamp(stream_id)
        # Should parse successfully even without dash
        assert isinstance(result, datetime)
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 9

    def test_parse_stream_timestamp_empty_string(self):
        """Test parsing empty string."""
        stream_id = ""
        result = parse_stream_timestamp(stream_id)
        # Should return current time on error
        assert isinstance(result, datetime)
        assert abs((datetime.now() - result).total_seconds()) < 2


@pytest.mark.skipif(not EVENNIA_AVAILABLE, reason="Evennia not available")
class TestCmdTurns:
    """Tests for the @turns command."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        return MagicMock(spec=redis.Redis)

    @pytest.fixture
    def mock_caller(self):
        """Create a mock command caller."""
        caller = MagicMock()
        caller.msg = MagicMock()
        return caller

    @pytest.fixture
    def cmd_turns(self, mock_caller):
        """Create a CmdTurns instance with mocked caller."""
        cmd = CmdTurns()
        cmd.caller = mock_caller
        return cmd

    def test_cmd_turns_no_agents(self, cmd_turns, mock_redis):
        """Test @turns with no turn requests in Redis."""
        mock_redis.keys.return_value = []

        with patch("redis.from_url", return_value=mock_redis):
            cmd_turns.func()

        # Should display message about no agents
        cmd_turns.caller.msg.assert_called_once()
        output = cmd_turns.caller.msg.call_args[0][0]
        assert "No agents with turn requests found" in output

    def test_cmd_turns_single_online_agent(self, cmd_turns, mock_redis):
        """Test @turns with one online agent."""
        now = datetime.now()
        heartbeat = (now - timedelta(minutes=1)).isoformat()

        mock_redis.keys.return_value = [b"agent:andi:turn_request"]
        mock_redis.hgetall.return_value = {
            b"status": b"ready",
            b"sequence_id": b"42",
            b"heartbeat_at": heartbeat.encode(),
            b"reason": b"events",
        }

        with patch("redis.from_url", return_value=mock_redis):
            cmd_turns.func()

        cmd_turns.caller.msg.assert_called_once()
        output = cmd_turns.caller.msg.call_args[0][0]
        assert "andi" in output
        assert "ready" in output
        assert "seq=42" in output
        assert "heartbeat=" in output

    def test_cmd_turns_multiple_agents_sorted(self, cmd_turns, mock_redis):
        """Test @turns with multiple agents sorted by sequence_id."""
        now = datetime.now()
        heartbeat = (now - timedelta(minutes=2)).isoformat()

        # Return keys in random order
        mock_redis.keys.return_value = [
            b"agent:charlie:turn_request",
            b"agent:andi:turn_request",
            b"agent:bob:turn_request",
        ]

        # Mock hgetall to return different sequence IDs
        def mock_hgetall(key):
            # Convert key to string for consistent checking
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key

            if "andi" in key_str:
                return {
                    b"status": b"ready",
                    b"sequence_id": b"10",
                    b"heartbeat_at": heartbeat.encode(),
                    b"reason": b"events",
                }
            elif "bob" in key_str:
                return {
                    b"status": b"in_progress",
                    b"sequence_id": b"5",
                    b"heartbeat_at": heartbeat.encode(),
                    b"reason": b"retry",
                }
            elif "charlie" in key_str:
                return {
                    b"status": b"assigned",
                    b"sequence_id": b"15",
                    b"heartbeat_at": heartbeat.encode(),
                    b"reason": b"idle",
                }

        mock_redis.hgetall.side_effect = mock_hgetall

        with patch("redis.from_url", return_value=mock_redis):
            cmd_turns.func()

        cmd_turns.caller.msg.assert_called_once()
        output = cmd_turns.caller.msg.call_args[0][0]

        # Find positions of agent names in output
        bob_pos = output.find("bob")
        andi_pos = output.find("andi")
        charlie_pos = output.find("charlie")

        # Verify they appear in sequence order (5 < 10 < 15)
        assert bob_pos < andi_pos < charlie_pos
        assert "seq=5" in output
        assert "seq=10" in output
        assert "seq=15" in output

    def test_cmd_turns_offline_agent(self, cmd_turns, mock_redis):
        """Test @turns with agent with stale heartbeat."""
        now = datetime.now()
        stale_heartbeat = (now - timedelta(minutes=10)).isoformat()

        mock_redis.keys.return_value = [b"agent:andi:turn_request"]
        mock_redis.hgetall.return_value = {
            b"status": b"ready",
            b"sequence_id": b"42",
            b"heartbeat_at": stale_heartbeat.encode(),
            b"reason": b"events",
        }

        with patch("redis.from_url", return_value=mock_redis):
            cmd_turns.func()

        cmd_turns.caller.msg.assert_called_once()
        output = cmd_turns.caller.msg.call_args[0][0]
        assert "Offline/Crashed Agents" in output
        assert "andi" in output
        assert "offline" in output

    def test_cmd_turns_mixed_online_offline(self, cmd_turns, mock_redis):
        """Test @turns with mix of online and offline agents."""
        now = datetime.now()
        fresh_heartbeat = (now - timedelta(minutes=1)).isoformat()
        stale_heartbeat = (now - timedelta(minutes=10)).isoformat()

        mock_redis.keys.return_value = [
            b"agent:andi:turn_request",
            b"agent:bob:turn_request",
        ]

        def mock_hgetall(key):
            # Convert key to string for consistent checking
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key

            if "andi" in key_str:
                return {
                    b"status": b"ready",
                    b"sequence_id": b"10",
                    b"heartbeat_at": fresh_heartbeat.encode(),
                    b"reason": b"events",
                }
            elif "bob" in key_str:
                return {
                    b"status": b"ready",
                    b"sequence_id": b"5",
                    b"heartbeat_at": stale_heartbeat.encode(),
                    b"reason": b"events",
                }

        mock_redis.hgetall.side_effect = mock_hgetall

        with patch("redis.from_url", return_value=mock_redis):
            cmd_turns.func()

        cmd_turns.caller.msg.assert_called_once()
        output = cmd_turns.caller.msg.call_args[0][0]
        assert "Online Agents" in output
        assert "Offline/Crashed Agents" in output
        # Andi should be online
        online_section = output.split("Offline/Crashed Agents")[0]
        assert "andi" in online_section
        # Bob should be offline
        offline_section = output.split("Offline/Crashed Agents")[1]
        assert "bob" in offline_section

    def test_cmd_turns_status_indicators(self, cmd_turns, mock_redis):
        """Test @turns displays correct status indicators."""
        now = datetime.now()
        heartbeat = (now - timedelta(minutes=1)).isoformat()

        mock_redis.keys.return_value = [
            b"agent:ready:turn_request",
            b"agent:progress:turn_request",
            b"agent:done:turn_request",
        ]

        def mock_hgetall(key):
            # Convert key to string for consistent checking
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key

            if "ready" in key_str:
                return {
                    b"status": b"ready",
                    b"sequence_id": b"1",
                    b"heartbeat_at": heartbeat.encode(),
                }
            elif "progress" in key_str:
                return {
                    b"status": b"in_progress",
                    b"sequence_id": b"2",
                    b"heartbeat_at": heartbeat.encode(),
                }
            elif "done" in key_str:
                return {
                    b"status": b"done",
                    b"sequence_id": b"3",
                    b"heartbeat_at": heartbeat.encode(),
                }

        mock_redis.hgetall.side_effect = mock_hgetall

        with patch("redis.from_url", return_value=mock_redis):
            cmd_turns.func()

        cmd_turns.caller.msg.assert_called_once()
        output = cmd_turns.caller.msg.call_args[0][0]
        # Check for status indicators (emojis may vary)
        assert "ready" in output
        assert "in_progress" in output or "progress" in output
        assert "done" in output

    def test_cmd_turns_with_status_reason(self, cmd_turns, mock_redis):
        """Test @turns displays status_reason when present."""
        now = datetime.now()
        heartbeat = (now - timedelta(minutes=1)).isoformat()

        mock_redis.keys.return_value = [b"agent:andi:turn_request"]
        mock_redis.hgetall.return_value = {
            b"status": b"fail",
            b"sequence_id": b"42",
            b"heartbeat_at": heartbeat.encode(),
            b"reason": b"events",
            b"status_reason": b"LLM timeout",
        }

        with patch("redis.from_url", return_value=mock_redis):
            cmd_turns.func()

        cmd_turns.caller.msg.assert_called_once()
        output = cmd_turns.caller.msg.call_args[0][0]
        assert "LLM timeout" in output

    def test_cmd_turns_paused_agent(self, cmd_turns, mock_redis):
        """Test @turns displays paused indicator for paused agents."""
        now = datetime.now()
        heartbeat = (now - timedelta(minutes=1)).isoformat()

        mock_redis.keys.return_value = [b"agent:andi:turn_request"]
        mock_redis.hgetall.return_value = {
            b"status": b"ready",
            b"sequence_id": b"42",
            b"heartbeat_at": heartbeat.encode(),
            b"reason": b"events",
        }
        # Mock pause flag is set
        mock_redis.get.return_value = b"1"
        # Mock sleep flag is not set
        mock_redis.hget.return_value = None

        with patch("redis.from_url", return_value=mock_redis):
            cmd_turns.func()

        cmd_turns.caller.msg.assert_called_once()
        output = cmd_turns.caller.msg.call_args[0][0]
        assert "paused" in output
        assert "sleeping" not in output

    def test_cmd_turns_sleeping_agent(self, cmd_turns, mock_redis):
        """Test @turns displays sleeping indicator for sleeping agents."""
        now = datetime.now()
        heartbeat = (now - timedelta(minutes=1)).isoformat()

        mock_redis.keys.return_value = [b"agent:andi:turn_request"]
        mock_redis.hgetall.return_value = {
            b"status": b"ready",
            b"sequence_id": b"42",
            b"heartbeat_at": heartbeat.encode(),
            b"reason": b"events",
        }
        # Mock pause flag is not set
        mock_redis.get.return_value = None
        # Mock sleep flag is set
        mock_redis.hget.return_value = b"true"

        with patch("redis.from_url", return_value=mock_redis):
            cmd_turns.func()

        cmd_turns.caller.msg.assert_called_once()
        output = cmd_turns.caller.msg.call_args[0][0]
        assert "sleeping" in output
        assert "paused" not in output

    def test_cmd_turns_paused_and_sleeping_agent(self, cmd_turns, mock_redis):
        """Test @turns displays both paused and sleeping indicators."""
        now = datetime.now()
        heartbeat = (now - timedelta(minutes=1)).isoformat()

        mock_redis.keys.return_value = [b"agent:andi:turn_request"]
        mock_redis.hgetall.return_value = {
            b"status": b"ready",
            b"sequence_id": b"42",
            b"heartbeat_at": heartbeat.encode(),
            b"reason": b"events",
        }
        # Mock both pause and sleep flags set
        mock_redis.get.return_value = b"1"
        mock_redis.hget.return_value = b"true"

        with patch("redis.from_url", return_value=mock_redis):
            cmd_turns.func()

        cmd_turns.caller.msg.assert_called_once()
        output = cmd_turns.caller.msg.call_args[0][0]
        assert "paused" in output
        assert "sleeping" in output

    def test_cmd_turns_no_pause_sleep_flags(self, cmd_turns, mock_redis):
        """Test @turns shows no flags when agent is not paused or sleeping."""
        now = datetime.now()
        heartbeat = (now - timedelta(minutes=1)).isoformat()

        mock_redis.keys.return_value = [b"agent:andi:turn_request"]
        mock_redis.hgetall.return_value = {
            b"status": b"ready",
            b"sequence_id": b"42",
            b"heartbeat_at": heartbeat.encode(),
            b"reason": b"events",
        }
        # Mock both flags not set
        mock_redis.get.return_value = None
        mock_redis.hget.return_value = None

        with patch("redis.from_url", return_value=mock_redis):
            cmd_turns.func()

        cmd_turns.caller.msg.assert_called_once()
        output = cmd_turns.caller.msg.call_args[0][0]
        assert "paused" not in output
        assert "sleeping" not in output


@pytest.mark.skipif(not EVENNIA_AVAILABLE, reason="Evennia not available")
class TestCmdEvents:
    """Tests for the @events command."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        return MagicMock(spec=redis.Redis)

    @pytest.fixture
    def mock_caller(self):
        """Create a mock command caller."""
        caller = MagicMock()
        caller.msg = MagicMock()
        return caller

    @pytest.fixture
    def cmd_events(self, mock_caller):
        """Create a CmdEvents instance with mocked caller."""
        cmd = CmdEvents()
        cmd.caller = mock_caller
        return cmd

    @pytest.fixture
    def sample_event_data(self):
        """Create sample event data for testing."""
        return {
            "type": "SPEECH",
            "content": "Hello, how are you today?",
            "sequence_id": 15,
            "self_action": False,
        }

    def test_cmd_events_no_args(self, cmd_events):
        """Test @events with no arguments."""
        cmd_events.args = ""
        cmd_events.func()

        cmd_events.caller.msg.assert_called_once()
        output = cmd_events.caller.msg.call_args[0][0]
        assert "Usage:" in output

    def test_cmd_events_valid_agent(self, cmd_events, mock_redis, sample_event_data):
        """Test @events with valid agent ID."""
        cmd_events.args = "andi"

        mock_redis.xlen.return_value = 5
        mock_redis.xrevrange.return_value = [
            (b"1736423400000-0", {b"data": json.dumps(sample_event_data).encode()})
        ]
        mock_redis.xrange.return_value = [
            (b"1736423300000-0", {b"data": json.dumps(sample_event_data).encode()})
        ]

        with patch("redis.from_url", return_value=mock_redis):
            cmd_events.func()

        cmd_events.caller.msg.assert_called_once()
        output = cmd_events.caller.msg.call_args[0][0]
        assert "Event Queue: andi" in output
        assert "5 events" in output or "Queue Length:|n 5" in output
        assert "SPEECH" in output
        assert "Hello, how are you today?" in output

    def test_cmd_events_empty_queue(self, cmd_events, mock_redis):
        """Test @events with empty event queue."""
        cmd_events.args = "andi"

        mock_redis.xlen.return_value = 0
        mock_redis.xrevrange.return_value = []

        with patch("redis.from_url", return_value=mock_redis):
            cmd_events.func()

        cmd_events.caller.msg.assert_called_once()
        output = cmd_events.caller.msg.call_args[0][0]
        assert "0 events" in output or "Queue Length:|n 0" in output
        assert "Queue is empty" in output or "empty" in output.lower()

    def test_cmd_events_with_custom_limit(self, cmd_events, mock_redis, sample_event_data):
        """Test @events with custom limit parameter."""
        cmd_events.args = "andi 25"

        mock_redis.xlen.return_value = 100
        mock_redis.xrevrange.return_value = [
            (
                f"173642340000{i}-0".encode(),
                {b"data": json.dumps(sample_event_data).encode()},
            )
            for i in range(25)
        ]
        mock_redis.xrange.return_value = [
            (b"1736423000000-0", {b"data": json.dumps(sample_event_data).encode()})
        ]

        with patch("redis.from_url", return_value=mock_redis):
            cmd_events.func()

        # Verify xrevrange was called with limit=25
        mock_redis.xrevrange.assert_called_once()
        args, kwargs = mock_redis.xrevrange.call_args
        assert kwargs.get("count") == 25 or args[3] == 25

    def test_cmd_events_invalid_limit_non_numeric(self, cmd_events):
        """Test @events with non-numeric limit."""
        cmd_events.args = "andi abc"

        cmd_events.func()

        cmd_events.caller.msg.assert_called_once()
        output = cmd_events.caller.msg.call_args[0][0]
        assert "Invalid limit" in output

    def test_cmd_events_invalid_limit_negative(self, cmd_events):
        """Test @events with negative limit."""
        cmd_events.args = "andi -5"

        cmd_events.func()

        cmd_events.caller.msg.assert_called_once()
        output = cmd_events.caller.msg.call_args[0][0]
        assert "Invalid limit" in output

    def test_cmd_events_invalid_limit_zero(self, cmd_events):
        """Test @events with zero limit."""
        cmd_events.args = "andi 0"

        cmd_events.func()

        cmd_events.caller.msg.assert_called_once()
        output = cmd_events.caller.msg.call_args[0][0]
        assert "Invalid limit" in output

    def test_cmd_events_self_action_flag(self, cmd_events, mock_redis):
        """Test @events displays self_action indicator."""
        cmd_events.args = "andi"

        self_action_event = {
            "type": "MOVEMENT",
            "content": "You move north.",
            "sequence_id": 20,
            "self_action": True,
        }

        mock_redis.xlen.return_value = 1
        mock_redis.xrevrange.return_value = [
            (b"1736423400000-0", {b"data": json.dumps(self_action_event).encode()})
        ]
        mock_redis.xrange.return_value = [
            (b"1736423400000-0", {b"data": json.dumps(self_action_event).encode()})
        ]

        with patch("redis.from_url", return_value=mock_redis):
            cmd_events.func()

        cmd_events.caller.msg.assert_called_once()
        output = cmd_events.caller.msg.call_args[0][0]
        assert "self-action" in output

    def test_cmd_events_truncates_long_content(self, cmd_events, mock_redis):
        """Test @events truncates long event content."""
        cmd_events.args = "andi"

        long_content = "A" * 100  # 100 characters
        long_event = {
            "type": "SPEECH",
            "content": long_content,
            "sequence_id": 15,
            "self_action": False,
        }

        mock_redis.xlen.return_value = 1
        mock_redis.xrevrange.return_value = [
            (b"1736423400000-0", {b"data": json.dumps(long_event).encode()})
        ]
        mock_redis.xrange.return_value = [
            (b"1736423400000-0", {b"data": json.dumps(long_event).encode()})
        ]

        with patch("redis.from_url", return_value=mock_redis):
            cmd_events.func()

        cmd_events.caller.msg.assert_called_once()
        output = cmd_events.caller.msg.call_args[0][0]
        # Should contain ellipsis for truncation
        assert "..." in output or len(output) < len(long_content)

    def test_cmd_events_multiple_event_types(self, cmd_events, mock_redis):
        """Test @events with multiple different event types."""
        cmd_events.args = "andi"

        events = [
            {
                "type": "SPEECH",
                "content": "Hello!",
                "sequence_id": 10,
                "self_action": False,
            },
            {
                "type": "MOVEMENT",
                "content": "Someone arrives",
                "sequence_id": 11,
                "self_action": False,
            },
            {
                "type": "ACTION",
                "content": "Bob waves",
                "sequence_id": 12,
                "self_action": False,
            },
        ]

        mock_redis.xlen.return_value = 3
        mock_redis.xrevrange.return_value = [
            (f"173642340000{i}-0".encode(), {b"data": json.dumps(event).encode()})
            for i, event in enumerate(events)
        ]
        mock_redis.xrange.return_value = [
            (b"1736423400000-0", {b"data": json.dumps(events[0]).encode()})
        ]

        with patch("redis.from_url", return_value=mock_redis):
            cmd_events.func()

        cmd_events.caller.msg.assert_called_once()
        output = cmd_events.caller.msg.call_args[0][0]
        assert "SPEECH" in output
        assert "MOVEMENT" in output
        assert "ACTION" in output

    def test_cmd_events_displays_timestamps(self, cmd_events, mock_redis, sample_event_data):
        """Test @events displays oldest and newest event timestamps."""
        cmd_events.args = "andi"

        mock_redis.xlen.return_value = 10
        mock_redis.xrevrange.return_value = [
            (b"1736423400000-0", {b"data": json.dumps(sample_event_data).encode()})
        ]
        mock_redis.xrange.return_value = [
            (b"1736423300000-0", {b"data": json.dumps(sample_event_data).encode()})
        ]

        with patch("redis.from_url", return_value=mock_redis):
            cmd_events.func()

        cmd_events.caller.msg.assert_called_once()
        output = cmd_events.caller.msg.call_args[0][0]
        assert "Oldest Event:" in output
        assert "Newest Event:" in output
        # Should contain time ago format
        assert "ago" in output

    def test_cmd_events_handles_parse_error(self, cmd_events, mock_redis):
        """Test @events handles JSON parse errors gracefully."""
        cmd_events.args = "andi"

        mock_redis.xlen.return_value = 1
        mock_redis.xrevrange.return_value = [
            (b"1736423400000-0", {b"data": b"invalid-json"})
        ]
        mock_redis.xrange.return_value = [
            (b"1736423400000-0", {b"data": b"invalid-json"})
        ]

        with patch("redis.from_url", return_value=mock_redis):
            cmd_events.func()

        cmd_events.caller.msg.assert_called_once()
        output = cmd_events.caller.msg.call_args[0][0]
        assert "parse error" in output

    def test_cmd_events_handles_missing_event_data(self, cmd_events, mock_redis):
        """Test @events handles missing event data field."""
        cmd_events.args = "andi"

        mock_redis.xlen.return_value = 1
        # Event without 'data' field
        mock_redis.xrevrange.return_value = [(b"1736423400000-0", {b"other": b"value"})]
        mock_redis.xrange.return_value = [(b"1736423400000-0", {b"other": b"value"})]

        with patch("redis.from_url", return_value=mock_redis):
            cmd_events.func()

        # Should not crash, may show empty or skip event
        cmd_events.caller.msg.assert_called_once()

    def test_cmd_events_sequence_id_display(self, cmd_events, mock_redis, sample_event_data):
        """Test @events displays sequence_id for each event."""
        cmd_events.args = "andi"

        event_with_seq = sample_event_data.copy()
        event_with_seq["sequence_id"] = 42

        mock_redis.xlen.return_value = 1
        mock_redis.xrevrange.return_value = [
            (b"1736423400000-0", {b"data": json.dumps(event_with_seq).encode()})
        ]
        mock_redis.xrange.return_value = [
            (b"1736423400000-0", {b"data": json.dumps(event_with_seq).encode()})
        ]

        with patch("redis.from_url", return_value=mock_redis):
            cmd_events.func()

        cmd_events.caller.msg.assert_called_once()
        output = cmd_events.caller.msg.call_args[0][0]
        assert "seq=42" in output
