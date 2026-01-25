# packages/aim-mud/tests/mud_tests/unit/worker/test_action_id.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for action_id generation in _emit_actions()."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from andimud_worker.mixins.datastore.actions import generate_action_id, ActionsMixin
from aim_mud_types import MUDAction


class TestGenerateActionId:
    """Tests for generate_action_id() function."""

    def test_generates_string(self):
        """generate_action_id returns a string."""
        action_id = generate_action_id()
        assert isinstance(action_id, str)

    def test_format_has_prefix(self):
        """action_id starts with 'act_' prefix."""
        action_id = generate_action_id()
        assert action_id.startswith("act_")

    def test_format_has_timestamp(self):
        """action_id contains a timestamp component."""
        action_id = generate_action_id()
        parts = action_id.split("_")
        assert len(parts) == 3
        # Second part should be numeric (timestamp in ms)
        assert parts[1].isdigit()

    def test_format_has_random_suffix(self):
        """action_id contains a random hex suffix."""
        action_id = generate_action_id()
        parts = action_id.split("_")
        assert len(parts) == 3
        # Third part should be hex (8 chars for 4 bytes)
        suffix = parts[2]
        assert len(suffix) == 8
        int(suffix, 16)  # Should not raise if valid hex

    def test_uniqueness(self):
        """Multiple calls generate unique action_ids."""
        ids = [generate_action_id() for _ in range(100)]
        assert len(set(ids)) == 100, "All generated IDs should be unique"

    def test_timestamp_increases(self):
        """Timestamps in sequential action_ids are non-decreasing."""
        import time

        id1 = generate_action_id()
        time.sleep(0.001)  # Brief pause to ensure different timestamp
        id2 = generate_action_id()

        ts1 = int(id1.split("_")[1])
        ts2 = int(id2.split("_")[1])

        assert ts2 >= ts1


class TestEmitActionsActionId:
    """Tests for action_id generation and inclusion in _emit_actions()."""

    @pytest.fixture
    def mock_worker(self):
        """Create a mock worker with required attributes."""
        worker = MagicMock()
        worker.redis = AsyncMock()
        worker.config = MagicMock()
        worker.config.agent_id = "test_agent"
        worker.config.action_stream = "mud:actions"
        worker._update_agent_profile = AsyncMock()
        return worker

    @pytest.fixture
    def sample_action(self):
        """Create a sample MUDAction."""
        return MUDAction(
            tool="emote",
            args={"text": "waves hello"}
        )

    @pytest.mark.asyncio
    async def test_emit_actions_returns_action_ids(self, mock_worker, sample_action):
        """_emit_actions returns tuple of (action_ids, expects_echo)."""
        from aim_mud_types.client import RedisMUDClient

        with patch.object(RedisMUDClient, "append_mud_action", new_callable=AsyncMock) as mock_append:
            with patch.object(RedisMUDClient, "trim_mud_actions_maxlen", new_callable=AsyncMock):
                mock_append.return_value = "1-0"

                action_ids, expects_echo = await ActionsMixin._emit_actions(mock_worker, [sample_action])

                assert isinstance(action_ids, list)
                assert len(action_ids) == 1
                assert action_ids[0].startswith("act_")
                assert expects_echo is True  # Normal actions expect echo

    @pytest.mark.asyncio
    async def test_emit_actions_includes_action_id_in_data(self, mock_worker, sample_action):
        """_emit_actions includes action_id in the Redis stream data."""
        from aim_mud_types.client import RedisMUDClient

        captured_data = []

        async def capture_append(data, stream_key):
            captured_data.append(data)
            return "1-0"

        with patch.object(RedisMUDClient, "append_mud_action", side_effect=capture_append):
            with patch.object(RedisMUDClient, "trim_mud_actions_maxlen", new_callable=AsyncMock):
                await ActionsMixin._emit_actions(mock_worker, [sample_action])

                assert len(captured_data) == 1
                payload = json.loads(captured_data[0]["data"])
                assert "action_id" in payload
                assert payload["action_id"].startswith("act_")

    @pytest.mark.asyncio
    async def test_emit_actions_multiple_actions_unique_ids(self, mock_worker):
        """_emit_actions generates unique action_id for each action."""
        from aim_mud_types.client import RedisMUDClient

        actions = [
            MUDAction(tool="emote", args={"text": "first action"}),
            MUDAction(tool="speak", args={"text": "second action"}),
            MUDAction(tool="emote", args={"text": "third action"}),
        ]

        captured_data = []

        async def capture_append(data, stream_key):
            captured_data.append(data)
            return f"{len(captured_data)}-0"

        with patch.object(RedisMUDClient, "append_mud_action", side_effect=capture_append):
            with patch.object(RedisMUDClient, "trim_mud_actions_maxlen", new_callable=AsyncMock):
                action_ids, expects_echo = await ActionsMixin._emit_actions(mock_worker, actions)

                assert len(action_ids) == 3
                # All IDs should be unique
                assert len(set(action_ids)) == 3
                assert expects_echo is True

                # Verify each payload has its own action_id
                for i, data in enumerate(captured_data):
                    payload = json.loads(data["data"])
                    assert payload["action_id"] == action_ids[i]

    @pytest.mark.asyncio
    async def test_emit_actions_empty_command_not_counted(self, mock_worker):
        """Actions with empty commands are skipped and not given action_ids."""
        from aim_mud_types.client import RedisMUDClient

        # Create an action that produces an empty command
        empty_action = MagicMock()
        empty_action.to_command.return_value = ""
        empty_action.tool = "empty_tool"
        empty_action.args = {}
        empty_action.expects_echo.return_value = True

        valid_action = MUDAction(tool="emote", args={"text": "hello"})

        with patch.object(RedisMUDClient, "append_mud_action", new_callable=AsyncMock) as mock_append:
            with patch.object(RedisMUDClient, "trim_mud_actions_maxlen", new_callable=AsyncMock):
                mock_append.return_value = "1-0"

                action_ids, expects_echo = await ActionsMixin._emit_actions(mock_worker, [empty_action, valid_action])

                # Only the valid action should produce an action_id
                assert len(action_ids) == 1
                assert mock_append.call_count == 1
                assert expects_echo is True

    @pytest.mark.asyncio
    async def test_emit_actions_error_handling(self, mock_worker, sample_action):
        """_emit_actions handles errors gracefully and continues."""
        from aim_mud_types.client import RedisMUDClient

        action1 = MUDAction(tool="emote", args={"text": "first"})
        action2 = MUDAction(tool="emote", args={"text": "second"})

        call_count = [0]

        async def fail_first(data, stream_key):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Redis error")
            return "2-0"

        with patch.object(RedisMUDClient, "append_mud_action", side_effect=fail_first):
            with patch.object(RedisMUDClient, "trim_mud_actions_maxlen", new_callable=AsyncMock):
                action_ids, expects_echo = await ActionsMixin._emit_actions(mock_worker, [action1, action2])

                # Only the second action succeeded
                assert len(action_ids) == 1
                assert expects_echo is True

    @pytest.mark.asyncio
    async def test_emit_actions_sets_last_emitted_action_ids(self, mock_worker, sample_action):
        """_emit_actions sets _last_emitted_action_ids on the worker."""
        from aim_mud_types.client import RedisMUDClient

        # Initialize the attributes (normally done in worker __init__)
        mock_worker._last_emitted_action_ids = []
        mock_worker._last_emitted_expects_echo = True

        actions = [
            MUDAction(tool="emote", args={"text": "first"}),
            MUDAction(tool="emote", args={"text": "second"}),
        ]

        with patch.object(RedisMUDClient, "append_mud_action", new_callable=AsyncMock) as mock_append:
            with patch.object(RedisMUDClient, "trim_mud_actions_maxlen", new_callable=AsyncMock):
                mock_append.return_value = "1-0"

                action_ids, expects_echo = await ActionsMixin._emit_actions(mock_worker, actions)

                # _last_emitted_action_ids should match return value
                assert mock_worker._last_emitted_action_ids == action_ids
                assert len(mock_worker._last_emitted_action_ids) == 2
                assert mock_worker._last_emitted_expects_echo == expects_echo


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
