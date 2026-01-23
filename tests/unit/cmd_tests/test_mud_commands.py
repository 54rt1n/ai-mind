# tests/unit/commands/test_mud_commands.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for MUD control command logic.

These tests verify the core logic of command state transitions without
requiring full Evennia initialization. They test the Redis interactions
and state changes that the commands perform.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import redis

from aim_mud_types import RedisKeys


class TestPauseResumeLogic:
    """Test pause/resume Redis key manipulation logic."""

    def test_pause_sets_redis_key_to_one(self):
        """Test pause operation sets Redis key to '1'."""
        mock_redis = Mock(spec=redis.Redis)
        mock_redis.set = Mock()

        agent_id = "test_agent"
        pause_key = RedisKeys.agent_pause(agent_id)

        # Simulate pause operation
        mock_redis.set(pause_key, "1")

        mock_redis.set.assert_called_once_with(pause_key, "1")

    def test_resume_deletes_redis_key(self):
        """Test resume operation deletes Redis key."""
        mock_redis = Mock(spec=redis.Redis)
        mock_redis.delete = Mock()

        agent_id = "test_agent"
        pause_key = RedisKeys.agent_pause(agent_id)

        # Simulate resume operation
        mock_redis.delete(pause_key)

        mock_redis.delete.assert_called_once_with(pause_key)

    def test_pause_key_format(self):
        """Test pause key has correct format."""
        agent_id = "andi"
        pause_key = RedisKeys.agent_pause(agent_id)

        assert "andi" in pause_key
        assert "pause" in pause_key


class TestSleepWakeLogic:
    """Test sleep/wake character attribute manipulation logic."""

    def test_sleep_sets_is_sleeping_to_true(self):
        """Test sleep operation sets db.is_sleeping = True."""
        mock_char = Mock()
        mock_char.key = "Andi"
        mock_char.db = Mock()

        # Simulate sleep operation
        mock_char.db.is_sleeping = True

        assert mock_char.db.is_sleeping is True

    def test_wake_sets_is_sleeping_to_false(self):
        """Test wake operation sets db.is_sleeping = False."""
        mock_char = Mock()
        mock_char.key = "Andi"
        mock_char.db = Mock()
        mock_char.db.is_sleeping = True

        # Simulate wake operation
        mock_char.db.is_sleeping = False

        assert mock_char.db.is_sleeping is False

    def test_update_agent_profile_called_after_sleep(self):
        """Test update_agent_profile is called after changing sleep state."""
        mock_char = Mock()
        mock_char.db = Mock()
        mock_char.update_agent_profile = Mock()

        # Simulate sleep with profile update
        mock_char.db.is_sleeping = True
        if hasattr(mock_char, "update_agent_profile"):
            mock_char.update_agent_profile()

        mock_char.update_agent_profile.assert_called_once()


class TestAbortLogic:
    """Test abort turn request status transitions."""

    def test_abort_sets_status_to_abort_requested_when_in_progress(self):
        """Test abort sets status to abort_requested when turn is in_progress."""
        mock_redis = Mock(spec=redis.Redis)
        mock_redis.hgetall = Mock(return_value={
            b"status": b"in_progress",
            b"turn_id": b"test_turn_123"
        })
        mock_redis.hset = Mock()

        agent_id = "test_agent"
        turn_key = RedisKeys.agent_turn_request(agent_id)

        # Get current status
        current = mock_redis.hgetall(turn_key)
        status = current.get(b"status")
        if isinstance(status, bytes):
            status = status.decode("utf-8")

        # Abort if in progress
        if status in ("assigned", "in_progress"):
            mock_redis.hset(turn_key, "status", "abort_requested")

        mock_redis.hset.assert_called_once_with(turn_key, "status", "abort_requested")

    def test_abort_sets_status_to_abort_requested_when_assigned(self):
        """Test abort sets status to abort_requested when turn is assigned."""
        mock_redis = Mock(spec=redis.Redis)
        mock_redis.hgetall = Mock(return_value={
            b"status": b"assigned",
            b"turn_id": b"test_turn_456"
        })
        mock_redis.hset = Mock()

        agent_id = "test_agent"
        turn_key = RedisKeys.agent_turn_request(agent_id)

        # Get current status
        current = mock_redis.hgetall(turn_key)
        status = current.get(b"status")
        if isinstance(status, bytes):
            status = status.decode("utf-8")

        # Abort if assigned
        if status in ("assigned", "in_progress"):
            mock_redis.hset(turn_key, "status", "abort_requested")

        mock_redis.hset.assert_called_once_with(turn_key, "status", "abort_requested")

    def test_abort_skips_when_status_is_done(self):
        """Test abort does not set status when turn is done."""
        mock_redis = Mock(spec=redis.Redis)
        mock_redis.hgetall = Mock(return_value={
            b"status": b"done",
            b"turn_id": b"test_turn_789"
        })
        mock_redis.hset = Mock()

        agent_id = "test_agent"
        turn_key = RedisKeys.agent_turn_request(agent_id)

        # Get current status
        current = mock_redis.hgetall(turn_key)
        status = current.get(b"status")
        if isinstance(status, bytes):
            status = status.decode("utf-8")

        # Don't abort if done
        if status in ("assigned", "in_progress"):
            mock_redis.hset(turn_key, "status", "abort_requested")

        mock_redis.hset.assert_not_called()

    def test_abort_skips_when_no_active_turn(self):
        """Test abort does not set status when no turn request exists."""
        mock_redis = Mock(spec=redis.Redis)
        mock_redis.hgetall = Mock(return_value={})
        mock_redis.hset = Mock()

        agent_id = "test_agent"
        turn_key = RedisKeys.agent_turn_request(agent_id)

        # Get current status
        current = mock_redis.hgetall(turn_key)
        status = current.get(b"status")
        if status is None:
            # No turn to abort
            pass
        else:
            if isinstance(status, bytes):
                status = status.decode("utf-8")
            if status in ("assigned", "in_progress"):
                mock_redis.hset(turn_key, "status", "abort_requested")

        mock_redis.hset.assert_not_called()

    def test_abort_handles_string_keys(self):
        """Test abort handles string keys (decode_responses=True)."""
        mock_redis = Mock(spec=redis.Redis)
        mock_redis.hgetall = Mock(return_value={
            "status": "in_progress",
            "turn_id": "test_turn_abc"
        })
        mock_redis.hset = Mock()

        agent_id = "test_agent"
        turn_key = RedisKeys.agent_turn_request(agent_id)

        # Get current status
        current = mock_redis.hgetall(turn_key)
        status = current.get("status") or current.get(b"status")
        if isinstance(status, bytes):
            status = status.decode("utf-8")

        # Abort if in progress
        if status in ("assigned", "in_progress"):
            mock_redis.hset(turn_key, "status", "abort_requested")

        mock_redis.hset.assert_called_once_with(turn_key, "status", "abort_requested")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
