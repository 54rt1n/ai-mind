# packages/aim-mud/tests/mud_tests/unit/worker/test_state_helpers.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for state helper methods used by main loop.

Tests the methods from StateMixin that _run_main_loop() depends on:
- _is_paused() - pause state checking
- _check_abort_requested() - abort request detection
- _should_act_spontaneously() - spontaneous action triggers

These are the building blocks that enable main loop state transitions.
"""

import pytest
from unittest.mock import AsyncMock
from datetime import datetime, timezone, timedelta

from andimud_worker.worker import MUDAgentWorker
from aim_mud_types import MUDSession


class TestIsPaused:
    """Test _is_paused() method."""

    @pytest.mark.asyncio
    async def test_returns_true_when_paused(self, test_worker, mock_redis):
        """Test _is_paused returns True when Redis flag is set."""
        mock_redis.get = AsyncMock(return_value=b"1")

        result = await test_worker._is_paused()

        assert result is True
        mock_redis.get.assert_called_once_with(test_worker.config.pause_key)

    @pytest.mark.asyncio
    async def test_returns_false_when_not_paused(self, test_worker, mock_redis):
        """Test _is_paused returns False when Redis flag is not set."""
        mock_redis.get = AsyncMock(return_value=b"0")

        result = await test_worker._is_paused()

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_flag_missing(self, test_worker, mock_redis):
        """Test _is_paused returns False when Redis flag doesn't exist."""
        mock_redis.get = AsyncMock(return_value=None)

        result = await test_worker._is_paused()

        assert result is False


class TestCheckAbortRequested:
    """Test _check_abort_requested() method."""

    @pytest.mark.asyncio
    async def test_returns_true_when_abort_requested(self, test_worker, mock_redis):
        """Test returns True and clears flag when abort_requested."""
        from aim_mud_types.helper import _utc_now
        # _get_turn_request returns abort_requested status
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"abort_requested",
            b"turn_id": b"test_turn_123",
            b"reason": b"events",
            b"heartbeat_at": _utc_now().isoformat().encode(),
            b"sequence_id": b"1",
        })

        # Track state transitions
        state_changes = []
        async def track_state(turn_id, status, **kwargs):
            state_changes.append((turn_id, status))

        test_worker._set_turn_request_state = AsyncMock(side_effect=track_state)

        result = await test_worker._check_abort_requested()

        assert result is True
        # Verify state was set to aborted
        assert len(state_changes) == 1
        assert state_changes[0] == ("test_turn_123", "aborted")

    @pytest.mark.asyncio
    async def test_returns_false_when_not_abort_requested(self, test_worker, mock_redis):
        """Test returns False when status is not abort_requested."""
        from aim_mud_types.helper import _utc_now
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"test_turn_456",
            b"reason": b"events",
            b"heartbeat_at": _utc_now().isoformat().encode(),
            b"sequence_id": b"1",
        })

        test_worker._set_turn_request_state = AsyncMock()

        result = await test_worker._check_abort_requested()

        assert result is False
        # Verify state was NOT changed
        assert not test_worker._set_turn_request_state.called

    @pytest.mark.asyncio
    async def test_handles_missing_turn_id(self, test_worker, mock_redis):
        """Test handles abort_requested with missing turn_id (should fail validation)."""
        from aim_mud_types.helper import _utc_now
        # Missing turn_id will cause MUDTurnRequest validation to fail
        # from_redis will return None in this case
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"abort_requested",
            b"reason": b"events",
            b"heartbeat_at": _utc_now().isoformat().encode(),
            b"sequence_id": b"1",
            # No turn_id - will cause validation failure
        })

        state_changes = []
        async def track_state(turn_id, status, **kwargs):
            state_changes.append((turn_id, status))

        test_worker._set_turn_request_state = AsyncMock(side_effect=track_state)

        result = await test_worker._check_abort_requested()

        # With MUDTurnRequest validation, missing turn_id causes from_redis to return None
        # So _check_abort_requested should return False (no valid turn request)
        assert result is False
        # Should NOT attempt state transition when turn_request is None
        assert len(state_changes) == 0


class TestShouldActSpontaneously:
    """Test _should_act_spontaneously() method."""

    def test_returns_false_when_no_session(self, test_worker):
        """Test returns False when session is None."""
        test_worker.session = None

        result = test_worker._should_act_spontaneously()

        assert result is False

    def test_returns_false_when_no_last_event(self, test_worker):
        """Test returns False when no events have occurred yet."""
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            last_event_id="0-0",
            last_event_time=None,  # No events yet
        )

        result = test_worker._should_act_spontaneously()

        assert result is False

    def test_returns_false_when_not_enough_time_since_event(self, test_worker):
        """Test returns False when not enough time elapsed since last event."""
        now = datetime.now(timezone.utc)
        recent_event = now - timedelta(seconds=30)  # 30 seconds ago

        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            last_event_id="1-0",
            last_event_time=recent_event,
            last_action_time=None,
        )
        # spontaneous_action_interval is 300 seconds (from fixture)
        assert test_worker.config.spontaneous_action_interval == 300.0

        result = test_worker._should_act_spontaneously()

        assert result is False

    def test_returns_true_when_enough_time_since_event_no_action(self, test_worker):
        """Test returns True when enough time passed since event and no action yet."""
        now = datetime.now(timezone.utc)
        old_event = now - timedelta(seconds=400)  # 400 seconds ago

        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            last_event_id="1-0",
            last_event_time=old_event,
            last_action_time=None,  # Never acted
        )

        result = test_worker._should_act_spontaneously()

        assert result is True

    def test_returns_true_when_enough_time_since_both(self, test_worker):
        """Test returns True when enough time since both event and action."""
        now = datetime.now(timezone.utc)
        old_event = now - timedelta(seconds=400)
        old_action = now - timedelta(seconds=350)

        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            last_event_id="1-0",
            last_event_time=old_event,
            last_action_time=old_action,
        )

        result = test_worker._should_act_spontaneously()

        assert result is True

    def test_returns_false_when_recent_action(self, test_worker):
        """Test returns False when action was recent even if event was old."""
        now = datetime.now(timezone.utc)
        old_event = now - timedelta(seconds=400)
        recent_action = now - timedelta(seconds=100)  # Recent action

        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            last_event_id="1-0",
            last_event_time=old_event,
            last_action_time=recent_action,  # Too recent
        )

        result = test_worker._should_act_spontaneously()

        assert result is False

    def test_boundary_exactly_at_interval(self, test_worker):
        """Test boundary condition at exactly the spontaneous interval."""
        now = datetime.now(timezone.utc)
        # Exactly 300 seconds ago (the interval)
        exact_event = now - timedelta(seconds=300)

        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            last_event_id="1-0",
            last_event_time=exact_event,
            last_action_time=None,
        )

        result = test_worker._should_act_spontaneously()

        # At exactly the boundary, should trigger (>= check)
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
