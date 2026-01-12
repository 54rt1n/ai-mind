# packages/aim-mud/tests/mud_tests/unit/worker/test_turn_request_helpers.py
# Tests for turn request helper methods
# Philosophy: Test real helper methods with mocked Redis responses

import pytest
from unittest.mock import AsyncMock
from datetime import datetime, timezone

from aim_mud_types.helper import _utc_now
from aim_mud_types.coordination import MUDTurnRequest


class TestGetTurnRequest:
    """Tests for _get_turn_request() helper method."""

    @pytest.mark.asyncio
    async def test_get_turn_request_returns_dict(self, test_worker):
        """Test that _get_turn_request() returns MUDTurnRequest object from Redis."""
        # Arrange
        expected = {
            b"status": b"ready",
            b"turn_id": b"turn123",
            b"reason": b"events",
            b"heartbeat_at": _utc_now().isoformat().encode(),
            b"sequence_id": b"1",
        }
        test_worker.redis.hgetall = AsyncMock(return_value=expected)

        # Act
        result = await test_worker._get_turn_request()

        # Assert
        assert result is not None
        assert result.status == "ready"
        assert result.turn_id == "turn123"
        assert test_worker.redis.hgetall.called

    @pytest.mark.asyncio
    async def test_get_turn_request_handles_missing_key(self, test_worker):
        """Test that _get_turn_request() returns None if key missing."""
        # Arrange
        test_worker.redis.hgetall = AsyncMock(return_value={})

        # Act
        result = await test_worker._get_turn_request()

        # Assert
        assert result is None


class TestUpdateTurnRequest:
    """Tests for update_turn_request() helper method."""

    @pytest.mark.asyncio
    async def test_update_turn_request_basic(self, test_worker):
        """Test updating turn request uses Lua script CAS."""
        # Arrange
        turn_request = MUDTurnRequest(
            status="in_progress",
            turn_id="turn123",
            reason="events",
            heartbeat_at=_utc_now(),
            sequence_id="1",
        )
        test_worker.redis.eval = AsyncMock(return_value=1)  # CAS success

        # Act
        result = await test_worker.update_turn_request(turn_request, expected_turn_id="turn123")

        # Assert
        assert result is True
        test_worker.redis.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_turn_request_with_message(self, test_worker):
        """Test updating request with status message."""
        # Arrange
        turn_request = MUDTurnRequest(
            status="done",
            turn_id="turn123",
            reason="events",
            heartbeat_at=_utc_now(),
            sequence_id="1",
            message="Completed successfully",
        )
        test_worker.redis.eval = AsyncMock(return_value=1)

        # Act
        result = await test_worker.update_turn_request(turn_request, expected_turn_id="turn123")

        # Assert
        assert result is True
        # Verify message is in the Lua script args
        call_args = test_worker.redis.eval.call_args[0]
        assert "Completed successfully" in call_args

    @pytest.mark.asyncio
    async def test_update_turn_request_cas_failure(self, test_worker):
        """Test that CAS failure returns False."""
        # Arrange
        turn_request = MUDTurnRequest(
            status="in_progress",
            turn_id="turn123",
            reason="events",
            heartbeat_at=_utc_now(),
            sequence_id="1",
        )
        test_worker.redis.eval = AsyncMock(return_value=0)  # CAS failed

        # Act
        result = await test_worker.update_turn_request(turn_request, expected_turn_id="turn456")

        # Assert
        assert result is False


class TestIsPaused:
    """Tests for _is_paused() helper method."""

    @pytest.mark.asyncio
    async def test_is_paused_returns_true_when_paused(self, test_worker):
        """Test that _is_paused() returns True when flag set."""
        # Arrange
        test_worker.redis.get = AsyncMock(return_value=b"1")

        # Act
        result = await test_worker._is_paused()

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_is_paused_returns_false_when_not_paused(self, test_worker):
        """Test that _is_paused() returns False when flag not set."""
        # Arrange
        test_worker.redis.get = AsyncMock(return_value=None)

        # Act
        result = await test_worker._is_paused()

        # Assert
        assert result is False


class TestCheckAbortRequestedDuplicate:
    """Tests for _check_abort_requested() helper method.

    Note: These duplicate tests in test_state_helpers.py which are more comprehensive.
    Keeping these for now to verify the same behavior.
    """

    @pytest.mark.asyncio
    async def test_check_abort_clears_flag_when_set(self, test_worker):
        """Test that abort status triggers state transition."""
        from aim_mud_types import TurnRequestStatus
        from unittest.mock import patch

        # Arrange
        turn_request = MUDTurnRequest(
            status="abort_requested",
            turn_id="turn123",
            reason="events",
            heartbeat_at=_utc_now(),
            sequence_id="1",
        )
        test_worker._get_turn_request = AsyncMock(return_value=turn_request)

        # Mock the helper function from aim_mud_types where it's imported
        with patch('aim_mud_types.turn_request_helpers.transition_turn_request_and_update_async') as mock_transition:
            mock_transition.return_value = None  # Async function

            # Act
            result = await test_worker._check_abort_requested()

            # Assert
            assert result is True
            # Verify transition was called with correct arguments
            assert mock_transition.called
            call_kwargs = mock_transition.call_args[1]
            assert call_kwargs['status'] == TurnRequestStatus.ABORTED
            assert call_kwargs['message'] == "Aborted by user request"

    @pytest.mark.asyncio
    async def test_check_abort_returns_false_when_not_set(self, test_worker):
        """Test that check returns False when status is not abort_requested."""
        # Arrange
        turn_request = MUDTurnRequest(
            status="ready",
            turn_id="turn123",
            reason="events",
            heartbeat_at=_utc_now(),
            sequence_id="1",
        )
        test_worker._get_turn_request = AsyncMock(return_value=turn_request)
        test_worker.update_turn_request = AsyncMock()

        # Act
        result = await test_worker._check_abort_requested()

        # Assert
        assert result is False
        test_worker.update_turn_request.assert_not_called()


class TestAtomicHeartbeatUpdate:
    """Tests for atomic_heartbeat_update() method."""

    @pytest.mark.asyncio
    async def test_atomic_heartbeat_success(self, test_worker):
        """Test successful atomic heartbeat update."""
        # Arrange
        test_worker.redis.eval = AsyncMock(return_value=1)  # Success

        # Act
        result = await test_worker.atomic_heartbeat_update()

        # Assert
        assert result == 1
        test_worker.redis.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_atomic_heartbeat_key_missing(self, test_worker):
        """Test atomic heartbeat when key doesn't exist (normal during shutdown)."""
        # Arrange
        test_worker.redis.eval = AsyncMock(return_value=0)  # Key missing

        # Act
        result = await test_worker.atomic_heartbeat_update()

        # Assert
        assert result == 0
        test_worker.redis.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_atomic_heartbeat_corrupted_hash(self, test_worker):
        """Test atomic heartbeat detects corrupted hash (missing required fields)."""
        # Arrange
        test_worker.redis.eval = AsyncMock(return_value=-1)  # Corrupted

        # Act
        result = await test_worker.atomic_heartbeat_update()

        # Assert
        assert result == -1
        test_worker.redis.eval.assert_called_once()
