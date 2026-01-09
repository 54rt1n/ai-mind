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


class TestSetTurnRequestState:
    """Tests for _set_turn_request_state() helper method."""

    @pytest.mark.asyncio
    async def test_set_turn_request_state_basic(self, test_worker):
        """Test setting turn request state uses Lua script CAS."""
        # Arrange
        test_worker.config.turn_request_ttl_seconds = 120  # Enable TTL for this test
        test_worker.redis.eval = AsyncMock(return_value=1)  # CAS success
        test_worker.redis.expire = AsyncMock()

        # Act
        result = await test_worker._set_turn_request_state("turn123", "in_progress")

        # Assert
        assert result is True
        test_worker.redis.eval.assert_called_once()
        test_worker.redis.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_turn_request_state_with_message(self, test_worker):
        """Test setting state with status message."""
        # Arrange
        test_worker.redis.eval = AsyncMock(return_value=1)
        test_worker.redis.expire = AsyncMock()

        # Act
        result = await test_worker._set_turn_request_state(
            "turn123", "done", message="Completed successfully"
        )

        # Assert
        assert result is True
        # Verify message is in the Lua script args
        call_args = test_worker.redis.eval.call_args[0]
        assert "Completed successfully" in call_args

    @pytest.mark.asyncio
    async def test_set_turn_request_state_with_extra_fields(self, test_worker):
        """Test setting state with extra fields."""
        # Arrange
        test_worker.redis.eval = AsyncMock(return_value=1)
        test_worker.redis.expire = AsyncMock()

        # Act
        result = await test_worker._set_turn_request_state(
            "turn123",
            "fail",
            extra_fields={"attempt_count": "2", "next_attempt_at": "2026-01-08T12:00:00"},
        )

        # Assert
        assert result is True
        # Verify extra fields are in the Lua script args
        call_args = test_worker.redis.eval.call_args[0]
        assert "2" in call_args  # attempt_count value
        assert "2026-01-08T12:00:00" in call_args  # next_attempt_at value

    @pytest.mark.asyncio
    async def test_set_turn_request_state_sets_ttl(self, test_worker):
        """Test that state updates refresh TTL when TTL>0."""
        # Arrange
        test_worker.config.turn_request_ttl_seconds = 120  # Enable TTL
        test_worker.redis.eval = AsyncMock(return_value=1)
        test_worker.redis.expire = AsyncMock()

        # Act
        await test_worker._set_turn_request_state("turn123", "in_progress")

        # Assert
        test_worker.redis.expire.assert_called_once()
        call_args = test_worker.redis.expire.call_args
        assert call_args[0][0] == test_worker._turn_request_key()
        assert call_args[0][1] == test_worker.config.turn_request_ttl_seconds

    @pytest.mark.asyncio
    async def test_set_turn_request_state_cas_failure(self, test_worker):
        """Test that CAS failure returns False."""
        # Arrange
        test_worker.redis.eval = AsyncMock(return_value=0)  # CAS failed

        # Act
        result = await test_worker._set_turn_request_state(
            "turn123", "in_progress", expected_turn_id="turn456"
        )

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


class TestCheckAbortRequested:
    """Tests for _check_abort_requested() helper method."""

    @pytest.mark.asyncio
    async def test_check_abort_clears_flag_when_set(self, test_worker):
        """Test that abort status triggers state transition."""
        # Arrange
        turn_request = MUDTurnRequest(
            status="abort_requested",
            turn_id="turn123",
        )
        test_worker._get_turn_request = AsyncMock(return_value=turn_request)
        test_worker._set_turn_request_state = AsyncMock(return_value=True)

        # Act
        result = await test_worker._check_abort_requested()

        # Assert
        assert result is True
        test_worker._set_turn_request_state.assert_called_once_with(
            "turn123", "aborted", message="Aborted by user request"
        )

    @pytest.mark.asyncio
    async def test_check_abort_returns_false_when_not_set(self, test_worker):
        """Test that check returns False when status is not abort_requested."""
        # Arrange
        turn_request = MUDTurnRequest(
            status="ready",
            turn_id="",
        )
        test_worker._get_turn_request = AsyncMock(return_value=turn_request)
        test_worker._set_turn_request_state = AsyncMock()

        # Act
        result = await test_worker._check_abort_requested()

        # Assert
        assert result is False
        test_worker._set_turn_request_state.assert_not_called()
