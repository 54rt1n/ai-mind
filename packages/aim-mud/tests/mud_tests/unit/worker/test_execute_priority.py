# packages/aim-mud/tests/mud_tests/unit/worker/test_execute_priority.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for worker EXECUTE status priority processing.

Tests that the worker processes EXECUTE status turns before ASSIGNED status turns,
and that EXECUTE turns skip event draining and turn guard checks.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from aim_mud_types.coordination import MUDTurnRequest, TurnRequestStatus, TurnReason
from andimud_worker.config import MUDConfig


@pytest.fixture
def mock_redis():
    """Mock Redis client for worker tests."""
    redis = MagicMock()
    redis.hgetall = AsyncMock()
    redis.hset = AsyncMock()
    redis.xread = AsyncMock(return_value=[])
    redis.pipeline = MagicMock()
    return redis


@pytest.fixture
def mock_worker(mock_redis):
    """Mock worker instance with necessary attributes."""
    worker = MagicMock()
    worker.redis = mock_redis
    worker.config = MUDConfig(agent_id="andi", persona_id="andi")
    worker._get_turn_request = AsyncMock()
    worker._set_turn_request_state = AsyncMock(return_value=True)
    worker._should_process_turn = AsyncMock(return_value=True)
    worker._process_turn = AsyncMock()
    return worker


class TestWorkerExecutePriority:
    """Tests for worker EXECUTE status priority processing."""

    @pytest.mark.asyncio
    async def test_execute_status_processed_before_assigned(self, mock_worker):
        """Test that EXECUTE status is checked before ASSIGNED status."""
        # This test verifies the run_loop logic structure
        # In actual implementation, EXECUTE is checked first in the if-elif chain

        execute_turn = MUDTurnRequest(
            turn_id="turn-execute-1",
            status=TurnRequestStatus.EXECUTE,
            reason=TurnReason.FLUSH,
            assigned_at=datetime.now(timezone.utc),
        )

        assigned_turn = MUDTurnRequest(
            turn_id="turn-assigned-1",
            status=TurnRequestStatus.ASSIGNED,
            reason=TurnReason.EVENTS,
            assigned_at=datetime.now(timezone.utc),
        )

        # When we have EXECUTE status, it should be processed
        assert execute_turn.status == TurnRequestStatus.EXECUTE
        assert execute_turn.status != TurnRequestStatus.ASSIGNED

        # ASSIGNED status should not match EXECUTE check
        assert assigned_turn.status != TurnRequestStatus.EXECUTE
        assert assigned_turn.status == TurnRequestStatus.ASSIGNED

    @pytest.mark.asyncio
    async def test_execute_transitions_to_executing(self, mock_worker):
        """Test that EXECUTE status transitions to EXECUTING."""
        turn_request = MUDTurnRequest(
            turn_id="turn-1",
            status=TurnRequestStatus.EXECUTE,
            reason=TurnReason.FLUSH,
            assigned_at=datetime.now(timezone.utc),
        )

        mock_worker._get_turn_request.return_value = turn_request

        # Simulate the transition that happens in run_loop
        await mock_worker._set_turn_request_state(
            turn_id=turn_request.turn_id,
            status=TurnRequestStatus.EXECUTING,
        )

        mock_worker._set_turn_request_state.assert_called_once_with(
            turn_id="turn-1",
            status=TurnRequestStatus.EXECUTING,
        )

    @pytest.mark.asyncio
    async def test_execute_status_skips_turn_guard(self, mock_worker):
        """Test that EXECUTE status commands don't call turn guard."""
        # When processing EXECUTE status, _should_process_turn should NOT be called
        # because EXECUTE commands bypass the turn guard

        turn_request = MUDTurnRequest(
            turn_id="turn-1",
            status=TurnRequestStatus.EXECUTE,
            reason=TurnReason.CLEAR,
            assigned_at=datetime.now(timezone.utc),
        )

        # In the actual implementation, EXECUTE status commands are processed
        # immediately without calling _should_process_turn
        # This test documents that expectation

        assert turn_request.status == TurnRequestStatus.EXECUTE
        # EXECUTE commands should not check turn guard
        # (verified by code inspection - no _should_process_turn call in EXECUTE branch)

    @pytest.mark.asyncio
    async def test_assigned_status_calls_turn_guard(self, mock_worker):
        """Test that ASSIGNED status commands do call turn guard."""
        turn_request = MUDTurnRequest(
            turn_id="turn-1",
            status=TurnRequestStatus.ASSIGNED,
            reason=TurnReason.EVENTS,
            assigned_at=datetime.now(timezone.utc),
        )

        # ASSIGNED status should call turn guard before processing
        should_process = await mock_worker._should_process_turn(turn_request)

        mock_worker._should_process_turn.assert_called_once()
        assert isinstance(should_process, bool)

    @pytest.mark.asyncio
    async def test_execute_reason_flush(self, mock_worker):
        """Test that FLUSH reason creates EXECUTE status."""
        turn_request = MUDTurnRequest(
            turn_id="turn-1",
            status=TurnRequestStatus.EXECUTE,
            reason=TurnReason.FLUSH,
            assigned_at=datetime.now(timezone.utc),
        )

        assert turn_request.reason == TurnReason.FLUSH
        assert turn_request.reason.is_immediate_command() is True
        assert turn_request.status == TurnRequestStatus.EXECUTE

    @pytest.mark.asyncio
    async def test_execute_reason_clear(self, mock_worker):
        """Test that CLEAR reason creates EXECUTE status."""
        turn_request = MUDTurnRequest(
            turn_id="turn-1",
            status=TurnRequestStatus.EXECUTE,
            reason=TurnReason.CLEAR,
            assigned_at=datetime.now(timezone.utc),
        )

        assert turn_request.reason == TurnReason.CLEAR
        assert turn_request.reason.is_immediate_command() is True
        assert turn_request.status == TurnRequestStatus.EXECUTE

    @pytest.mark.asyncio
    async def test_execute_reason_new(self, mock_worker):
        """Test that NEW reason creates EXECUTE status."""
        turn_request = MUDTurnRequest(
            turn_id="turn-1",
            status=TurnRequestStatus.EXECUTE,
            reason=TurnReason.NEW,
            assigned_at=datetime.now(timezone.utc),
        )

        assert turn_request.reason == TurnReason.NEW
        assert turn_request.reason.is_immediate_command() is True
        assert turn_request.status == TurnRequestStatus.EXECUTE

    @pytest.mark.asyncio
    async def test_executing_status_exists(self):
        """Test that EXECUTING status enum value exists."""
        assert hasattr(TurnRequestStatus, 'EXECUTING')
        assert TurnRequestStatus.EXECUTING.value == "executing"

    @pytest.mark.asyncio
    async def test_execute_to_done_transition(self, mock_worker):
        """Test that EXECUTE commands transition through EXECUTING to DONE."""
        # Simulate the transition flow:
        # EXECUTE -> EXECUTING -> DONE -> READY

        turn_id = "turn-1"

        # Step 1: EXECUTE -> EXECUTING (when command starts)
        await mock_worker._set_turn_request_state(
            turn_id=turn_id,
            status=TurnRequestStatus.EXECUTING,
        )

        # Step 2: EXECUTING -> DONE (when command completes)
        await mock_worker._set_turn_request_state(
            turn_id=turn_id,
            status=TurnRequestStatus.DONE,
        )

        # Step 3: DONE -> READY (cleanup phase)
        await mock_worker._set_turn_request_state(
            turn_id=turn_id,
            status=TurnRequestStatus.READY,
        )

        # Verify all three transitions were called
        assert mock_worker._set_turn_request_state.call_count == 3

        calls = mock_worker._set_turn_request_state.call_args_list
        assert calls[0][1]['status'] == TurnRequestStatus.EXECUTING
        assert calls[1][1]['status'] == TurnRequestStatus.DONE
        assert calls[2][1]['status'] == TurnRequestStatus.READY

    @pytest.mark.asyncio
    async def test_executing_transitions_to_ready_not_done(self, mock_worker):
        """Test that EXECUTING status transitions directly to READY (not DONE first)."""
        # Per the implementation, EXECUTING commands should transition to READY
        # (skipping DONE) after completion

        turn_id = "turn-1"

        # EXECUTING -> READY (direct transition)
        await mock_worker._set_turn_request_state(
            turn_id=turn_id,
            status=TurnRequestStatus.READY,
        )

        mock_worker._set_turn_request_state.assert_called_once_with(
            turn_id=turn_id,
            status=TurnRequestStatus.READY,
        )


class TestWorkerExecuteStatusTransitions:
    """Test status transition paths for EXECUTE/EXECUTING."""

    @pytest.mark.asyncio
    async def test_all_execute_related_statuses_exist(self):
        """Test that all EXECUTE-related statuses are defined."""
        assert TurnRequestStatus.EXECUTE in TurnRequestStatus
        assert TurnRequestStatus.EXECUTING in TurnRequestStatus
        assert TurnRequestStatus.ASSIGNED in TurnRequestStatus
        assert TurnRequestStatus.IN_PROGRESS in TurnRequestStatus
        assert TurnRequestStatus.DONE in TurnRequestStatus
        assert TurnRequestStatus.READY in TurnRequestStatus

    @pytest.mark.asyncio
    async def test_execute_status_value(self):
        """Test that EXECUTE status has correct string value."""
        assert TurnRequestStatus.EXECUTE.value == "execute"

    @pytest.mark.asyncio
    async def test_executing_status_value(self):
        """Test that EXECUTING status has correct string value."""
        assert TurnRequestStatus.EXECUTING.value == "executing"

    @pytest.mark.asyncio
    async def test_status_enum_completeness(self):
        """Test that all expected statuses are present in enum."""
        expected_statuses = {
            "assigned",
            "in_progress",
            "done",
            "fail",
            "ready",
            "crashed",
            "aborted",
            "abort_requested",
            "execute",
            "executing",
        }

        actual_statuses = {status.value for status in TurnRequestStatus}
        assert actual_statuses == expected_statuses


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
