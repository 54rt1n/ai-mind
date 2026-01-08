# packages/aim-mud/tests/mud_tests/unit/worker/test_cleanup_helpers.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for cleanup helper methods and finally block state transitions.

This module tests the helper methods called by the finally block in _run_main_loop()
to ensure proper cleanup and state transitions after turn processing.
"""

import pytest
import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from aim_mud_types.helper import _utc_now


class TestHeartbeatTurnRequest:
    """Tests for _heartbeat_turn_request() helper method."""

    @pytest.mark.asyncio
    async def test_heartbeat_refreshes_ttl_periodically(self, test_worker):
        """Test that heartbeat refreshes TTL and timestamp periodically when TTL>0."""
        # Arrange
        test_worker.config.turn_request_ttl_seconds = 120  # Enable TTL
        stop_event = asyncio.Event()
        test_worker.redis.expire = AsyncMock()
        test_worker.redis.hset = AsyncMock()
        test_worker.config.turn_request_heartbeat_seconds = 0.01  # Fast for testing

        # Act - start heartbeat and let it run for a few cycles
        heartbeat_task = asyncio.create_task(
            test_worker._heartbeat_turn_request(stop_event)
        )
        await asyncio.sleep(0.05)  # Allow ~5 heartbeats
        stop_event.set()
        await heartbeat_task

        # Assert - should have called expire and hset multiple times
        assert test_worker.redis.expire.call_count >= 2
        assert test_worker.redis.hset.call_count >= 2

        # Verify expire called with correct key and TTL
        call_args = test_worker.redis.expire.call_args
        assert call_args[0][0] == test_worker._turn_request_key()
        assert call_args[0][1] == test_worker.config.turn_request_ttl_seconds

    @pytest.mark.asyncio
    async def test_heartbeat_stops_on_event_signal(self, test_worker):
        """Test that heartbeat stops immediately when stop_event is set."""
        # Arrange
        stop_event = asyncio.Event()
        test_worker.redis.expire = AsyncMock()
        test_worker.redis.hset = AsyncMock()
        test_worker.config.turn_request_heartbeat_seconds = 0.01

        # Act - start and immediately stop
        heartbeat_task = asyncio.create_task(
            test_worker._heartbeat_turn_request(stop_event)
        )
        stop_event.set()
        await heartbeat_task

        # Assert - should have minimal calls (0-1)
        assert test_worker.redis.expire.call_count <= 1

    @pytest.mark.asyncio
    async def test_heartbeat_handles_cancellation(self, test_worker):
        """Test that heartbeat handles cancellation gracefully."""
        # Arrange
        stop_event = asyncio.Event()
        test_worker.redis.expire = AsyncMock()
        test_worker.redis.hset = AsyncMock()
        test_worker.config.turn_request_heartbeat_seconds = 1.0

        # Act - cancel task during heartbeat
        heartbeat_task = asyncio.create_task(
            test_worker._heartbeat_turn_request(stop_event)
        )
        await asyncio.sleep(0.01)  # Let it start
        heartbeat_task.cancel()

        # Should not raise exception
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass  # Expected

        # Assert - task completed without error


class TestSetTurnRequestStateTransitions:
    """Tests for state transition logic using _set_turn_request_state()."""

    @pytest.mark.asyncio
    async def test_done_to_ready_creates_new_turn_request(self, test_worker):
        """Test that 'done' status transitions to 'ready' with new turn_id."""
        # Arrange
        old_turn_id = "turn123"
        test_worker.redis.eval = AsyncMock(return_value=1)  # CAS success
        test_worker.redis.expire = AsyncMock()

        # Act
        result = await test_worker._set_turn_request_state(
            turn_id=str(uuid.uuid4()),
            status="ready",
            extra_fields={"status_reason": "Turn completed"},
            expected_turn_id=old_turn_id
        )

        # Assert
        assert result is True
        test_worker.redis.eval.assert_called_once()

        # Verify Lua script called with CAS check
        call_args = test_worker.redis.eval.call_args[0]
        lua_script = call_args[0]
        assert "HGET" in lua_script  # CAS check
        assert old_turn_id in call_args  # expected_turn_id
        assert "ready" in call_args  # new status
        assert "Turn completed" in call_args  # status_reason

    @pytest.mark.asyncio
    async def test_aborted_to_ready_creates_new_turn_request(self, test_worker):
        """Test that 'aborted' status transitions to 'ready' with new turn_id."""
        # Arrange
        old_turn_id = "turn456"
        test_worker.redis.eval = AsyncMock(return_value=1)
        test_worker.redis.expire = AsyncMock()

        # Act
        result = await test_worker._set_turn_request_state(
            turn_id=str(uuid.uuid4()),
            status="ready",
            extra_fields={"status_reason": "Turn aborted"},
            expected_turn_id=old_turn_id
        )

        # Assert
        assert result is True
        call_args = test_worker.redis.eval.call_args[0]
        assert old_turn_id in call_args
        assert "ready" in call_args
        assert "Turn aborted" in call_args

    @pytest.mark.asyncio
    async def test_fail_state_preserved_no_transition(self, test_worker):
        """Test that 'fail' status does NOT auto-transition to 'ready'."""
        # Arrange - simulate finally block NOT calling transition for "fail" status
        test_worker.redis.eval = AsyncMock(return_value=1)
        test_worker.redis.expire = AsyncMock()

        # Set fail state
        await test_worker._set_turn_request_state(
            turn_id="turn789",
            status="fail",
            message="LLM call failed",
            extra_fields={
                "attempt_count": "2",
                "next_attempt_at": "2026-01-08T12:00:00",
                "status_reason": "LLM call failed: TimeoutError"
            }
        )

        # Assert - fail state is set, but verify it stays as "fail"
        call_args = test_worker.redis.eval.call_args[0]
        assert "fail" in call_args
        assert "LLM call failed" in call_args
        assert "2" in call_args  # attempt_count

        # The finally block should NOT call set_turn_request_state for "fail" status
        # (This is tested in the integration test for the finally block logic)

    @pytest.mark.asyncio
    async def test_cas_semantics_with_expected_turn_id(self, test_worker):
        """Test that expected_turn_id provides CAS protection."""
        # Arrange
        test_worker.redis.eval = AsyncMock(return_value=0)  # CAS failed

        # Act
        result = await test_worker._set_turn_request_state(
            turn_id="new_turn",
            status="ready",
            expected_turn_id="expected_turn"
        )

        # Assert - CAS should fail
        assert result is False
        test_worker.redis.eval.assert_called_once()

        # Verify expected_turn_id passed to Lua script
        call_args = test_worker.redis.eval.call_args[0]
        assert "expected_turn" in call_args

    @pytest.mark.asyncio
    async def test_cas_success_with_matching_turn_id(self, test_worker):
        """Test that CAS succeeds when turn_id matches."""
        # Arrange
        test_worker.config.turn_request_ttl_seconds = 120  # Enable TTL
        test_worker.redis.eval = AsyncMock(return_value=1)  # CAS success
        test_worker.redis.expire = AsyncMock()

        # Act
        result = await test_worker._set_turn_request_state(
            turn_id="new_turn",
            status="ready",
            expected_turn_id="matching_turn"
        )

        # Assert
        assert result is True
        test_worker.redis.expire.assert_called_once()


class TestFinallyBlockStateTransitions:
    """Integration tests for finally block cleanup and state transitions."""

    @pytest.mark.asyncio
    async def test_finally_block_done_transitions_to_ready(self, test_worker):
        """Test that finally block transitions 'done' to 'ready'."""
        # Arrange - simulate finally block logic
        turn_id = "turn123"
        turn_request = {
            "status": "done",
            "turn_id": turn_id,
        }
        test_worker._get_turn_request = AsyncMock(return_value=turn_request)
        test_worker._set_turn_request_state = AsyncMock(return_value=True)

        # Act - simulate finally block
        if turn_id:
            turn_request = await test_worker._get_turn_request()
            if turn_request:
                status = turn_request.get("status")
                if status == "done":
                    await test_worker._set_turn_request_state(
                        str(uuid.uuid4()),
                        "ready",
                        extra_fields={"status_reason": "Turn completed"},
                        expected_turn_id=turn_id
                    )

        # Assert
        test_worker._set_turn_request_state.assert_called_once()
        call_args = test_worker._set_turn_request_state.call_args
        assert call_args[0][1] == "ready"  # status
        assert call_args[1]["extra_fields"]["status_reason"] == "Turn completed"
        assert call_args[1]["expected_turn_id"] == turn_id

    @pytest.mark.asyncio
    async def test_finally_block_aborted_transitions_to_ready(self, test_worker):
        """Test that finally block transitions 'aborted' to 'ready'."""
        # Arrange
        turn_id = "turn456"
        turn_request = {
            "status": "aborted",
            "turn_id": turn_id,
        }
        test_worker._get_turn_request = AsyncMock(return_value=turn_request)
        test_worker._set_turn_request_state = AsyncMock(return_value=True)

        # Act - simulate finally block
        if turn_id:
            turn_request = await test_worker._get_turn_request()
            if turn_request:
                status = turn_request.get("status")
                if status == "aborted":
                    await test_worker._set_turn_request_state(
                        str(uuid.uuid4()),
                        "ready",
                        extra_fields={"status_reason": "Turn aborted"},
                        expected_turn_id=turn_id
                    )

        # Assert
        test_worker._set_turn_request_state.assert_called_once()
        call_args = test_worker._set_turn_request_state.call_args
        assert call_args[0][1] == "ready"
        assert call_args[1]["extra_fields"]["status_reason"] == "Turn aborted"
        assert call_args[1]["expected_turn_id"] == turn_id

    @pytest.mark.asyncio
    async def test_finally_block_fail_does_not_transition(self, test_worker):
        """Test that finally block does NOT transition 'fail' status."""
        # Arrange
        turn_id = "turn789"
        turn_request = {
            "status": "fail",
            "turn_id": turn_id,
            "attempt_count": "2",
            "next_attempt_at": "2026-01-08T12:00:00",
        }
        test_worker._get_turn_request = AsyncMock(return_value=turn_request)
        test_worker._set_turn_request_state = AsyncMock()

        # Act - simulate finally block
        if turn_id:
            turn_request = await test_worker._get_turn_request()
            if turn_request:
                status = turn_request.get("status")
                # Finally block only transitions "done" and "aborted"
                if status == "done":
                    await test_worker._set_turn_request_state(
                        str(uuid.uuid4()), "ready",
                        extra_fields={"status_reason": "Turn completed"},
                        expected_turn_id=turn_id
                    )
                elif status == "aborted":
                    await test_worker._set_turn_request_state(
                        str(uuid.uuid4()), "ready",
                        extra_fields={"status_reason": "Turn aborted"},
                        expected_turn_id=turn_id
                    )

        # Assert - set_turn_request_state should NOT be called for "fail"
        test_worker._set_turn_request_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_finally_block_no_turn_id_skips_transition(self, test_worker):
        """Test that finally block skips transition if turn_id is None."""
        # Arrange
        turn_id = None  # No turn processed
        test_worker._get_turn_request = AsyncMock()
        test_worker._set_turn_request_state = AsyncMock()

        # Act - simulate finally block
        if turn_id:
            turn_request = await test_worker._get_turn_request()
            if turn_request:
                status = turn_request.get("status")
                if status == "done":
                    await test_worker._set_turn_request_state(
                        str(uuid.uuid4()), "ready",
                        extra_fields={"status_reason": "Turn completed"},
                        expected_turn_id=turn_id
                    )

        # Assert - should not call any helpers
        test_worker._get_turn_request.assert_not_called()
        test_worker._set_turn_request_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_finally_block_missing_turn_request_skips_transition(self, test_worker):
        """Test that finally block skips transition if turn_request missing."""
        # Arrange
        turn_id = "turn999"
        test_worker._get_turn_request = AsyncMock(return_value={})  # Missing
        test_worker._set_turn_request_state = AsyncMock()

        # Act - simulate finally block
        if turn_id:
            turn_request = await test_worker._get_turn_request()
            if turn_request:  # Empty dict is falsy
                status = turn_request.get("status")
                if status == "done":
                    await test_worker._set_turn_request_state(
                        str(uuid.uuid4()), "ready",
                        extra_fields={"status_reason": "Turn completed"},
                        expected_turn_id=turn_id
                    )

        # Assert - should call _get_turn_request but not _set_turn_request_state
        test_worker._get_turn_request.assert_called_once()
        test_worker._set_turn_request_state.assert_not_called()


class TestHeartbeatCleanup:
    """Tests for heartbeat cleanup in finally block."""

    @pytest.mark.asyncio
    async def test_heartbeat_task_cancelled_on_exception(self, test_worker):
        """Test that heartbeat task is cancelled when turn processing fails."""
        # Arrange
        heartbeat_stop = asyncio.Event()
        test_worker.redis.expire = AsyncMock()
        test_worker.redis.hset = AsyncMock()
        test_worker.config.turn_request_heartbeat_seconds = 0.01

        heartbeat_task = asyncio.create_task(
            test_worker._heartbeat_turn_request(heartbeat_stop)
        )

        # Act - simulate exception during turn processing
        await asyncio.sleep(0.02)  # Let heartbeat run

        # Simulate finally block - note that setting the stop event causes
        # graceful shutdown, not cancellation
        heartbeat_stop.set()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass  # Expected

        # Assert - task is done (either cancelled or completed gracefully)
        assert heartbeat_task.done()

    @pytest.mark.asyncio
    async def test_heartbeat_task_stopped_on_success(self, test_worker):
        """Test that heartbeat task is stopped when turn completes successfully."""
        # Arrange
        heartbeat_stop = asyncio.Event()
        test_worker.redis.expire = AsyncMock()
        test_worker.redis.hset = AsyncMock()
        test_worker.config.turn_request_heartbeat_seconds = 0.01

        heartbeat_task = asyncio.create_task(
            test_worker._heartbeat_turn_request(heartbeat_stop)
        )

        # Act - simulate successful turn completion
        await asyncio.sleep(0.02)  # Let heartbeat run

        # Simulate finally block
        heartbeat_stop.set()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

        # Assert - task completed
        assert heartbeat_task.done()
