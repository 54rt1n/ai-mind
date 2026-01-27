# packages/aim-mud/tests/mud_tests/unit/worker/test_startup_recovery.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for startup recovery logic in _announce_presence().

Philosophy: Real objects with mocked external services only.
- Mock: Redis (external service)
- Real: MUDAgentWorker, MUDConfig, all recovery logic
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, call
import uuid

from andimud_worker.worker import MUDAgentWorker
from andimud_worker.config import MUDConfig
from aim_mud_types.helper import _utc_now


@pytest.fixture
def mock_redis_with_expire():
    """Mock Redis with expire tracking."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.exists = AsyncMock(return_value=0)
    redis.hget = AsyncMock(return_value=None)
    redis.hset = AsyncMock(return_value=1)
    redis.hgetall = AsyncMock(return_value={})
    redis.hdel = AsyncMock(return_value=1)
    redis.xadd = AsyncMock(return_value=b"stream-id-123")
    redis.xread = AsyncMock(return_value=[])
    redis.xack = AsyncMock(return_value=1)
    redis.close = AsyncMock()
    redis.expire = AsyncMock(return_value=1)
    redis.eval = AsyncMock(return_value=1)
    return redis


@pytest.fixture
def worker_with_no_ttl(test_mud_config, mock_redis_with_expire, test_config, test_persona):
    """Worker configured with TTL=0 (no expiration)."""
    test_mud_config.turn_request_ttl_seconds = 0
    worker = MUDAgentWorker(
        config=test_mud_config,
        redis_client=mock_redis_with_expire,
        chat_config=test_config,
    )
    worker.persona = test_persona
    # Mock conversation_manager to avoid wakeup logic issues
    worker.conversation_manager = AsyncMock()
    worker.conversation_manager.get_entry_count = AsyncMock(return_value=1)  # Skip wakeup seeding
    return worker


@pytest.fixture
def worker_with_ttl(test_mud_config, mock_redis_with_expire, test_config, test_persona):
    """Worker configured with TTL=120 (expiration enabled)."""
    test_mud_config.turn_request_ttl_seconds = 120
    worker = MUDAgentWorker(
        config=test_mud_config,
        redis_client=mock_redis_with_expire,
        chat_config=test_config,
    )
    worker.persona = test_persona
    # Mock conversation_manager to avoid wakeup logic issues
    worker.conversation_manager = AsyncMock()
    worker.conversation_manager.get_entry_count = AsyncMock(return_value=1)  # Skip wakeup seeding
    return worker


class TestTTLRemoval:
    """Tests for conditional TTL behavior (Part 1)."""

    @pytest.mark.skip(reason="TTL feature removed - set_turn_request_state method doesn't exist")
    @pytest.mark.asyncio
    async def test_set_turn_request_state_no_ttl(self, worker_with_no_ttl):
        """When TTL=0, set_turn_request_state should NOT call expire()."""
        turn_id = str(uuid.uuid4())

        await worker_with_no_ttl.set_turn_request_state(
            turn_id=turn_id,
            status="ready",
            message="Test message"
        )

        # Verify expire was NOT called
        worker_with_no_ttl.redis.expire.assert_not_called()

    @pytest.mark.skip(reason="TTL feature removed - set_turn_request_state method doesn't exist")
    @pytest.mark.asyncio
    async def test_set_turn_request_state_with_ttl(self, worker_with_ttl):
        """When TTL>0, set_turn_request_state should call expire()."""
        turn_id = str(uuid.uuid4())

        await worker_with_ttl.set_turn_request_state(
            turn_id=turn_id,
            status="ready",
            message="Test message"
        )

        # Verify expire was called with correct TTL
        worker_with_ttl.redis.expire.assert_called_once()
        call_args = worker_with_ttl.redis.expire.call_args
        assert call_args[0][1] == 120  # TTL value

    @pytest.mark.skip(reason="TTL feature removed - heartbeat no longer manages TTL")
    @pytest.mark.asyncio
    async def test_heartbeat_no_ttl(self, worker_with_no_ttl):
        """When TTL=0, heartbeat should NOT set TTL (empty TTL arg in Lua script)."""
        import asyncio

        # Mock eval to return success
        worker_with_no_ttl.redis.eval = AsyncMock(return_value=1)
        worker_with_no_ttl.running = True  # Worker is running

        # Set up an initial turn_request in Redis
        turn_id = str(uuid.uuid4())
        worker_with_no_ttl.redis.hgetall.return_value = {
            b"turn_id": turn_id.encode(),
            b"status": b"ready",
            b"heartbeat_at": str(int(_utc_now().timestamp())).encode(),
            b"sequence_id": b"1",
        }

        # Use a shorter heartbeat interval for testing
        worker_with_no_ttl.config.turn_request_heartbeat_seconds = 0.05
        stop_event = asyncio.Event()

        # Start heartbeat
        task = asyncio.create_task(
            worker_with_no_ttl._heartbeat_turn_request(stop_event)
        )

        # Wait for one heartbeat cycle
        await asyncio.sleep(0.15)

        # Stop heartbeat
        stop_event.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Verify eval was called with empty TTL argument (5th positional arg)
        assert worker_with_no_ttl.redis.eval.called
        call_args = worker_with_no_ttl.redis.eval.call_args[0]
        # TTL arg should be empty string when TTL is 0
        assert call_args[4] == ""

    @pytest.mark.skip(reason="TTL feature removed - heartbeat no longer manages TTL")
    @pytest.mark.asyncio
    async def test_heartbeat_with_ttl(self, worker_with_ttl):
        """When TTL>0, heartbeat should set TTL via Lua script."""
        import asyncio

        # Mock eval to return success
        worker_with_ttl.redis.eval = AsyncMock(return_value=1)
        worker_with_ttl.running = True  # Worker is running

        # Set up an initial turn_request in Redis
        turn_id = str(uuid.uuid4())
        worker_with_ttl.redis.hgetall.return_value = {
            b"turn_id": turn_id.encode(),
            b"status": b"ready",
            b"heartbeat_at": str(int(_utc_now().timestamp())).encode(),
            b"sequence_id": b"1",
        }

        # Use a shorter heartbeat interval for testing
        worker_with_ttl.config.turn_request_heartbeat_seconds = 0.05
        stop_event = asyncio.Event()

        # Start heartbeat
        task = asyncio.create_task(
            worker_with_ttl._heartbeat_turn_request(stop_event)
        )

        # Wait for one heartbeat cycle
        await asyncio.sleep(0.15)

        # Stop heartbeat
        stop_event.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Verify eval was called with TTL argument (5th positional arg)
        assert worker_with_ttl.redis.eval.called
        call_args = worker_with_ttl.redis.eval.call_args[0]
        # TTL arg should be "120" when TTL is 120
        assert call_args[4] == "120"
        # TTL is now set inside Lua script, not via separate expire() call
        # The Lua script contains EXPIRE command when TTL arg is non-empty
        lua_script = call_args[0]
        assert "EXPIRE" in lua_script


class TestStartupRecoveryBranch1:
    """Tests for Branch 1: No turn_request → create with status='ready'."""

    @pytest.mark.asyncio
    async def test_no_turn_request_creates_fresh_ready(self, worker_with_no_ttl, mock_redis_with_expire):
        """When no turn_request exists, create fresh 'ready' state."""
        # Setup: No turn_request exists
        mock_redis_with_expire.hgetall.return_value = {}

        # Mock session and world_state to skip wakeup logic
        worker_with_no_ttl.session = None

        await worker_with_no_ttl._announce_presence()

        # Verify eval was called to set turn_request (via set_turn_request_state)
        assert mock_redis_with_expire.eval.called

        # Check the Lua script arguments included status="ready"
        lua_call = mock_redis_with_expire.eval.call_args
        # ARGV in Lua script: expected_turn_id, turn_id, field pairs...
        # Find "status" field in arguments
        args = lua_call[0][3:]  # Skip script, num_keys, key

        # Fields are passed as: "turn_id", <id>, "status", <status>, "heartbeat_at", <time>, ...
        assert "status" in args
        status_idx = args.index("status")
        assert args[status_idx + 1] == "ready"

    @pytest.mark.asyncio
    async def test_no_turn_request_logs_fresh_start(self, worker_with_no_ttl, mock_redis_with_expire, caplog):
        """Verify proper logging when creating fresh turn_request."""
        import logging
        caplog.set_level(logging.INFO)

        # Setup: No turn_request exists
        mock_redis_with_expire.hgetall.return_value = {}
        worker_with_no_ttl.session = None

        await worker_with_no_ttl._announce_presence()

        # Verify log message
        assert "No turn_request found, creating fresh ready state" in caplog.text


class TestStartupRecoveryBranch2:
    """Tests for Branch 2: Problem states → convert to 'retry' or 'fail' with recovery."""

    @pytest.mark.asyncio
    async def test_in_progress_converts_to_retry(self, worker_with_no_ttl, mock_redis_with_expire):
        """When status='in_progress', convert to 'retry' with backoff (below max attempts)."""
        # Setup: turn_request in "in_progress" state
        turn_id = str(uuid.uuid4())
        mock_redis_with_expire.hgetall.return_value = {
            b"turn_id": turn_id.encode(),
            b"status": b"in_progress",
            b"attempt_count": b"0",
            b"sequence_id": b"1",
        }
        worker_with_no_ttl.session = None

        await worker_with_no_ttl._announce_presence()

        # Verify eval was called to set retry state
        assert mock_redis_with_expire.eval.called
        lua_call = mock_redis_with_expire.eval.call_args
        args = lua_call[0][3:]  # Skip script, num_keys, key

        # Verify status was set to "retry" (not "fail")
        assert "status" in args
        status_idx = args.index("status")
        assert args[status_idx + 1] == "retry"

        # Verify attempt_count was incremented to 1
        assert "attempt_count" in args
        attempt_idx = args.index("attempt_count")
        assert args[attempt_idx + 1] == "1"

        # Verify next_attempt_at was set (not empty)
        assert "next_attempt_at" in args
        next_attempt_idx = args.index("next_attempt_at")
        next_attempt_at = args[next_attempt_idx + 1]
        assert next_attempt_at != ""
        # Verify it's a valid Unix timestamp
        assert next_attempt_at.isdigit()
        datetime.fromtimestamp(int(next_attempt_at), tz=timezone.utc)

        # Verify completed_at was set
        assert "completed_at" in args
        completed_at_idx = args.index("completed_at")
        completed_at = args[completed_at_idx + 1]
        assert completed_at != ""
        # Verify it's a valid Unix timestamp
        assert completed_at.isdigit()
        datetime.fromtimestamp(int(completed_at), tz=timezone.utc)

    @pytest.mark.asyncio
    async def test_crashed_converts_to_retry(self, worker_with_no_ttl, mock_redis_with_expire):
        """When status='crashed', convert to 'retry' with backoff (below max attempts)."""
        turn_id = str(uuid.uuid4())
        mock_redis_with_expire.hgetall.return_value = {
            b"turn_id": turn_id.encode(),
            b"status": b"crashed",
            b"attempt_count": b"1",
            b"sequence_id": b"1",
        }
        worker_with_no_ttl.session = None

        await worker_with_no_ttl._announce_presence()

        # Verify status was set to "retry" (not "fail")
        lua_call = mock_redis_with_expire.eval.call_args
        args = lua_call[0][3:]
        assert "status" in args
        status_idx = args.index("status")
        assert args[status_idx + 1] == "retry"

        # Verify attempt_count was incremented to 2
        assert "attempt_count" in args
        attempt_idx = args.index("attempt_count")
        assert args[attempt_idx + 1] == "2"

        # Verify completed_at was set
        assert "completed_at" in args
        completed_at_idx = args.index("completed_at")
        completed_at = args[completed_at_idx + 1]
        assert completed_at != ""
        # Verify it's a valid Unix timestamp
        assert completed_at.isdigit()
        datetime.fromtimestamp(int(completed_at), tz=timezone.utc)

    # Note: ASSIGNED is NOT a problem state - it's work waiting to be picked up.
    # The test for ASSIGNED is in TestStartupRecoveryBranch3 below.

    @pytest.mark.asyncio
    async def test_abort_requested_converts_to_retry(self, worker_with_no_ttl, mock_redis_with_expire):
        """When status='abort_requested', convert to 'retry' with backoff (below max attempts)."""
        turn_id = str(uuid.uuid4())
        mock_redis_with_expire.hgetall.return_value = {
            b"turn_id": turn_id.encode(),
            b"status": b"abort_requested",
            b"attempt_count": b"0",
            b"sequence_id": b"1",
        }
        worker_with_no_ttl.session = None

        await worker_with_no_ttl._announce_presence()

        # Verify conversion to retry (not fail)
        lua_call = mock_redis_with_expire.eval.call_args
        args = lua_call[0][3:]
        assert "status" in args
        status_idx = args.index("status")
        assert args[status_idx + 1] == "retry"

        # Verify completed_at was set
        assert "completed_at" in args

    @pytest.mark.asyncio
    async def test_max_attempts_reached_sets_fail(self, worker_with_no_ttl, mock_redis_with_expire):
        """When max attempts reached, set FAIL status with empty next_attempt_at."""
        worker_with_no_ttl.config.llm_failure_max_attempts = 3
        turn_id = str(uuid.uuid4())
        mock_redis_with_expire.hgetall.return_value = {
            b"turn_id": turn_id.encode(),
            b"status": b"in_progress",
            b"attempt_count": b"2",  # Will increment to 3
            b"sequence_id": b"1",
        }
        worker_with_no_ttl.session = None

        await worker_with_no_ttl._announce_presence()

        # Verify status was set to "fail" (not "retry")
        lua_call = mock_redis_with_expire.eval.call_args
        args = lua_call[0][3:]
        assert "status" in args
        status_idx = args.index("status")
        assert args[status_idx + 1] == "fail"

        # Verify next_attempt_at is NOT included (None values are omitted during serialization)
        # This is correct - when max attempts is reached, there's no next attempt
        assert "next_attempt_at" not in args

        # Verify completed_at was set
        assert "completed_at" in args
        completed_at_idx = args.index("completed_at")
        completed_at = args[completed_at_idx + 1]
        assert completed_at != ""
        # Verify it's a valid Unix timestamp
        assert completed_at.isdigit()
        datetime.fromtimestamp(int(completed_at), tz=timezone.utc)

    @pytest.mark.asyncio
    async def test_exponential_backoff_calculation(self, worker_with_no_ttl, mock_redis_with_expire):
        """Verify exponential backoff is calculated correctly."""
        worker_with_no_ttl.config.llm_failure_backoff_base_seconds = 30
        worker_with_no_ttl.config.llm_failure_backoff_max_seconds = 600

        turn_id = str(uuid.uuid4())
        mock_redis_with_expire.hgetall.return_value = {
            b"turn_id": turn_id.encode(),
            b"status": b"in_progress",
            b"attempt_count": b"1",  # Will increment to 2
            b"sequence_id": b"1",
        }
        worker_with_no_ttl.session = None

        before = _utc_now()
        await worker_with_no_ttl._announce_presence()
        after = _utc_now()

        # Verify next_attempt_at is set
        lua_call = mock_redis_with_expire.eval.call_args
        args = lua_call[0][3:]
        next_attempt_idx = args.index("next_attempt_at")
        next_attempt_str = args[next_attempt_idx + 1]
        next_attempt = datetime.fromtimestamp(int(next_attempt_str), tz=timezone.utc)

        # For attempt 2, backoff should be 30 * 2^(2-1) = 60 seconds
        expected_backoff = 60
        expected_time = before + timedelta(seconds=expected_backoff)

        # Allow 5 second tolerance for test execution time
        assert abs((next_attempt - expected_time).total_seconds()) < 5

    @pytest.mark.asyncio
    async def test_backoff_respects_max(self, worker_with_no_ttl, mock_redis_with_expire):
        """Verify backoff respects max_seconds cap."""
        worker_with_no_ttl.config.llm_failure_backoff_base_seconds = 30
        worker_with_no_ttl.config.llm_failure_backoff_max_seconds = 100
        worker_with_no_ttl.config.llm_failure_max_attempts = 10  # Allow more attempts so we test backoff cap

        turn_id = str(uuid.uuid4())
        mock_redis_with_expire.hgetall.return_value = {
            b"turn_id": turn_id.encode(),
            b"status": b"in_progress",
            b"attempt_count": b"5",  # Will increment to 6, exponential would be huge
            b"sequence_id": b"1",
        }
        worker_with_no_ttl.session = None

        before = _utc_now()
        await worker_with_no_ttl._announce_presence()
        after = _utc_now()

        # Get next_attempt_at
        lua_call = mock_redis_with_expire.eval.call_args
        args = lua_call[0][3:]
        next_attempt_idx = args.index("next_attempt_at")
        next_attempt_str = args[next_attempt_idx + 1]
        next_attempt = datetime.fromtimestamp(int(next_attempt_str), tz=timezone.utc)

        # Should be capped at max_seconds (100)
        expected_time = before + timedelta(seconds=100)

        # Allow 5 second tolerance
        assert abs((next_attempt - expected_time).total_seconds()) < 5

    @pytest.mark.asyncio
    async def test_problem_state_logs_warning(self, worker_with_no_ttl, mock_redis_with_expire, caplog):
        """Verify warning is logged for problem states."""
        import logging
        caplog.set_level(logging.WARNING)

        turn_id = str(uuid.uuid4())
        mock_redis_with_expire.hgetall.return_value = {
            b"turn_id": turn_id.encode(),
            b"status": b"crashed",
            b"attempt_count": b"0",
            b"sequence_id": b"1",
        }
        worker_with_no_ttl.session = None

        await worker_with_no_ttl._announce_presence()

        # Verify warning log (status now shows as enum representation)
        assert "Startup recovery: found turn_request in problem state 'TurnRequestStatus.CRASHED'" in caplog.text


class TestStartupRecoveryBranch3:
    """Tests for Branch 3: Normal states → update heartbeat only."""

    @pytest.mark.asyncio
    async def test_ready_state_updates_heartbeat(self, worker_with_no_ttl, mock_redis_with_expire):
        """When status='ready', only update heartbeat via atomic update."""
        turn_id = str(uuid.uuid4())
        mock_redis_with_expire.hgetall.return_value = {
            b"turn_id": turn_id.encode(),
            b"status": b"ready",
            b"sequence_id": b"1",
        }
        mock_redis_with_expire.eval = AsyncMock(return_value=1)  # Success
        worker_with_no_ttl.session = None

        await worker_with_no_ttl._announce_presence()

        # Verify eval was called (for atomic heartbeat update)
        assert mock_redis_with_expire.eval.called
        call_args = mock_redis_with_expire.eval.call_args[0]

        # Verify it's the atomic heartbeat update Lua script
        lua_script = call_args[0]
        assert "HSET" in lua_script
        assert "heartbeat_at" in lua_script

        # Verify heartbeat_at argument (3rd positional arg) is a valid Unix timestamp
        heartbeat_at = call_args[3]
        assert heartbeat_at.isdigit()
        datetime.fromtimestamp(int(heartbeat_at), tz=timezone.utc)

    @pytest.mark.asyncio
    async def test_done_state_updates_heartbeat(self, worker_with_no_ttl, mock_redis_with_expire):
        """When status='done', only update heartbeat via atomic update."""
        turn_id = str(uuid.uuid4())
        mock_redis_with_expire.hgetall.return_value = {
            b"turn_id": turn_id.encode(),
            b"status": b"done",
            b"sequence_id": b"1",
        }
        mock_redis_with_expire.eval = AsyncMock(return_value=1)  # Success
        worker_with_no_ttl.session = None

        await worker_with_no_ttl._announce_presence()

        # Verify eval was called (for atomic heartbeat update)
        assert mock_redis_with_expire.eval.called

    @pytest.mark.asyncio
    async def test_fail_state_updates_heartbeat(self, worker_with_no_ttl, mock_redis_with_expire):
        """When status='fail', only update heartbeat via atomic update."""
        turn_id = str(uuid.uuid4())
        mock_redis_with_expire.hgetall.return_value = {
            b"turn_id": turn_id.encode(),
            b"status": b"fail",
            b"next_attempt_at": str(int((_utc_now() + timedelta(seconds=60)).timestamp())).encode(),
            b"sequence_id": b"1",
        }
        mock_redis_with_expire.eval = AsyncMock(return_value=1)  # Success
        worker_with_no_ttl.session = None

        await worker_with_no_ttl._announce_presence()

        # Verify eval was called (for atomic heartbeat update)
        assert mock_redis_with_expire.eval.called

    @pytest.mark.asyncio
    async def test_assigned_state_updates_heartbeat(self, worker_with_no_ttl, mock_redis_with_expire):
        """When status='assigned', only update heartbeat - work will be picked up by main loop.

        ASSIGNED is NOT a problem state - it's work that was assigned but the worker
        hasn't started processing yet. On restart, the worker should just pick it up.
        """
        turn_id = str(uuid.uuid4())
        mock_redis_with_expire.hgetall.return_value = {
            b"turn_id": turn_id.encode(),
            b"status": b"assigned",
            b"sequence_id": b"1",
        }
        mock_redis_with_expire.eval = AsyncMock(return_value=1)  # Success
        worker_with_no_ttl.session = None

        await worker_with_no_ttl._announce_presence()

        # Verify eval was called (for atomic heartbeat update)
        assert mock_redis_with_expire.eval.called
        call_args = mock_redis_with_expire.eval.call_args[0]

        # Verify it's the atomic heartbeat update Lua script (NOT status change)
        lua_script = call_args[0]
        assert "HSET" in lua_script
        assert "heartbeat_at" in lua_script

        # Verify status is NOT being changed (ASSIGNED work should be picked up)
        args = call_args[3:]
        assert "status" not in args

    @pytest.mark.asyncio
    async def test_normal_state_logs_info(self, worker_with_no_ttl, mock_redis_with_expire, caplog):
        """Verify info log for normal states."""
        import logging
        caplog.set_level(logging.INFO)

        turn_id = str(uuid.uuid4())
        mock_redis_with_expire.hgetall.return_value = {
            b"turn_id": turn_id.encode(),
            b"status": b"ready",
            b"sequence_id": b"1",
        }
        mock_redis_with_expire.eval = AsyncMock(return_value=1)  # Success
        worker_with_no_ttl.session = None

        await worker_with_no_ttl._announce_presence()

        # Verify info log (status now shows as enum representation)
        assert "Startup: turn_request in normal state 'TurnRequestStatus.READY'" in caplog.text


class TestStartupRecoveryEdgeCases:
    """Edge case tests for startup recovery."""

    @pytest.mark.asyncio
    async def test_missing_attempt_count_defaults_to_zero(self, worker_with_no_ttl, mock_redis_with_expire):
        """When attempt_count is missing, default to 0."""
        turn_id = str(uuid.uuid4())
        mock_redis_with_expire.hgetall.return_value = {
            b"turn_id": turn_id.encode(),
            b"status": b"in_progress",
            b"sequence_id": b"1",
            # No attempt_count field
        }
        worker_with_no_ttl.session = None

        await worker_with_no_ttl._announce_presence()

        # Verify attempt_count was incremented from 0 to 1
        lua_call = mock_redis_with_expire.eval.call_args
        args = lua_call[0][3:]
        attempt_idx = args.index("attempt_count")
        assert args[attempt_idx + 1] == "1"

    @pytest.mark.asyncio
    async def test_missing_turn_id_generates_new(self, worker_with_no_ttl, mock_redis_with_expire):
        """When turn_id is missing, generate a new UUID."""
        old_turn_id = str(uuid.uuid4())
        mock_redis_with_expire.hgetall.return_value = {
            b"turn_id": old_turn_id.encode(),
            b"status": b"in_progress",
            b"sequence_id": b"1",
        }
        worker_with_no_ttl.session = None

        await worker_with_no_ttl._announce_presence()

        # Verify a turn_id was set (should be a valid UUID)
        lua_call = mock_redis_with_expire.eval.call_args
        args = lua_call[0][3:]
        turn_id_idx = args.index("turn_id")
        turn_id = args[turn_id_idx + 1]

        # Verify it's a valid UUID
        uuid.UUID(turn_id)

    @pytest.mark.asyncio
    async def test_status_reason_includes_original_state(self, worker_with_no_ttl, mock_redis_with_expire):
        """Verify status_reason includes the original problem state."""
        turn_id = str(uuid.uuid4())
        mock_redis_with_expire.hgetall.return_value = {
            b"turn_id": turn_id.encode(),
            b"status": b"crashed",
            b"attempt_count": b"0",
            b"sequence_id": b"1",
        }
        worker_with_no_ttl.session = None

        await worker_with_no_ttl._announce_presence()

        # Verify status_reason includes "crashed"
        lua_call = mock_redis_with_expire.eval.call_args
        args = lua_call[0][3:]
        reason_idx = args.index("status_reason")
        status_reason = args[reason_idx + 1]
        assert "crashed" in status_reason.lower()
