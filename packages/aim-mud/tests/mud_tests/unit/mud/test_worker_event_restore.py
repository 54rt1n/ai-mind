# tests/unit/mud/test_worker_event_restore.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for MUD agent worker event restoration logic.

Tests the event position restoration mechanism that saves stream position
before draining and restores it on failure to enable retries.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from andimud_worker.worker import MUDAgentWorker
from andimud_worker.config import MUDConfig
from andimud_worker.exceptions import AbortRequestedException
from andimud_worker.commands.result import CommandResult
from aim_mud_types import MUDSession, MUDEvent, EventType, ActorType


@pytest.fixture
def mud_config():
    """Create a test MUD configuration."""
    return MUDConfig(
        agent_id="test_agent",
        persona_id="test_persona",
        redis_url="redis://localhost:6379",
        spontaneous_check_interval=60.0,
        spontaneous_action_interval=300.0,
        llm_failure_max_attempts=3,
        llm_failure_backoff_base_seconds=5,
        llm_failure_backoff_max_seconds=60,
    )


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    redis = AsyncMock()
    # String operations
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    # Hash operations
    redis.hgetall = AsyncMock(return_value={})
    redis.hget = AsyncMock(return_value=None)
    redis.hset = AsyncMock(return_value=1)
    redis.hdel = AsyncMock(return_value=1)
    # Stream operations
    redis.xinfo_stream = AsyncMock(return_value={"last-generated-id": "0"})
    redis.xread = AsyncMock(return_value=[])
    redis.xadd = AsyncMock(return_value="1-0")
    # Key operations
    redis.expire = AsyncMock(return_value=True)
    # Pipeline
    mock_pipeline = AsyncMock()
    mock_pipeline.execute = AsyncMock(return_value=[])
    redis.pipeline = MagicMock(return_value=mock_pipeline)
    # Cleanup
    redis.aclose = AsyncMock()
    return redis


@pytest.fixture
def worker_with_session(mud_config, mock_redis):
    """Create a worker with initialized session."""
    worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
    worker.session = MUDSession(
        agent_id="test_agent",
        persona_id="test_persona",
        room_id="#123",
        last_event_id="0",
        pending_events=[],
        pending_self_actions=[],
    )
    worker.pending_events = []
    return worker


@pytest.fixture
def sample_events():
    """Sample events for testing."""
    return [
        MUDEvent(
            event_type=EventType.SPEECH,
            actor="Prax",
            actor_type=ActorType.PLAYER,
            room_id="#123",
            content="Hello!",
            timestamp="2026-01-06T12:00:00+00:00",
        ),
        MUDEvent(
            event_type=EventType.EMOTE,
            actor="Prax",
            actor_type=ActorType.PLAYER,
            room_id="#123",
            content="waves",
            timestamp="2026-01-06T12:00:01+00:00",
        ),
    ]


class TestEventPositionRestoreOnException:
    """Test event position restoration when exceptions occur."""

    @pytest.mark.asyncio
    async def test_event_position_restored_on_exception(
        self, worker_with_session, sample_events
    ):
        """Test that event position is restored when process_turn raises exception.

        After the fix, _restore_event_position only rolls back in memory -
        it does NOT persist to Redis.
        """
        worker = worker_with_session

        # Set up initial state
        worker.session.last_event_id = "0"

        # Mock drain to return events and update last_event_id
        async def mock_drain():
            worker.session.last_event_id = "123-0"
            return sample_events

        # Mock process_turn to raise exception
        async def mock_process_turn(events):
            raise RuntimeError("LLM API failure")

        with patch.object(
            worker, "_drain_with_settle", side_effect=mock_drain
        ), patch.object(
            worker, "process_turn", side_effect=mock_process_turn
        ), patch.object(
            worker, "_update_agent_profile", new_callable=AsyncMock
        ) as mock_update:
            # Call _restore_event_position directly with saved ID
            saved_id = "0"
            await worker._restore_event_position(saved_id)

            # Assert position restored IN MEMORY
            assert worker.session.last_event_id == "0"
            assert worker.pending_events == []
            assert worker.session.pending_self_actions == []

            # CRITICAL: Redis persistence NOT called (this is the fix)
            mock_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_event_position_restored_in_worker_loop_on_exception(
        self, worker_with_session, sample_events, mock_redis
    ):
        """Test event position restoration in worker loop when exception occurs."""
        worker = worker_with_session

        # Mock turn request
        turn_request = {
            "turn_id": "turn_123",
            "reason": "events",
            "attempt_count": "0",
        }
        mock_redis.hgetall.return_value = turn_request

        # Track state changes
        state_changes = []

        async def mock_set_state(turn_id, status, message=None, extra_fields=None, expected_turn_id=None):
            state_changes.append({"turn_id": turn_id, "status": status})

        # Mock drain to return events and update last_event_id
        async def mock_drain():
            worker.session.last_event_id = "123-0"
            worker.pending_events = sample_events
            return sample_events

        # Mock command registry to return incomplete result (fall through to process_turn)
        mock_result = CommandResult(complete=False, flush_drain=False)

        # Mock process_turn to raise exception
        async def mock_process_turn(events):
            raise RuntimeError("LLM API failure")

        with patch.object(
            worker, "_get_turn_request", return_value=turn_request
        ), patch.object(
            worker, "_set_turn_request_state", side_effect=mock_set_state
        ), patch.object(
            worker, "_drain_with_settle", side_effect=mock_drain
        ), patch.object(
            worker.command_registry, "execute", return_value=mock_result
        ), patch.object(
            worker, "process_turn", side_effect=mock_process_turn
        ), patch.object(
            worker, "_update_agent_profile", new_callable=AsyncMock
        ) as mock_update, patch.object(
            worker, "_heartbeat_turn_request", new_callable=AsyncMock
        ):
            # Set initial state
            original_id = worker.session.last_event_id

            try:
                # Run one iteration of the worker loop (will fail)
                # We need to simulate the try-except block from worker.py lines 298-372
                saved_event_id = worker.session.last_event_id
                worker.pending_events = await worker._drain_with_settle()

                # Execute command (returns incomplete, so falls through)
                result = await worker.command_registry.execute(worker, **turn_request)

                # Try to process turn (raises exception)
                await worker.process_turn(worker.pending_events)
            except RuntimeError:
                # Simulate exception handler that restores position
                await worker._restore_event_position(saved_event_id)

            # Assert position was restored to original
            assert worker.session.last_event_id == original_id
            assert worker.pending_events == []
            assert worker.session.pending_self_actions == []


class TestEventPositionRestoreOnAbort:
    """Test event position restoration on AbortRequestedException."""

    @pytest.mark.asyncio
    async def test_event_position_restored_on_abort(
        self, worker_with_session, sample_events
    ):
        """Test that event position is restored when command raises AbortRequestedException."""
        worker = worker_with_session

        # Set up initial state
        original_id = "0"
        worker.session.last_event_id = original_id

        # Mock drain to return events and update last_event_id
        async def mock_drain():
            worker.session.last_event_id = "456-0"
            worker.pending_events = sample_events
            return sample_events

        # Mock command to raise AbortRequestedException
        async def mock_execute(worker, **kwargs):
            raise AbortRequestedException("User aborted turn")

        with patch.object(
            worker, "_drain_with_settle", side_effect=mock_drain
        ), patch.object(
            worker.command_registry, "execute", side_effect=mock_execute
        ):
            try:
                saved_event_id = worker.session.last_event_id
                worker.pending_events = await worker._drain_with_settle()
                await worker.command_registry.execute(worker)
            except AbortRequestedException:
                # Simulate abort handler
                await worker._restore_event_position(saved_event_id)

            # Assert position was restored
            assert worker.session.last_event_id == original_id
            assert worker.pending_events == []
            assert worker.session.pending_self_actions == []


class TestEventPositionNotRestoredOnSuccess:
    """Test that event position is NOT restored on successful processing."""

    @pytest.mark.asyncio
    async def test_event_position_not_restored_on_success(
        self, worker_with_session, sample_events
    ):
        """Test that event position remains at post-drain value on success."""
        worker = worker_with_session

        # Set up initial state
        worker.session.last_event_id = "0"

        # Mock drain to return events and update last_event_id
        async def mock_drain():
            worker.session.last_event_id = "789-0"
            worker.pending_events = sample_events
            return sample_events

        # Mock process_turn to succeed
        async def mock_process_turn(events):
            pass  # Success

        with patch.object(
            worker, "_drain_with_settle", side_effect=mock_drain
        ), patch.object(
            worker, "process_turn", side_effect=mock_process_turn
        ):
            saved_event_id = worker.session.last_event_id
            worker.pending_events = await worker._drain_with_settle()

            # Process turn successfully
            await worker.process_turn(worker.pending_events)

            # On success, we don't call _restore_event_position
            # Assert position remains at post-drain value
            assert worker.session.last_event_id == "789-0"
            assert worker.session.last_event_id != saved_event_id


class TestEventPositionFlushDrain:
    """Test event position handling with flush_drain flag."""

    @pytest.mark.asyncio
    async def test_event_position_not_restored_when_flush_drain_true(
        self, worker_with_session, sample_events
    ):
        """Test that event position is NOT restored when flush_drain=True."""
        worker = worker_with_session

        # Set up initial state
        original_id = "0"
        worker.session.last_event_id = original_id

        # Mock drain to return events and update last_event_id
        async def mock_drain():
            worker.session.last_event_id = "999-0"
            worker.pending_events = sample_events
            return sample_events

        # Mock command to return flush_drain=True
        mock_result = CommandResult(complete=True, flush_drain=True)

        with patch.object(
            worker, "_drain_with_settle", side_effect=mock_drain
        ), patch.object(
            worker.command_registry, "execute", return_value=mock_result
        ):
            # Simulate worker loop logic
            saved_event_id = worker.session.last_event_id
            worker.pending_events = await worker._drain_with_settle()
            result = await worker.command_registry.execute(worker)

            # When flush_drain=True, saved_event_id should be set to None
            if result.flush_drain:
                worker.pending_events = []
                saved_event_id = None

            # Now if we try to restore, it should skip
            await worker._restore_event_position(saved_event_id)

            # Assert position was NOT restored (stayed at post-drain value)
            assert worker.session.last_event_id == "999-0"


class TestRestoreClearsPendingBuffers:
    """Test that restoration clears all pending buffers."""

    @pytest.mark.asyncio
    async def test_restore_clears_pending_self_actions(
        self, worker_with_session, sample_events
    ):
        """Test that both pending_events and pending_self_actions are cleared on restore."""
        worker = worker_with_session

        # Populate both buffers
        worker.pending_events = sample_events.copy()
        worker.session.pending_self_actions = [sample_events[0]]
        worker.session.last_event_id = "100-0"

        # Restore to earlier position
        await worker._restore_event_position("50-0")

        # Assert both buffers cleared
        assert worker.pending_events == []
        assert worker.session.pending_self_actions == []
        assert worker.session.last_event_id == "50-0"


class TestRestoreWithZeroEventId:
    """Test restoration with zero event ID (stream start)."""

    @pytest.mark.asyncio
    async def test_restore_with_zero_event_id(self, worker_with_session, sample_events):
        """Test that restoration works correctly when restoring to '0'.

        After the fix, _restore_event_position only rolls back in memory -
        it does NOT persist to Redis.
        """
        worker = worker_with_session

        # Start at "0"
        worker.session.last_event_id = "0"

        # Mock drain to advance position
        async def mock_drain():
            worker.session.last_event_id = "123-0"
            worker.pending_events = sample_events
            return sample_events

        with patch.object(
            worker, "_drain_with_settle", side_effect=mock_drain
        ), patch.object(
            worker, "_update_agent_profile", new_callable=AsyncMock
        ) as mock_update:
            # Save initial position
            saved_id = worker.session.last_event_id  # "0"

            # Drain events (advances to 123-0)
            await worker._drain_with_settle()

            # Simulate failure and restore
            await worker._restore_event_position(saved_id)

            # Assert restored to "0" IN MEMORY
            assert worker.session.last_event_id == "0"

            # CRITICAL: Redis persistence NOT called (this is the fix)
            mock_update.assert_not_called()


class TestRestoreDoesNotUpdateRedis:
    """Test that restoration does NOT update Redis profile (fix validation)."""

    @pytest.mark.asyncio
    async def test_restore_does_not_update_redis_profile(self, worker_with_session):
        """Test that _restore_event_position does NOT call _update_agent_profile.

        After the fix, _restore_event_position only rolls back in memory -
        it does NOT persist to Redis. Redis persistence only happens for
        confirmed speech turns.
        """
        worker = worker_with_session

        # Set up state
        worker.session.last_event_id = "200-0"
        worker.pending_events = []
        worker.session.pending_self_actions = []

        with patch.object(
            worker, "_update_agent_profile", new_callable=AsyncMock
        ) as mock_update:
            # Restore to earlier position
            await worker._restore_event_position("150-0")

            # Assert position rolled back IN MEMORY
            assert worker.session.last_event_id == "150-0"

            # CRITICAL: Redis persistence NOT called (this is the fix)
            mock_update.assert_not_called()


class TestRestoreSkipsWhenSavedIdNone:
    """Test that restoration is skipped when saved_event_id is None."""

    @pytest.mark.asyncio
    async def test_restore_skips_when_saved_id_none(self, worker_with_session):
        """Test that _restore_event_position does nothing when saved_event_id is None."""
        worker = worker_with_session

        # Set up state
        original_id = "300-0"
        worker.session.last_event_id = original_id
        worker.pending_events = ["event1", "event2"]

        with patch.object(
            worker, "_update_agent_profile", new_callable=AsyncMock
        ) as mock_update:
            # Call with None (simulates flush_drain=True case)
            await worker._restore_event_position(None)

            # Assert no changes made
            assert worker.session.last_event_id == original_id
            assert len(worker.pending_events) == 2  # Not cleared
            mock_update.assert_not_called()
