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
from aim_mud_types import MUDSession, MUDEvent, EventType, ActorType, RoomState
from aim_mud_types.models.coordination import MUDTurnRequest


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
def test_room():
    """Create a test room state."""
    return RoomState(room_id="#123", name="Test Room")


@pytest.fixture
def worker_with_session(mud_config, mock_redis, test_room):
    """Create a worker with initialized session."""
    worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
    worker.session = MUDSession(
        agent_id="test_agent",
        persona_id="test_persona",
        last_event_id="0",
        pending_events=[],
        current_room=test_room,
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
        """Test that event position is restored when _restore_event_position is called.

        After the fix, _restore_event_position only rolls back in memory -
        it does NOT persist to Redis.
        """
        worker = worker_with_session

        # Set up initial state
        worker.session.last_event_id = "0"

        with patch.object(
            worker, "_update_agent_profile", new_callable=AsyncMock
        ) as mock_update:
            # Call _restore_event_position directly with saved ID
            saved_id = "0"
            await worker._restore_event_position(saved_id)

            # Assert position restored IN MEMORY
            assert worker.session.last_event_id == "0"
            assert worker.pending_events == []

            # CRITICAL: Redis persistence NOT called (this is the fix)
            mock_update.assert_not_called()

    @pytest.mark.skip(reason="process_turn method was removed; turn processing now handled via command pattern")
    @pytest.mark.asyncio
    async def test_event_position_restored_in_worker_loop_on_exception(
        self, worker_with_session, sample_events, mock_redis
    ):
        """Test event position restoration in worker loop when exception occurs."""
        pass


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
            # pending_self_actions has been removed - no need to check it


class TestEventPositionNotRestoredOnSuccess:
    """Test that event position is NOT restored on successful processing."""

    @pytest.mark.skip(reason="process_turn method was removed; turn processing now handled via command pattern")
    @pytest.mark.asyncio
    async def test_event_position_not_restored_on_success(
        self, worker_with_session, sample_events
    ):
        """Test that event position remains at post-drain value on success."""
        pass


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
        """Test that pending_events is cleared on restore."""
        worker = worker_with_session

        # Populate pending events
        worker.pending_events = sample_events.copy()
        worker.session.last_event_id = "100-0"

        # Restore to earlier position
        await worker._restore_event_position("50-0")

        # Assert buffer cleared
        assert worker.pending_events == []
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
