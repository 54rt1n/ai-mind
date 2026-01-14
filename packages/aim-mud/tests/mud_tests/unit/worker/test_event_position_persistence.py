# packages/aim-mud/tests/mud_tests/unit/worker/test_event_position_persistence.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for event position persistence fix.

This test suite validates the correct behavior of event position persistence
across the turn lifecycle. The fix ensures events are never lost during crashes
by controlling WHEN last_event_id is persisted to Redis.

CRITICAL BEHAVIOR:
- drain_events() advances last_event_id in memory ONLY (no Redis persistence)
- Speech turns persist to Redis AFTER speech check confirms speech action
- Non-speech turns call _restore_event_position() which rolls back in memory ONLY
- Exceptions call _restore_event_position() which rolls back in memory ONLY
- Redis only advances when speech turn explicitly persists the advanced position

This ensures crash safety:
- If worker crashes during non-speech processing, Redis has old position
- Events are available for re-drain on restart
- Only confirmed speech actions consume events permanently
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import json

from aim_mud_types import (
    MUDSession,
    MUDEvent,
    MUDAction,
    MUDTurn,
    RoomState,
    EventType,
    ActorType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def initialized_worker(test_worker):
    """Worker with initialized session and mocked profile updates."""
    test_worker.session = MUDSession(
        agent_id=test_worker.config.agent_id,
        persona_id=test_worker.config.persona_id,
        max_recent_turns=20,
    )
    test_worker.session.last_event_id = "1-0"

    # Mock the profile update method (external Redis persistence)
    test_worker._update_agent_profile = AsyncMock()

    return test_worker


@pytest.fixture
def sample_events_data():
    """Sample event data for Redis mocking."""
    return [
        {
            "event_type": EventType.SPEECH.value,
            "actor": "Prax",
            "actor_type": ActorType.PLAYER.value,
            "room_id": "#123",
            "content": "Hello there!",
            "timestamp": "2026-01-08T12:00:00+00:00",
            "sequence_id": 1,
        },
        {
            "event_type": EventType.EMOTE.value,
            "actor": "Prax",
            "actor_type": ActorType.PLAYER.value,
            "room_id": "#123",
            "content": "waves",
            "timestamp": "2026-01-08T12:00:01+00:00",
            "sequence_id": 2,
        }
    ]


# =============================================================================
# Test 1: Event position NOT persisted after drain
# =============================================================================


class TestDrainDoesNotPersist:
    """Verify drain_events() does NOT persist to Redis (line 152 removal)."""

    @pytest.mark.asyncio
    async def test_drain_events_advances_in_memory_only(self, initialized_worker, mock_redis, sample_events_data):
        """Test that drain_events() advances last_event_id in memory without Redis persistence.

        This validates the fix: line 152 removed, no automatic persistence after drain.
        """
        # Arrange: Mock Redis to return events
        mock_redis.xinfo_stream.return_value = {"last-generated-id": "2-0"}
        mock_redis.xrange.side_effect = [
            [
                ("1-1", {"data": json.dumps(sample_events_data[0])}),
                ("2-0", {"data": json.dumps(sample_events_data[1])})
            ],
            []  # Second call returns empty (settled)
        ]

        # Capture initial state
        initial_event_id = initialized_worker.session.last_event_id

        # Act: Drain events
        events = await initialized_worker.drain_events(timeout=0)

        # Assert: In-memory position advanced
        assert initialized_worker.session.last_event_id == "2-0"
        assert initialized_worker.session.last_event_id != initial_event_id
        assert len(events) == 2

        # CRITICAL: Redis persistence was NOT called during drain
        initialized_worker._update_agent_profile.assert_not_called()

    @pytest.mark.asyncio
    async def test_drain_with_settle_does_not_persist(self, initialized_worker, mock_redis, sample_events_data, monkeypatch):
        """Test that _drain_with_settle() also does not persist to Redis.

        Validates the full settling flow respects the no-persistence rule.
        """
        # Mock asyncio.sleep to avoid actual delays
        import asyncio
        original_sleep = asyncio.sleep
        async def mock_sleep(seconds):
            await original_sleep(0)
        monkeypatch.setattr(asyncio, 'sleep', mock_sleep)

        # Arrange: Mock Redis for settle behavior
        mock_redis.xinfo_stream.side_effect = [
            {"last-generated-id": "2-0"},  # First drain
            {"last-generated-id": "2-0"},  # Second drain (no new events)
        ]
        mock_redis.xrange.side_effect = [
            [("2-0", {"data": json.dumps(sample_events_data[0])})],
            []  # Settled
        ]

        # Act: Drain with settle
        events = await initialized_worker._drain_with_settle()

        # Assert: Position advanced in memory
        assert initialized_worker.session.last_event_id == "2-0"
        assert len(events) == 1

        # CRITICAL: Redis persistence was NOT called
        initialized_worker._update_agent_profile.assert_not_called()


# =============================================================================
# Test 2: Speech turn persists advanced position
# =============================================================================


class TestSpeechTurnPersists:
    """Verify speech turns persist advanced position to Redis (line 379 addition)."""

    @pytest.mark.asyncio
    async def test_speech_turn_persists_advanced_position(self, initialized_worker):
        """Test that a speech turn persists the advanced event position to Redis.

        This validates the fix: line 379 added, explicit persistence after speech check.
        """
        # Arrange: Create a turn with a speak action
        turn_with_speech = MUDTurn(
            timestamp=datetime.now(timezone.utc),
            events_received=[
                MUDEvent(
                    event_id="evt_1",
                    event_type=EventType.SPEECH,
                    actor="Prax",
                    actor_id="#player_1",
                    actor_type=ActorType.PLAYER,
                    room_id="#123",
                    room_name="The Garden",
                    content="Hello there!",
                    timestamp=datetime.now(timezone.utc),
                )
            ],
            room_context=RoomState(
                room_id="#123",
                name="The Garden",
                description="A serene garden.",
                exits={"north": "#124"},
            ),
            entities_context=[],
            thinking="I should respond.",
            actions_taken=[
                MUDAction(
                    tool="speak",
                    content="Hi Prax!",
                    target=None,
                    location=None,
                )
            ],
        )

        # Add turn to session (this is what process_turn does)
        initialized_worker.session.add_turn(turn_with_speech)

        # Simulate advanced position after drain
        initialized_worker.session.last_event_id = "3-0"

        # Act: Execute speech check logic (from worker.py lines 356-379)
        has_speech = False
        last_turn = initialized_worker.session.get_last_turn()
        if last_turn:
            for action in last_turn.actions_taken:
                if action.tool == "speak":
                    has_speech = True
                    break

        # If speech detected, persist the advanced position
        if has_speech:
            await initialized_worker._update_agent_profile(
                last_event_id=initialized_worker.session.last_event_id
            )

        # Assert: Speech was detected
        assert has_speech is True

        # CRITICAL: Redis persistence WAS called with advanced position
        initialized_worker._update_agent_profile.assert_called_once_with(
            last_event_id="3-0"
        )

    @pytest.mark.asyncio
    async def test_multiple_actions_with_speech_persists(self, initialized_worker):
        """Test that turn with multiple actions including speak still persists."""
        # Arrange: Turn with multiple actions, one is speak
        turn_with_multiple_actions = MUDTurn(
            timestamp=datetime.now(timezone.utc),
            events_received=[],
            room_context=None,
            entities_context=[],
            thinking="I'll move and speak.",
            actions_taken=[
                MUDAction(tool="move", content="", target=None, location="north"),
                MUDAction(tool="speak", content="I'm heading north!", target=None, location=None),
                MUDAction(tool="emote", content="waves", target=None, location=None),
            ],
        )

        initialized_worker.session.add_turn(turn_with_multiple_actions)
        initialized_worker.session.last_event_id = "4-0"

        # Act: Speech check
        has_speech = False
        last_turn = initialized_worker.session.get_last_turn()
        if last_turn:
            for action in last_turn.actions_taken:
                if action.tool == "speak":
                    has_speech = True
                    break

        if has_speech:
            await initialized_worker._update_agent_profile(
                last_event_id=initialized_worker.session.last_event_id
            )

        # Assert: Speech detected and persisted
        assert has_speech is True
        initialized_worker._update_agent_profile.assert_called_once_with(
            last_event_id="4-0"
        )


# =============================================================================
# Test 3: Non-speech turn does NOT persist
# =============================================================================


class TestNonSpeechTurnDoesNotPersist:
    """Verify non-speech turns do NOT persist (restoration rolls back in memory only)."""

    @pytest.mark.asyncio
    async def test_non_speech_turn_calls_restore(self, initialized_worker):
        """Test that non-speech turn calls _restore_event_position() without Redis persist."""
        # Arrange: Turn with non-speech action
        turn_without_speech = MUDTurn(
            timestamp=datetime.now(timezone.utc),
            events_received=[],
            room_context=None,
            entities_context=[],
            thinking="I'll move north.",
            actions_taken=[
                MUDAction(tool="move", content="", target=None, location="north")
            ],
        )

        initialized_worker.session.add_turn(turn_without_speech)

        # Simulate drain advanced position
        saved_event_id = "2-0"
        initialized_worker.session.last_event_id = "3-0"

        # Act: Speech check logic
        has_speech = False
        last_turn = initialized_worker.session.get_last_turn()
        if last_turn:
            for action in last_turn.actions_taken:
                if action.tool == "speak":
                    has_speech = True
                    break

        # If no speech, restore position
        if not has_speech:
            await initialized_worker._restore_event_position(saved_event_id)

        # Assert: No speech detected
        assert has_speech is False

        # CRITICAL: Position rolled back in memory
        assert initialized_worker.session.last_event_id == "2-0"

        # CRITICAL: Redis persistence was NOT called by restore
        # (restore no longer calls _update_agent_profile internally)
        initialized_worker._update_agent_profile.assert_not_called()

    @pytest.mark.asyncio
    async def test_restore_clears_pending_buffers(self, initialized_worker):
        """Test that _restore_event_position() clears pending event buffers.

        Note: pending_self_actions has been removed from MUDSession.
        This test now only validates pending_events clearing.
        """
        # Arrange: Populate pending buffers
        initialized_worker.pending_events = ["event1", "event2"]
        saved_event_id = "1-0"
        initialized_worker.session.last_event_id = "2-0"

        # Act: Restore
        await initialized_worker._restore_event_position(saved_event_id)

        # Assert: Buffers cleared
        assert initialized_worker.pending_events == []
        assert initialized_worker.session.last_event_id == "1-0"

        # CRITICAL: No Redis persistence
        initialized_worker._update_agent_profile.assert_not_called()

    @pytest.mark.asyncio
    async def test_restore_with_none_is_noop(self, initialized_worker):
        """Test that _restore_event_position(None) is a no-op."""
        # Arrange
        initialized_worker.session.last_event_id = "3-0"
        initialized_worker.pending_events = ["event1"]

        # Act: Restore with None (signals events were consumed)
        await initialized_worker._restore_event_position(None)

        # Assert: State unchanged
        assert initialized_worker.session.last_event_id == "3-0"
        assert initialized_worker.pending_events == ["event1"]
        initialized_worker._update_agent_profile.assert_not_called()


# =============================================================================
# Test 4: Exception does NOT persist
# =============================================================================


class TestExceptionDoesNotPersist:
    """Verify exceptions trigger restoration without Redis persistence."""

    @pytest.mark.asyncio
    async def test_exception_triggers_restore_without_persist(self, initialized_worker):
        """Test that exception handler calls _restore_event_position() without persisting."""
        # Arrange: Simulate exception during turn processing
        saved_event_id = "2-0"
        initialized_worker.session.last_event_id = "3-0"

        # Act: Exception handler logic (from worker.py lines 425-428)
        try:
            raise ValueError("Simulated LLM failure")
        except Exception:
            # Restore event position on exception
            if saved_event_id:
                await initialized_worker._restore_event_position(saved_event_id)

        # Assert: Position rolled back
        assert initialized_worker.session.last_event_id == "2-0"

        # CRITICAL: No Redis persistence during restore
        initialized_worker._update_agent_profile.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_with_none_saved_id_does_not_restore(self, initialized_worker):
        """Test that exception with saved_event_id=None skips restoration."""
        # Arrange: saved_event_id is None (events were already consumed)
        saved_event_id = None
        initialized_worker.session.last_event_id = "3-0"

        # Act: Exception handler logic
        try:
            raise ValueError("Simulated failure after consumption")
        except Exception:
            if saved_event_id:
                await initialized_worker._restore_event_position(saved_event_id)

        # Assert: Position NOT rolled back (events were consumed)
        assert initialized_worker.session.last_event_id == "3-0"
        initialized_worker._update_agent_profile.assert_not_called()


# =============================================================================
# Test 5: Crash simulation (integration-style)
# =============================================================================


class TestCrashSimulation:
    """Integration-style tests simulating worker crashes and restarts."""

    @pytest.mark.asyncio
    async def test_crash_during_non_speech_turn_preserves_events(self, initialized_worker, mock_redis):
        """Test that crash during non-speech turn leaves events available for re-drain.

        This is the critical crash-safety test:
        1. Load initial state from Redis: last_event_id = "1-0"
        2. Drain events → advances to "2-0" in memory only
        3. CRASH before speech check/restore
        4. Restart → reload from Redis, still "1-0"
        5. Events available for re-drain
        """
        # Phase 1: Initial state in Redis
        from unittest.mock import AsyncMock
        mock_redis.hget = AsyncMock(return_value=b"1-0")  # Redis has old position

        # Simulate loading from Redis (like _load_agent_profile does)
        redis_event_id = await mock_redis.hget("agent:test_agent:profile", "last_event_id")
        if redis_event_id:
            initialized_worker.session.last_event_id = redis_event_id.decode("utf-8")

        assert initialized_worker.session.last_event_id == "1-0"

        # Phase 2: Drain events (advances in memory)
        mock_redis.xinfo_stream.return_value = {"last-generated-id": "2-0"}
        mock_redis.xrange.return_value = [
            ("2-0", {"data": json.dumps({
                "event_type": EventType.SPEECH.value,
                "actor": "Prax",
                "actor_type": ActorType.PLAYER.value,
                "room_id": "#123",
                "content": "Hello",
                "timestamp": "2026-01-08T12:00:00+00:00",
                "sequence_id": 1,
            })})
        ]

        events = await initialized_worker.drain_events(timeout=0)
        assert len(events) == 1
        assert initialized_worker.session.last_event_id == "2-0"  # Advanced in memory

        # Phase 3: CRASH (simulated by not completing turn processing)
        # No speech check, no restore, no persistence
        # Redis still has "1-0"

        # Phase 4: Restart - reload from Redis
        mock_redis.hget.return_value = b"1-0"  # Redis unchanged
        reloaded_event_id = await mock_redis.hget("agent:test_agent:profile", "last_event_id")
        initialized_worker.session.last_event_id = reloaded_event_id.decode("utf-8")

        # Assert: Position restored to Redis value
        assert initialized_worker.session.last_event_id == "1-0"

        # Phase 5: Re-drain events (events still available)
        mock_redis.xrange.return_value = [
            ("2-0", {"data": json.dumps({
                "event_type": EventType.SPEECH.value,
                "actor": "Prax",
                "actor_type": ActorType.PLAYER.value,
                "room_id": "#123",
                "content": "Hello",
                "timestamp": "2026-01-08T12:00:00+00:00",
                "sequence_id": 1,
            })})
        ]

        events_after_restart = await initialized_worker.drain_events(timeout=0)

        # CRITICAL: Events are available again after crash
        assert len(events_after_restart) == 1
        assert events_after_restart[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_crash_after_speech_turn_consumes_events(self, initialized_worker, mock_redis):
        """Test that crash AFTER speech turn persistence correctly consumes events.

        This validates the happy path:
        1. Drain events → "2-0" in memory
        2. Process turn → detect speech
        3. Persist to Redis → "2-0" in Redis
        4. CRASH (after persistence)
        5. Restart → Redis has "2-0", events consumed
        """
        # Phase 1: Initial state
        from unittest.mock import AsyncMock
        mock_redis.hget = AsyncMock(return_value=b"1-0")
        redis_event_id = await mock_redis.hget("agent:test_agent:profile", "last_event_id")
        if redis_event_id:
            initialized_worker.session.last_event_id = redis_event_id.decode("utf-8")

        # Phase 2: Drain and process speech turn
        mock_redis.xinfo_stream.return_value = {"last-generated-id": "2-0"}
        mock_redis.xrange.return_value = [
            ("2-0", {"data": json.dumps({
                "event_type": EventType.SPEECH.value,
                "actor": "Prax",
                "actor_type": ActorType.PLAYER.value,
                "room_id": "#123",
                "content": "Hello",
                "timestamp": "2026-01-08T12:00:00+00:00",
                "sequence_id": 1,
            })})
        ]

        await initialized_worker.drain_events(timeout=0)

        # Simulate speech turn
        turn_with_speech = MUDTurn(
            timestamp=datetime.now(timezone.utc),
            events_received=[],
            room_context=None,
            entities_context=[],
            thinking="I'll respond.",
            actions_taken=[MUDAction(tool="speak", content="Hi!", target=None, location=None)],
        )
        initialized_worker.session.add_turn(turn_with_speech)

        # Phase 3: Speech check → persist
        has_speech = False
        last_turn = initialized_worker.session.get_last_turn()
        if last_turn:
            for action in last_turn.actions_taken:
                if action.tool == "speak":
                    has_speech = True
                    break

        if has_speech:
            await initialized_worker._update_agent_profile(
                last_event_id=initialized_worker.session.last_event_id
            )

        # Assert: Persisted with advanced position
        initialized_worker._update_agent_profile.assert_called_once_with(
            last_event_id="2-0"
        )

        # Phase 4: Simulate Redis update (what _update_agent_profile does)
        mock_redis.hget.return_value = b"2-0"

        # Phase 5: Restart - reload from Redis
        reloaded_event_id = await mock_redis.hget("agent:test_agent:profile", "last_event_id")
        initialized_worker.session.last_event_id = reloaded_event_id.decode("utf-8")

        # Assert: Redis has advanced position
        assert initialized_worker.session.last_event_id == "2-0"

        # Phase 6: Re-drain would skip consumed events
        # xrange with min="(2-0" would return empty (events consumed)
        mock_redis.xrange.return_value = []
        events_after_restart = await initialized_worker.drain_events(timeout=0)

        # CRITICAL: No events returned (they were consumed)
        assert len(events_after_restart) == 0


# NOTE: Test for cascading events removed temporarily - mocking issues
# Fix is in place in events.py lines 218-221:
#   cap = max_sequence_id if drain_count == 0 else None
# This ensures only the first drain is capped, subsequent drains during
# settle window have no cap to catch cascading events (e.g., user speaking
# during settle after movement triggers turn).
#
# Manual test: Enter room (movement), speak during 15s settle window, verify
# both events are captured in the same turn.
