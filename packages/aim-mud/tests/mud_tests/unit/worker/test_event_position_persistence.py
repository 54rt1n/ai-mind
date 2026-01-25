# packages/aim-mud/tests/mud_tests/unit/worker/test_event_position_persistence.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for event position tracking in conversation entries.

NOTE: The event position restoration mechanism via _restore_event_position()
has been removed. Events are now consumed when they're pushed to conversation
history, with each conversation entry tracking its own last_event_id. The
get_last_event_id() method scans all entries and returns the MAX event_id.

This file tests the remaining event position persistence behavior:
- Drain does not persist immediately to Redis
- Conversation entries track last_event_id per entry
- Recovery from conversation entries on startup
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
    """Verify drain_events() does NOT persist to Redis."""

    @pytest.mark.asyncio
    async def test_drain_events_advances_in_memory_only(self, initialized_worker, mock_redis, sample_events_data):
        """Test that drain_events() advances last_event_id in memory without Redis persistence."""
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

        # Redis persistence was NOT called during drain
        initialized_worker._update_agent_profile.assert_not_called()

    @pytest.mark.asyncio
    async def test_drain_with_settle_does_not_persist(self, initialized_worker, mock_redis, sample_events_data, monkeypatch):
        """Test that _drain_with_settle() also does not persist to Redis."""
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

        # Redis persistence was NOT called
        initialized_worker._update_agent_profile.assert_not_called()


# =============================================================================
# Test 2: Recovery from conversation (source of truth)
# =============================================================================


class TestRecoveryFromConversation:
    """Verify event position recovery from conversation entries on startup."""

    @pytest.mark.asyncio
    async def test_load_agent_profile_recovers_from_conversation(self, initialized_worker, mock_redis):
        """Test that _load_agent_profile recovers last_event_id from conversation if available.

        The conversation is the source of truth because it persists the position
        atomically with the turn data.
        """
        from andimud_worker.conversation import MUDConversationManager

        # Arrange: Agent profile has stale position
        mock_redis.hgetall = AsyncMock(return_value={
            b"last_event_id": b"1-0",  # Stale position
            b"conversation_id": b"test_conv_123",
            b"persona_id": b"andi",
        })

        # Set up conversation manager mock
        mock_conv_manager = MagicMock(spec=MUDConversationManager)
        mock_conv_manager.get_last_event_id = AsyncMock(return_value="3-0")  # Newer position
        mock_conv_manager.set_conversation_id = MagicMock()
        initialized_worker.conversation_manager = mock_conv_manager

        # Act: Load agent profile
        await initialized_worker._load_agent_profile()

        # Assert: Position was recovered from conversation (source of truth)
        assert initialized_worker.session.last_event_id == "3-0"

        # Verify conversation was queried
        mock_conv_manager.get_last_event_id.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_agent_profile_uses_profile_when_conversation_empty(self, initialized_worker, mock_redis):
        """Test that _load_agent_profile uses profile position when conversation has no last_event_id."""
        from andimud_worker.conversation import MUDConversationManager

        # Arrange: Agent profile has a position
        mock_redis.hgetall = AsyncMock(return_value={
            b"last_event_id": b"2-0",
            b"persona_id": b"andi",
        })

        # Conversation manager returns None (no entries or no last_event_id)
        mock_conv_manager = MagicMock(spec=MUDConversationManager)
        mock_conv_manager.get_last_event_id = AsyncMock(return_value=None)
        mock_conv_manager.set_conversation_id = MagicMock()
        initialized_worker.conversation_manager = mock_conv_manager

        # Act: Load agent profile
        await initialized_worker._load_agent_profile()

        # Assert: Position from profile is used (conversation had nothing)
        assert initialized_worker.session.last_event_id == "2-0"

    @pytest.mark.asyncio
    async def test_load_agent_profile_handles_no_conversation_manager(self, initialized_worker, mock_redis):
        """Test that _load_agent_profile works when conversation_manager is not set."""
        # Arrange: Agent profile has a position, no conversation manager
        mock_redis.hgetall = AsyncMock(return_value={
            b"last_event_id": b"2-0",
            b"persona_id": b"andi",
        })
        initialized_worker.conversation_manager = None

        # Act: Load agent profile (should not crash)
        await initialized_worker._load_agent_profile()

        # Assert: Position from profile is used
        assert initialized_worker.session.last_event_id == "2-0"
