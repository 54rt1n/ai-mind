# tests/mud_tests/unit/worker/test_speech_event_detection.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for speech event detection after turn processing.

These tests validate the correct approach for checking whether a completed
turn includes a speech action, which is used by the worker to determine
whether events should be consumed or restored.

Correct behavior:
- After process_turn() completes, the turn is added to session.recent_turns
- The worker checks the most recent turn for speech actions
- The code uses session.get_last_turn() to access the last completed turn
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from aim_mud_types import (
    MUDSession,
    MUDEvent,
    MUDAction,
    MUDTurn,
    RoomState,
    EntityState,
    EventType,
    ActorType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_session() -> MUDSession:
    """Create a sample MUDSession with a completed turn that includes speech."""
    # Create a turn with a speak action
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
        thinking="I should respond to the greeting.",
        actions_taken=[
            MUDAction(
                tool="speak",
                content="Hi Prax! How are you?",
                target=None,
                location=None,
            )
        ],
    )

    # Create session and add the turn
    session = MUDSession(
        agent_id="test_agent",
        persona_id="test_persona",
    )
    session.add_turn(turn_with_speech)

    return session


@pytest.fixture
def mock_redis():
    """Mock Redis client for worker tests."""
    redis = AsyncMock()
    redis.hget = AsyncMock(return_value=None)
    redis.hgetall = AsyncMock(return_value={})
    redis.hset = AsyncMock(return_value=1)
    redis.xadd = AsyncMock(return_value=b"1234567890-0")
    redis.xread = AsyncMock(return_value=[])
    redis.xrange = AsyncMock(return_value=[])
    return redis


# =============================================================================
# Test Cases
# =============================================================================


@pytest.mark.asyncio
async def test_speech_event_detection_with_speech_action(sample_session):
    """Test that speech is correctly detected in the last turn.

    Validates that when a turn includes a 'speak' action, the speech
    detection logic correctly identifies it using get_last_turn().
    """
    # Arrange: Session with a completed turn that includes speech
    session = sample_session

    # Act: Use the correct API to check for speech in the last turn
    has_speech = False
    last_turn = session.get_last_turn()
    if last_turn:
        for action in last_turn.actions_taken:
            if action.tool == "speak":
                has_speech = True
                break

    # Assert: This works correctly
    assert has_speech is True


@pytest.mark.asyncio
async def test_speech_event_detection_with_non_speech_turn():
    """Test speech detection when turn has no speech action.

    This validates that the logic correctly identifies non-speech turns.
    """
    # Arrange: Create a turn with a non-speech action (e.g., move)
    turn_without_speech = MUDTurn(
        timestamp=datetime.now(timezone.utc),
        events_received=[],
        room_context=None,
        entities_context=[],
        thinking="I should move north.",
        actions_taken=[
            MUDAction(
                tool="move",
                content="",
                target=None,
                location="north",
            )
        ],
    )

    session = MUDSession(
        agent_id="test_agent",
        persona_id="test_persona",
    )
    session.add_turn(turn_without_speech)

    # Act: Check for speech using correct approach
    has_speech = False
    last_turn = session.get_last_turn()
    if last_turn:
        for action in last_turn.actions_taken:
            if action.tool == "speak":
                has_speech = True
                break

    # Assert: No speech detected
    assert has_speech is False


@pytest.mark.asyncio
async def test_speech_event_detection_with_empty_session():
    """Test speech detection when session has no turns yet.

    This validates proper handling of empty session state.
    """
    # Arrange: Empty session
    session = MUDSession(
        agent_id="test_agent",
        persona_id="test_persona",
    )

    # Act: Check for speech in empty session
    has_speech = False
    last_turn = session.get_last_turn()
    if last_turn:
        for action in last_turn.actions_taken:
            if action.tool == "speak":
                has_speech = True
                break

    # Assert: No turn means no speech
    assert has_speech is False
    assert last_turn is None
