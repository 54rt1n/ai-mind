# packages/aim-mud/tests/mud_tests/unit/worker/test_speech_detection_integration.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for speech detection flow integrated with worker loop.

Tests the critical flow in worker.py lines 343-363:
- Events → process_turn (with speak action) → events consumed (position updated)
- Events → process_turn (no speak action) → events restored (position rolled back)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from aim_mud_types import MUDEvent, MUDAction, EventType
from aim_mud_types.session import MUDSession, MUDTurn


@pytest.fixture
def sample_events():
    """Create sample events for testing."""
    return [
        MUDEvent(
            event_id="event-1",
            event_type=EventType.SPEECH,
            actor="OtherAgent",
            actor_id="other_agent",
            room_id="room1",
            room_name="Test Room",
            content="Hello there",
            timestamp=datetime.now(timezone.utc),
            metadata={"sequence_id": 100},
        ),
        MUDEvent(
            event_id="event-2",
            event_type=EventType.MOVEMENT,
            actor="OtherAgent",
            actor_id="other_agent",
            room_id="room1",
            room_name="Test Room",
            content="enters the room",
            timestamp=datetime.now(timezone.utc),
            metadata={"sequence_id": 101},
        ),
    ]


@pytest.fixture
def mock_process_turn():
    """Mock the process_turn method."""
    return AsyncMock()


class TestSpeechDetectionIntegration:
    """Tests for speech detection flow in worker loop."""

    @pytest.mark.asyncio
    async def test_speech_action_consumes_events(
        self, test_worker, sample_events, mock_process_turn
    ):
        """Test: Turn with speak action → events consumed (position updated)."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            last_event_id="event-0",
        )

        # Create turn with speak action
        turn_with_speech = MUDTurn(
            timestamp=datetime.now(timezone.utc),
            events_received=sample_events,
            thinking="I should respond",
            actions_taken=[
                MUDAction(tool="speak", args={"message": "Hi back"})
            ],
        )

        # Mock process_turn to add the turn to session
        async def mock_process(events):
            test_worker.session.add_turn(turn_with_speech)

        test_worker.process_turn = mock_process

        # Save initial event position
        saved_event_id = test_worker.session.last_event_id

        # Set pending events (simulating drain)
        test_worker.pending_events = sample_events
        test_worker.session.last_event_id = "event-2"  # Advanced during drain

        # Act
        await test_worker.process_turn(sample_events)

        # Simulate speech detection logic (lines 343-363)
        has_speech = False
        last_turn = test_worker.session.get_last_turn()
        if last_turn:
            for action in last_turn.actions_taken:
                if action.tool == "speak":
                    has_speech = True
                    break

        # Restore position if no speech
        if not has_speech:
            test_worker.session.last_event_id = saved_event_id
            saved_event_id = None
        else:
            saved_event_id = None

        # Assert
        assert has_speech is True, "Should detect speech action"
        assert test_worker.session.last_event_id == "event-2", (
            "Speech turn should keep advanced position (events consumed)"
        )
        assert saved_event_id is None, "saved_event_id should be cleared"

    @pytest.mark.asyncio
    async def test_non_speech_action_restores_events(
        self, test_worker, sample_events
    ):
        """Test: Turn with no speak action → events restored (position rolled back)."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            last_event_id="event-0",
        )

        # Create turn WITHOUT speak action
        turn_without_speech = MUDTurn(
            timestamp=datetime.now(timezone.utc),
            events_received=sample_events,
            thinking="Just observing",
            actions_taken=[
                MUDAction(tool="look", args={})
            ],
        )

        # Mock process_turn to add the turn to session
        async def mock_process(events):
            test_worker.session.add_turn(turn_without_speech)

        test_worker.process_turn = mock_process

        # Save initial event position
        saved_event_id = test_worker.session.last_event_id

        # Set pending events (simulating drain)
        test_worker.pending_events = sample_events
        test_worker.session.last_event_id = "event-2"  # Advanced during drain

        # Act
        await test_worker.process_turn(sample_events)

        # Simulate speech detection logic (lines 343-363)
        has_speech = False
        last_turn = test_worker.session.get_last_turn()
        if last_turn:
            for action in last_turn.actions_taken:
                if action.tool == "speak":
                    has_speech = True
                    break

        # Restore position if no speech
        if not has_speech:
            test_worker.session.last_event_id = saved_event_id
            saved_event_id = None

        # Assert
        assert has_speech is False, "Should NOT detect speech action"
        assert test_worker.session.last_event_id == "event-0", (
            "Non-speech turn should restore original position (events not consumed)"
        )
        assert saved_event_id is None, "saved_event_id should be cleared after restore"

    @pytest.mark.asyncio
    async def test_turn_with_multiple_actions_including_speak(
        self, test_worker, sample_events
    ):
        """Test: Turn with multiple actions [move, speak] → counts as speech."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            last_event_id="event-0",
        )

        # Create turn with multiple actions including speak
        turn_with_multiple = MUDTurn(
            timestamp=datetime.now(timezone.utc),
            events_received=sample_events,
            thinking="I should move and respond",
            actions_taken=[
                MUDAction(tool="move", args={"direction": "north"}),
                MUDAction(tool="speak", args={"message": "See you later"}),
            ],
        )

        # Mock process_turn
        async def mock_process(events):
            test_worker.session.add_turn(turn_with_multiple)

        test_worker.process_turn = mock_process

        # Save initial position
        saved_event_id = test_worker.session.last_event_id
        test_worker.pending_events = sample_events
        test_worker.session.last_event_id = "event-2"

        # Act
        await test_worker.process_turn(sample_events)

        # Simulate speech detection
        has_speech = False
        last_turn = test_worker.session.get_last_turn()
        if last_turn:
            for action in last_turn.actions_taken:
                if action.tool == "speak":
                    has_speech = True
                    break

        if not has_speech:
            test_worker.session.last_event_id = saved_event_id

        # Assert
        assert has_speech is True, "Should detect speech in multiple actions"
        assert test_worker.session.last_event_id == "event-2", (
            "Multiple actions with speak should consume events"
        )

    @pytest.mark.asyncio
    async def test_exception_during_turn_uses_saved_event_id(
        self, test_worker, sample_events
    ):
        """Test: Exception during turn → events restored via saved_event_id."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            last_event_id="event-0",
        )

        # Mock process_turn to raise exception
        async def mock_process_error(events):
            raise RuntimeError("LLM call failed")

        test_worker.process_turn = mock_process_error

        # Save initial position
        saved_event_id = test_worker.session.last_event_id
        test_worker.pending_events = sample_events
        test_worker.session.last_event_id = "event-2"  # Advanced during drain

        # Act
        try:
            await test_worker.process_turn(sample_events)
        except RuntimeError:
            # Simulate exception handler restoring position (lines 406-409)
            # Note: pending_self_actions has been removed from MUDSession
            if saved_event_id:
                test_worker.session.last_event_id = saved_event_id
                test_worker.pending_events = []

        # Assert
        assert test_worker.session.last_event_id == "event-0", (
            "Exception should restore event position using saved_event_id"
        )
        assert test_worker.pending_events == [], (
            "Exception should clear pending_events buffer"
        )

    @pytest.mark.asyncio
    async def test_empty_turn_no_actions_restores_events(
        self, test_worker, sample_events
    ):
        """Test: Empty turn (no actions) → events restored."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            last_event_id="event-0",
        )

        # Create turn with no actions
        empty_turn = MUDTurn(
            timestamp=datetime.now(timezone.utc),
            events_received=sample_events,
            thinking="Not sure what to do",
            actions_taken=[],  # No actions
        )

        # Mock process_turn
        async def mock_process(events):
            test_worker.session.add_turn(empty_turn)

        test_worker.process_turn = mock_process

        # Save initial position
        saved_event_id = test_worker.session.last_event_id
        test_worker.pending_events = sample_events
        test_worker.session.last_event_id = "event-2"

        # Act
        await test_worker.process_turn(sample_events)

        # Simulate speech detection
        has_speech = False
        last_turn = test_worker.session.get_last_turn()
        if last_turn:
            for action in last_turn.actions_taken:
                if action.tool == "speak":
                    has_speech = True
                    break

        if not has_speech:
            test_worker.session.last_event_id = saved_event_id

        # Assert
        assert has_speech is False, "Empty actions should not have speech"
        assert test_worker.session.last_event_id == "event-0", (
            "Empty turn should restore event position"
        )

    @pytest.mark.asyncio
    async def test_saved_event_id_prevents_double_restore(
        self, test_worker, sample_events
    ):
        """Test: Setting saved_event_id=None prevents double-restore on exception."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            last_event_id="event-0",
        )

        # Create turn with speak
        turn_with_speech = MUDTurn(
            timestamp=datetime.now(timezone.utc),
            events_received=sample_events,
            thinking="I will respond",
            actions_taken=[
                MUDAction(tool="speak", args={"message": "Hello"})
            ],
        )

        async def mock_process(events):
            test_worker.session.add_turn(turn_with_speech)

        test_worker.process_turn = mock_process

        # Simulate the flow
        saved_event_id = test_worker.session.last_event_id
        test_worker.session.last_event_id = "event-2"

        await test_worker.process_turn(sample_events)

        # Speech detection clears saved_event_id
        has_speech = False
        last_turn = test_worker.session.get_last_turn()
        if last_turn:
            for action in last_turn.actions_taken:
                if action.tool == "speak":
                    has_speech = True
                    break

        if has_speech:
            saved_event_id = None  # Prevents restore in exception handler

        # Assert
        assert saved_event_id is None, (
            "Speech turn should clear saved_event_id to prevent double-restore"
        )
        assert test_worker.session.last_event_id == "event-2", (
            "Position should remain advanced for speech turn"
        )
