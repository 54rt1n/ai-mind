# packages/aim-mud/tests/mud_tests/unit/worker/test_non_reactive_events.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for NON_REACTIVE event handling in EventsCommand.

Verifies:
1. NON_REACTIVE events are pushed to conversation (not lost on early return)
2. TERMINAL/CODE_FILE/CODE_ACTION events are treated as non-reactive
3. Mixed reactive + non-reactive events trigger normal turn processing
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aim_mud_types import MUDEvent, MUDTurnRequest, EventType, TurnRequestStatus
from aim_mud_types.models.session import MUDSession
from andimud_worker.commands.events import EventsCommand
from andimud_worker.commands.result import CommandResult


@pytest.fixture
def non_reactive_event():
    """Create a NON_REACTIVE event."""
    return MUDEvent(
        event_id="event-nr-1",
        event_type=EventType.NON_REACTIVE,
        actor="Andi",
        actor_id="andi",
        room_id="room1",
        room_name="Test Room",
        content="fidgets with her silver band",
        timestamp=datetime.now(timezone.utc),
        metadata={"sequence_id": 100},
    )


@pytest.fixture
def terminal_event():
    """Create a TERMINAL event."""
    return MUDEvent(
        event_id="event-term-1",
        event_type=EventType.TERMINAL,
        actor="Andi",
        actor_id="andi",
        room_id="room1",
        room_name="Test Room",
        content="bash_exec: ls output",
        timestamp=datetime.now(timezone.utc),
        metadata={"sequence_id": 101, "tool_name": "bash_exec"},
    )


@pytest.fixture
def code_file_event():
    """Create a CODE_FILE event."""
    return MUDEvent(
        event_id="event-cf-1",
        event_type=EventType.CODE_FILE,
        actor="Andi",
        actor_id="andi",
        room_id="room1",
        room_name="Test Room",
        content="cat output: file contents here",
        timestamp=datetime.now(timezone.utc),
        metadata={"sequence_id": 102, "command": "cat"},
    )


@pytest.fixture
def code_action_event():
    """Create a CODE_ACTION event."""
    return MUDEvent(
        event_id="event-ca-1",
        event_type=EventType.CODE_ACTION,
        actor="Andi",
        actor_id="andi",
        room_id="room1",
        room_name="Test Room",
        content="sed output: file modified",
        timestamp=datetime.now(timezone.utc),
        metadata={"sequence_id": 103, "command": "sed"},
    )


@pytest.fixture
def speech_event():
    """Create a SPEECH event (reactive)."""
    return MUDEvent(
        event_id="event-speech-1",
        event_type=EventType.SPEECH,
        actor="Nova",
        actor_id="nova",
        room_id="room1",
        room_name="Test Room",
        content="Hello Andi!",
        timestamp=datetime.now(timezone.utc),
        metadata={"sequence_id": 104},
    )


@pytest.fixture
def turn_request():
    """Create a test turn request."""
    return MUDTurnRequest(
        agent_id="test_agent",
        turn_id="turn-123",
        reason="events",
        status=TurnRequestStatus.ASSIGNED,
        sequence_id=100,
    )


class TestNonReactiveEventConversationPush:
    """Tests that NON_REACTIVE events are pushed to conversation."""

    @pytest.mark.asyncio
    async def test_non_reactive_only_pushed_to_conversation(
        self, test_worker, non_reactive_event, turn_request
    ):
        """NON_REACTIVE-only events should be pushed to conversation before early return."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
        )
        test_worker._check_agent_is_sleeping = AsyncMock(return_value=False)
        test_worker._push_events_to_conversation = AsyncMock()

        events = [non_reactive_event]

        # Act
        cmd = EventsCommand()
        result = await cmd.execute(
            test_worker,
            events=events,
            **turn_request.model_dump(),
        )

        # Assert
        assert result.complete is True
        assert "Non-reactive" in result.message
        # CRITICAL: Events should be pushed to conversation
        test_worker._push_events_to_conversation.assert_called_once_with(events)

    @pytest.mark.asyncio
    async def test_terminal_event_pushed_to_conversation(
        self, test_worker, terminal_event, turn_request
    ):
        """TERMINAL events should be pushed to conversation without triggering turn."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
        )
        test_worker._check_agent_is_sleeping = AsyncMock(return_value=False)
        test_worker._push_events_to_conversation = AsyncMock()

        events = [terminal_event]

        # Act
        cmd = EventsCommand()
        result = await cmd.execute(
            test_worker,
            events=events,
            **turn_request.model_dump(),
        )

        # Assert
        assert result.complete is True
        assert "Non-reactive" in result.message
        test_worker._push_events_to_conversation.assert_called_once_with(events)


class TestTerminalCodeEventsNonReactive:
    """Tests that TERMINAL/CODE_FILE/CODE_ACTION are treated as non-reactive."""

    @pytest.mark.asyncio
    async def test_terminal_event_no_turn_trigger(
        self, test_worker, terminal_event, turn_request
    ):
        """TERMINAL events should not trigger turn processing."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
        )
        test_worker._check_agent_is_sleeping = AsyncMock(return_value=False)
        test_worker._push_events_to_conversation = AsyncMock()
        test_worker.take_turn = AsyncMock()

        events = [terminal_event]

        # Act
        cmd = EventsCommand()
        result = await cmd.execute(
            test_worker,
            events=events,
            **turn_request.model_dump(),
        )

        # Assert
        assert result.complete is True
        # take_turn should NOT be called for non-reactive events
        test_worker.take_turn.assert_not_called()

    @pytest.mark.asyncio
    async def test_code_file_event_no_turn_trigger(
        self, test_worker, code_file_event, turn_request
    ):
        """CODE_FILE events should not trigger turn processing."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
        )
        test_worker._check_agent_is_sleeping = AsyncMock(return_value=False)
        test_worker._push_events_to_conversation = AsyncMock()
        test_worker.take_turn = AsyncMock()

        events = [code_file_event]

        # Act
        cmd = EventsCommand()
        result = await cmd.execute(
            test_worker,
            events=events,
            **turn_request.model_dump(),
        )

        # Assert
        assert result.complete is True
        test_worker.take_turn.assert_not_called()

    @pytest.mark.asyncio
    async def test_code_action_event_no_turn_trigger(
        self, test_worker, code_action_event, turn_request
    ):
        """CODE_ACTION events should not trigger turn processing."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
        )
        test_worker._check_agent_is_sleeping = AsyncMock(return_value=False)
        test_worker._push_events_to_conversation = AsyncMock()
        test_worker.take_turn = AsyncMock()

        events = [code_action_event]

        # Act
        cmd = EventsCommand()
        result = await cmd.execute(
            test_worker,
            events=events,
            **turn_request.model_dump(),
        )

        # Assert
        assert result.complete is True
        test_worker.take_turn.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_non_reactive_types_all_pushed(
        self, test_worker, non_reactive_event, terminal_event, code_file_event, turn_request
    ):
        """Multiple non-reactive event types should all be pushed to conversation."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
        )
        test_worker._check_agent_is_sleeping = AsyncMock(return_value=False)
        test_worker._push_events_to_conversation = AsyncMock()

        events = [non_reactive_event, terminal_event, code_file_event]

        # Act
        cmd = EventsCommand()
        result = await cmd.execute(
            test_worker,
            events=events,
            **turn_request.model_dump(),
        )

        # Assert
        assert result.complete is True
        test_worker._push_events_to_conversation.assert_called_once_with(events)


class TestMixedEventsTriggerTurn:
    """Tests that mixed reactive + non-reactive events trigger normal processing."""

    @pytest.mark.asyncio
    async def test_mixed_events_trigger_turn(
        self, test_worker, non_reactive_event, speech_event, turn_request
    ):
        """Mixed reactive + non-reactive events should trigger turn processing."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
        )
        test_worker._check_agent_is_sleeping = AsyncMock(return_value=False)
        test_worker._push_events_to_conversation = AsyncMock()

        # Mock take_turn to return a decision
        from aim_mud_types.models.decision import DecisionResult, DecisionType
        mock_decision = DecisionResult(
            decision_type=DecisionType.WAIT,
            args={},
            thinking="",
            raw_response="<wait/>",
            cleaned_response="<wait/>",
        )
        test_worker.take_turn = AsyncMock(return_value=mock_decision)
        test_worker._last_emitted_action_ids = []

        events = [non_reactive_event, speech_event]

        # Act
        cmd = EventsCommand()
        result = await cmd.execute(
            test_worker,
            events=events,
            **turn_request.model_dump(),
        )

        # Assert
        assert result.complete is True
        # take_turn SHOULD be called because there's a reactive event
        test_worker.take_turn.assert_called_once()

    @pytest.mark.asyncio
    async def test_terminal_plus_speech_triggers_turn(
        self, test_worker, terminal_event, speech_event, turn_request
    ):
        """TERMINAL + SPEECH events should trigger turn (SPEECH is reactive)."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
        )
        test_worker._check_agent_is_sleeping = AsyncMock(return_value=False)
        test_worker._push_events_to_conversation = AsyncMock()

        from aim_mud_types.models.decision import DecisionResult, DecisionType
        mock_decision = DecisionResult(
            decision_type=DecisionType.SPEAK,
            args={"text": "Hello!"},
            thinking="",
            raw_response="<speak>Hello!</speak>",
            cleaned_response="<speak>Hello!</speak>",
        )
        test_worker.take_turn = AsyncMock(return_value=mock_decision)
        test_worker._last_emitted_action_ids = ["act_123_abc"]

        events = [terminal_event, speech_event]

        # Act
        cmd = EventsCommand()
        result = await cmd.execute(
            test_worker,
            events=events,
            **turn_request.model_dump(),
        )

        # Assert
        assert result.complete is True
        # take_turn SHOULD be called
        test_worker.take_turn.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
