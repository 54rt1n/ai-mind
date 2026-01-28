# tests/mud_tests/unit/mediator/test_echo_clearing.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for mediator echo clearing (MUDLOGIC V2 Phase 3)."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aim_mud_types import EventType, MUDEvent, TurnRequestStatus
from aim_mud_types.models.coordination import MUDTurnRequest


@pytest.fixture
def mock_mediator():
    """Create mock mediator with required attributes."""
    mediator = MagicMock()
    mediator.redis = AsyncMock()
    mediator.config = MagicMock()
    mediator.config.event_stream = "mud:events"
    mediator.config.event_poll_timeout = 1.0
    mediator.running = True
    mediator.registered_agents = {"agent_123"}
    mediator._turn_index = 0
    mediator.last_event_id = "0"

    # Import the mixin methods
    from andimud_mediator.mixins.events import EventsMixin
    mediator._clear_pending_for_echo = EventsMixin._clear_pending_for_echo.__get__(mediator)
    mediator._update_turn_request = EventsMixin._update_turn_request.__get__(mediator)
    mediator._get_turn_request = AsyncMock()

    return mediator


@pytest.mark.asyncio
async def test_echo_clearing_single_action(mock_mediator):
    """Test echo clearing for single pending action."""
    agent_id = "agent_123"
    action_id = "action_abc"

    # Setup turn request with single pending action
    turn_request = MUDTurnRequest(
        turn_id="turn_001",
        status=TurnRequestStatus.PENDING,
        sequence_id=100,
        metadata={"pending_action_ids": [action_id]}
    )
    mock_mediator._get_turn_request.return_value = turn_request

    # Create echo event
    event = MUDEvent(
        event_type=EventType.SPEECH,
        actor="Agent",
        actor_id="char_123",
        action_id=action_id,
        room_id="test_room",
        content="You say, 'Hello'",
    )

    # Mock update_turn_request
    with patch("aim_mud_types.client.RedisMUDClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Call echo clearing
        await mock_mediator._clear_pending_for_echo(event, [agent_id])

        # Verify update was called
        assert mock_client.update_turn_request.called
        call_args = mock_client.update_turn_request.call_args
        updated_request = call_args[0][1]  # Second positional arg

        # Verify status cleared to READY
        assert updated_request.status == TurnRequestStatus.READY
        assert updated_request.metadata == {}


@pytest.mark.asyncio
async def test_echo_clearing_multiple_actions_partial(mock_mediator):
    """Test partial echo clearing with multiple pending actions."""
    agent_id = "agent_123"
    action_id_1 = "action_abc"
    action_id_2 = "action_def"

    # Setup turn request with two pending actions
    turn_request = MUDTurnRequest(
        turn_id="turn_001",
        status=TurnRequestStatus.PENDING,
        sequence_id=100,
        metadata={"pending_action_ids": [action_id_1, action_id_2]}
    )
    mock_mediator._get_turn_request.return_value = turn_request

    # Create echo for first action only
    event = MUDEvent(
        event_type=EventType.SPEECH,
        actor="Agent",
        actor_id="char_123",
        action_id=action_id_1,
        room_id="test_room",
        content="You say, 'Hello'",
    )

    with patch("aim_mud_types.client.RedisMUDClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        await mock_mediator._clear_pending_for_echo(event, [agent_id])

        # Verify update was called
        assert mock_client.update_turn_request.called
        call_args = mock_client.update_turn_request.call_args
        updated_request = call_args[0][1]

        # Verify still PENDING with one action remaining
        assert updated_request.status == TurnRequestStatus.PENDING
        assert updated_request.metadata["pending_action_ids"] == [action_id_2]


@pytest.mark.asyncio
async def test_echo_clearing_no_match(mock_mediator):
    """Test echo clearing when action_id doesn't match."""
    agent_id = "agent_123"

    # Setup turn request with different action_id
    turn_request = MUDTurnRequest(
        turn_id="turn_001",
        status=TurnRequestStatus.PENDING,
        sequence_id=100,
        metadata={"pending_action_ids": ["action_other"]}
    )
    mock_mediator._get_turn_request.return_value = turn_request

    # Create echo event with non-matching action_id
    event = MUDEvent(
        event_type=EventType.SPEECH,
        actor="Agent",
        actor_id="char_123",
        action_id="action_nomatch",
        room_id="test_room",
        content="You say, 'Hello'",
    )

    with patch("aim_mud_types.client.RedisMUDClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        await mock_mediator._clear_pending_for_echo(event, [agent_id])

        # Verify NO update was called (no match)
        assert not mock_client.update_turn_request.called


@pytest.mark.asyncio
async def test_echo_clearing_agent_not_pending(mock_mediator):
    """Test echo clearing when agent is not in PENDING status."""
    agent_id = "agent_123"
    action_id = "action_abc"

    # Setup turn request in READY status (not PENDING)
    turn_request = MUDTurnRequest(
        turn_id="turn_001",
        status=TurnRequestStatus.READY,
        sequence_id=100,
        metadata={}
    )
    mock_mediator._get_turn_request.return_value = turn_request

    # Create echo event
    event = MUDEvent(
        event_type=EventType.SPEECH,
        actor="Agent",
        actor_id="char_123",
        action_id=action_id,
        room_id="test_room",
        content="You say, 'Hello'",
    )

    with patch("aim_mud_types.client.RedisMUDClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        await mock_mediator._clear_pending_for_echo(event, [agent_id])

        # Verify NO update was called (agent not PENDING)
        assert not mock_client.update_turn_request.called


@pytest.mark.asyncio
async def test_echo_clearing_error_handling(mock_mediator):
    """Test echo clearing handles errors gracefully."""
    agent_id = "agent_123"
    action_id = "action_abc"

    # Setup turn request
    turn_request = MUDTurnRequest(
        turn_id="turn_001",
        status=TurnRequestStatus.PENDING,
        sequence_id=100,
        metadata={"pending_action_ids": [action_id]}
    )
    mock_mediator._get_turn_request.return_value = turn_request

    # Create echo event
    event = MUDEvent(
        event_type=EventType.SPEECH,
        actor="Agent",
        actor_id="char_123",
        action_id=action_id,
        room_id="test_room",
        content="You say, 'Hello'",
    )

    with patch("aim_mud_types.client.RedisMUDClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.update_turn_request.side_effect = Exception("Redis error")
        mock_client_class.return_value = mock_client

        # Should NOT raise exception
        await mock_mediator._clear_pending_for_echo(event, [agent_id])

        # Error was caught and logged, execution continued
        assert mock_client.update_turn_request.called
