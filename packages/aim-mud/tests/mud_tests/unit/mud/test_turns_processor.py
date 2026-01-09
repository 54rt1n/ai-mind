# tests/unit/mud/test_turns_processor.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for turn processor self-action guidance formatting.

This test covers the bug where format_self_action_guidance() calls
_format_self_event() with underscore prefix, but the actual function
is defined as format_self_event() without underscore.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from aim_mud_types import EventType, MUDEvent, MUDSession, RoomState, EntityState
from andimud_worker.turns.processor.base import BaseTurnProcessor


@pytest.fixture
def mock_worker():
    """Create a mock worker with minimal session setup."""
    worker = MagicMock()

    # Create session without pending_self_actions (removed field)
    worker.session = MUDSession(
        agent_id="andi",
        persona_id="andi",
        current_room=RoomState(
            room_id="#123",
            name="The Kitchen",
            description="A warm kitchen.",
            exits={"north": "#124"},
        ),
        entities_present=[
            EntityState(
                entity_id="char1",
                name="Andi",
                entity_type="ai",
                is_self=True,
            ),
        ],
        pending_events=[],
    )

    # Mock async methods
    worker._load_agent_world_state = AsyncMock(return_value=("#123", "char1"))
    worker._load_room_profile = AsyncMock()
    worker.conversation_manager = MagicMock()
    worker.conversation_manager.push_user_turn = AsyncMock()

    return worker


class TestTurnProcessor:
    """Tests for BaseTurnProcessor.setup_turn() basic functionality."""

    @pytest.mark.asyncio
    async def test_setup_turn_with_self_actions(self, mock_worker):
        """Test setup_turn() updates session and pushes user turn.

        Note: pending_self_actions has been removed. Self-action guidance
        is now generated during turn processing via _create_event() in
        PhasedTurnProcessor, not during setup.
        """
        # Create a concrete processor subclass for testing
        class TestProcessor(BaseTurnProcessor):
            async def _decide_action(self, turn_request, events):
                return [], ""

        processor = TestProcessor(mock_worker)

        # Create some external events
        events = [
            MUDEvent(
                event_id="1704096000002-0",
                event_type=EventType.SPEECH,
                actor="Prax",
                actor_id="prax_id",
                room_id="#123",
                room_name="The Kitchen",
                content="Hello, Andi!",
            ),
        ]

        # Should complete without error
        await processor.setup_turn(events)

        # Verify session was updated
        assert mock_worker.session.pending_events == events
        assert mock_worker.session.last_event_time == events[-1].timestamp

        # Verify push_user_turn was called
        mock_worker.conversation_manager.push_user_turn.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_turn_without_self_actions(self, mock_worker):
        """Test setup_turn() with regular events."""
        class TestProcessor(BaseTurnProcessor):
            async def _decide_action(self, turn_request, events):
                return [], ""

        processor = TestProcessor(mock_worker)

        events = [
            MUDEvent(
                event_id="1704096000002-0",
                event_type=EventType.SPEECH,
                actor="Prax",
                actor_id="prax_id",
                room_id="#123",
                content="Hello!",
            ),
        ]

        await processor.setup_turn(events)

        # Verify session updated correctly
        assert mock_worker.session.pending_events == events

    @pytest.mark.asyncio
    async def test_setup_turn_with_various_self_action_types(self, mock_worker):
        """Test setup_turn() handles different event types correctly."""
        class TestProcessor(BaseTurnProcessor):
            async def _decide_action(self, turn_request, events):
                return [], ""

        processor = TestProcessor(mock_worker)

        # Mix of different event types
        events = [
            MUDEvent(
                event_id="1704096000000-0",
                event_type=EventType.MOVEMENT,
                actor="Prax",
                actor_id="prax_id",
                room_id="#123",
                room_name="The Garden",
                content="arrives from the north",
            ),
            MUDEvent(
                event_id="1704096000001-0",
                event_type=EventType.EMOTE,
                actor="Nova",
                actor_id="nova_id",
                room_id="#123",
                content="smiles warmly",
            ),
        ]

        await processor.setup_turn(events)

        # Verify all events were stored
        assert len(mock_worker.session.pending_events) == 2
        assert mock_worker.session.pending_events == events
