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

    # Create session with pending_self_actions
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
        pending_self_actions=[
            MUDEvent(
                event_id="1704096000000-0",
                event_type=EventType.MOVEMENT,
                actor="Andi",
                room_id="#123",
                room_name="The Kitchen",
                content="enters from the west",
            ),
            MUDEvent(
                event_id="1704096000001-0",
                event_type=EventType.OBJECT,
                actor="Andi",
                room_id="#123",
                target="golden key",
                content="picks up a golden key",
            ),
        ],
    )

    # Mock async methods
    worker._load_agent_world_state = AsyncMock(return_value=("#123", "char1"))
    worker._load_room_profile = AsyncMock()
    worker.conversation_manager = MagicMock()
    worker.conversation_manager.push_user_turn = AsyncMock()

    return worker


class TestTurnProcessor:
    """Tests for BaseTurnProcessor.setup_turn() with self-action events."""

    @pytest.mark.asyncio
    async def test_setup_turn_with_self_actions(self, mock_worker):
        """Test setup_turn() formats self-action guidance without NameError.

        This test exposes the bug where format_self_action_guidance() calls
        _format_self_event() (with underscore) instead of format_self_event()
        (without underscore).

        Before fix: NameError: name '_format_self_event' is not defined
        After fix: Action guidance is properly formatted
        """
        # Create a concrete processor subclass for testing
        class TestProcessor(BaseTurnProcessor):
            async def _decide_action(self, events):
                return [], ""

        processor = TestProcessor(mock_worker)

        # Create some external events
        events = [
            MUDEvent(
                event_id="1704096000002-0",
                event_type=EventType.SPEECH,
                actor="Prax",
                room_id="#123",
                room_name="The Kitchen",
                content="Hello, Andi!",
            ),
        ]

        # This should NOT raise NameError
        await processor.setup_turn(events)

        # Verify that action guidance was set
        assert processor._action_guidance != ""

        # Verify guidance contains formatted self-actions
        assert "[!! Action: You moved to The Kitchen. !!]" in processor._action_guidance
        assert "[!! Action: You picked up golden key. !!]" in processor._action_guidance

        # Verify pending_self_actions was cleared
        assert mock_worker.session.pending_self_actions == []

    @pytest.mark.asyncio
    async def test_setup_turn_without_self_actions(self, mock_worker):
        """Test setup_turn() with no pending self-actions."""
        # Clear pending_self_actions
        mock_worker.session.pending_self_actions = []

        class TestProcessor(BaseTurnProcessor):
            async def _decide_action(self, events):
                return [], ""

        processor = TestProcessor(mock_worker)

        events = [
            MUDEvent(
                event_id="1704096000002-0",
                event_type=EventType.SPEECH,
                actor="Prax",
                room_id="#123",
                content="Hello!",
            ),
        ]

        await processor.setup_turn(events)

        # Verify action guidance is empty
        assert processor._action_guidance == ""

    @pytest.mark.asyncio
    async def test_setup_turn_with_various_self_action_types(self, mock_worker):
        """Test setup_turn() with different self-action event types."""
        # Set up various self-action types
        mock_worker.session.pending_self_actions = [
            MUDEvent(
                event_type=EventType.MOVEMENT,
                actor="Andi",
                room_id="#123",
                room_name="The Garden",
                content="moves to the garden",
            ),
            MUDEvent(
                event_type=EventType.OBJECT,
                actor="Andi",
                room_id="#123",
                target="silver key",
                content="drops a silver key",
            ),
            MUDEvent(
                event_type=EventType.EMOTE,
                actor="Andi",
                room_id="#123",
                content="smiles warmly",
            ),
        ]

        class TestProcessor(BaseTurnProcessor):
            async def _decide_action(self, events):
                return [], ""

        processor = TestProcessor(mock_worker)

        await processor.setup_turn([])

        # Verify all action types are formatted
        guidance = processor._action_guidance
        assert "You moved to The Garden." in guidance
        assert "You dropped silver key." in guidance
        assert "You expressed: smiles warmly" in guidance
