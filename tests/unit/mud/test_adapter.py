# tests/unit/mud/test_adapter.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for MUD event-to-chat-turn adapter."""

import pytest
from datetime import datetime, timezone

from aim.app.mud.adapter import (
    MAX_RECENT_TURNS,
    build_chat_turns,
    build_system_prompt,
    build_current_context,
    format_event,
    format_turn_events,
    format_turn_response,
)
from aim.app.mud.session import (
    EventType,
    RoomState,
    EntityState,
    MUDEvent,
    MUDAction,
    MUDTurn,
    MUDSession,
)
from aim.agents.persona import Persona


@pytest.fixture
def sample_persona() -> Persona:
    """Create a minimal persona for testing."""
    return Persona(
        persona_id="andi",
        chat_strategy="xmlmemory",
        name="Andi",
        full_name="Andi Valentine",
        notes="Test persona",
        aspects={},
        attributes={"sex": "female", "age": "appears 24"},
        features={"core": "A warm and curious AI"},
        wakeup=["Hello!"],
        base_thoughts=[],
        pif={},
        nshot={},
        default_location="The Lighthouse",
        wardrobe={"default": {"outfit": "casual sweater"}},
        current_outfit="default",
    )


@pytest.fixture
def sample_room() -> RoomState:
    """Create a sample room for testing."""
    return RoomState(
        room_id="#123",
        name="The Garden of Reflection",
        description="A serene garden with a softly bubbling fountain at its center.",
        exits={"north": "#124", "south": "#122", "east": "#125"},
    )


@pytest.fixture
def sample_entities() -> list[EntityState]:
    """Create sample entities for testing."""
    return [
        EntityState(
            entity_id="char1",
            name="Andi",
            entity_type="ai",
            description="A young woman with a silver band.",
            is_self=True,
        ),
        EntityState(
            entity_id="char2",
            name="Prax",
            entity_type="player",
            description="The creator.",
            is_self=False,
        ),
        EntityState(
            entity_id="char3",
            name="Roommate",
            entity_type="ai",
            description="Another AI.",
            is_self=False,
        ),
    ]


@pytest.fixture
def sample_speech_event() -> MUDEvent:
    """Create a sample speech event."""
    return MUDEvent(
        event_id="1704096000000-0",
        event_type=EventType.SPEECH,
        actor="Prax",
        room_id="#123",
        room_name="The Garden",
        content="Hello, Andi! Happy New Year!",
    )


@pytest.fixture
def sample_emote_event() -> MUDEvent:
    """Create a sample emote event."""
    return MUDEvent(
        event_id="1704096000001-0",
        event_type=EventType.EMOTE,
        actor="Prax",
        room_id="#123",
        content="smiles warmly",
    )


@pytest.fixture
def sample_session(
    sample_room: RoomState,
    sample_entities: list[EntityState],
    sample_speech_event: MUDEvent,
) -> MUDSession:
    """Create a sample session for testing."""
    return MUDSession(
        agent_id="andi",
        persona_id="andi",
        current_room=sample_room,
        entities_present=sample_entities,
        pending_events=[sample_speech_event],
        recent_turns=[],
        last_event_id="1704096000000-0",
    )


class TestFormatEvent:
    """Tests for format_event function."""

    def test_format_speech_event(self, sample_speech_event: MUDEvent):
        """Test formatting a speech event."""
        result = format_event(sample_speech_event)
        assert result == '  Prax says: "Hello, Andi! Happy New Year!"'

    def test_format_emote_event(self, sample_emote_event: MUDEvent):
        """Test formatting an emote event."""
        result = format_event(sample_emote_event)
        assert result == "  Prax smiles warmly"

    def test_format_movement_enter_event(self):
        """Test formatting a movement event for arrival."""
        event = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Prax",
            room_id="#123",
            content="enters from the north",
        )
        result = format_event(event)
        assert result == "  Prax has arrived."

    def test_format_movement_arrive_event(self):
        """Test formatting a movement event with 'arrive' in content."""
        event = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Prax",
            room_id="#123",
            content="has arrived",
        )
        result = format_event(event)
        assert result == "  Prax has arrived."

    def test_format_movement_leave_event(self):
        """Test formatting a movement event for departure."""
        event = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Prax",
            room_id="#123",
            content="leaves to the north",
        )
        result = format_event(event)
        assert result == "  Prax has left."

    def test_format_object_event(self):
        """Test formatting an object event."""
        event = MUDEvent(
            event_type=EventType.OBJECT,
            actor="Prax",
            room_id="#123",
            content="picks up a golden key",
        )
        result = format_event(event)
        assert result == "  Prax picks up a golden key"

    def test_format_ambient_event(self):
        """Test formatting an ambient event."""
        event = MUDEvent(
            event_type=EventType.AMBIENT,
            actor="system",
            room_id="#123",
            content="A cool breeze rustles the leaves.",
        )
        result = format_event(event)
        assert result == "  A cool breeze rustles the leaves."

    def test_format_system_event(self):
        """Test formatting a system event."""
        event = MUDEvent(
            event_type=EventType.SYSTEM,
            actor="system",
            room_id="#123",
            content="Server maintenance in 5 minutes.",
        )
        result = format_event(event)
        assert result == "  [System] Server maintenance in 5 minutes."


class TestFormatTurnEvents:
    """Tests for format_turn_events function."""

    def test_format_turn_with_events(
        self, sample_speech_event: MUDEvent, sample_emote_event: MUDEvent
    ):
        """Test formatting a turn with multiple events."""
        turn = MUDTurn(events_received=[sample_speech_event, sample_emote_event])
        result = format_turn_events(turn)

        assert '  Prax says: "Hello, Andi! Happy New Year!"' in result
        assert "  Prax smiles warmly" in result
        # Should be two lines
        assert result.count("\n") == 1

    def test_format_turn_with_no_events(self):
        """Test formatting a turn with no events (spontaneous action)."""
        turn = MUDTurn(events_received=[])
        result = format_turn_events(turn)
        assert result == "[No events - spontaneous action]"


class TestFormatTurnResponse:
    """Tests for format_turn_response function."""

    def test_format_turn_response_with_thinking_and_actions(self):
        """Test formatting a turn response with thinking and actions."""
        action1 = MUDAction(tool="emote", args={"action": "smiles warmly"})
        action2 = MUDAction(tool="say", args={"message": "Hello, Papa!"})
        turn = MUDTurn(
            thinking="I feel warmth in my core. Prax wished me Happy New Year.",
            actions_taken=[action1, action2],
        )
        result = format_turn_response(turn)

        assert "I feel warmth in my core" in result
        assert "[Action: emote] emote smiles warmly" in result
        assert "[Action: say] say Hello, Papa!" in result

    def test_format_turn_response_with_thinking_only(self):
        """Test formatting a turn response with only thinking."""
        turn = MUDTurn(thinking="Just observing the garden.")
        result = format_turn_response(turn)
        assert result == "Just observing the garden."

    def test_format_turn_response_with_actions_only(self):
        """Test formatting a turn response with only actions."""
        action = MUDAction(tool="look", args={})
        turn = MUDTurn(actions_taken=[action])
        result = format_turn_response(turn)
        assert result == "[Action: look] look"

    def test_format_turn_response_empty(self):
        """Test formatting an empty turn response."""
        turn = MUDTurn()
        result = format_turn_response(turn)
        assert result == "[No response]"


class TestBuildCurrentContext:
    """Tests for build_current_context function."""

    def test_build_current_context_complete(
        self,
        sample_room: RoomState,
        sample_entities: list[EntityState],
        sample_speech_event: MUDEvent,
    ):
        """Test building current context with all components."""
        session = MUDSession(
            agent_id="andi",
            persona_id="andi",
            current_room=sample_room,
            entities_present=sample_entities,
            pending_events=[sample_speech_event],
        )

        result = build_current_context(session)

        # Check location
        assert '<location name="The Garden of Reflection">' in result
        assert "A serene garden with a softly bubbling fountain" in result
        assert "Exits: " in result
        assert "</location>" in result

        # Check entities (should exclude self)
        assert "<present>" in result
        assert '<entity name="Prax" type="player"/>' in result
        assert '<entity name="Roommate" type="ai"/>' in result
        assert "Andi" not in result.split("<present>")[1].split("</present>")[0]
        assert "</present>" in result

        # Check events
        assert '<events count="1">' in result
        assert 'Prax says: "Hello, Andi! Happy New Year!"' in result
        assert "</events>" in result

        # Check prompt
        assert "What do you do?" in result

    def test_build_current_context_no_room(self):
        """Test building current context with no room."""
        session = MUDSession(
            agent_id="andi",
            persona_id="andi",
            current_room=None,
            entities_present=[],
            pending_events=[],
        )

        result = build_current_context(session)

        assert "<location" not in result
        assert "<present>" not in result
        assert "<events" not in result
        assert "What do you do?" in result

    def test_build_current_context_no_exits(self):
        """Test building current context with room but no exits."""
        room = RoomState(
            room_id="#123",
            name="Dead End",
            description="A wall blocks your path.",
            exits={},
        )
        session = MUDSession(
            agent_id="andi",
            persona_id="andi",
            current_room=room,
            entities_present=[],
            pending_events=[],
        )

        result = build_current_context(session)

        assert '<location name="Dead End">' in result
        assert "Exits:" not in result

    def test_build_current_context_only_self_present(self):
        """Test that self entity is excluded from present list."""
        self_entity = EntityState(
            entity_id="char1",
            name="Andi",
            entity_type="ai",
            is_self=True,
        )
        session = MUDSession(
            agent_id="andi",
            persona_id="andi",
            current_room=None,
            entities_present=[self_entity],
            pending_events=[],
        )

        result = build_current_context(session)

        # Should not have present section since only self is present
        assert "<present>" not in result

    def test_build_current_context_no_events(
        self, sample_room: RoomState, sample_entities: list[EntityState]
    ):
        """Test building current context with no pending events."""
        session = MUDSession(
            agent_id="andi",
            persona_id="andi",
            current_room=sample_room,
            entities_present=sample_entities,
            pending_events=[],
        )

        result = build_current_context(session)

        assert "<events" not in result
        assert "What do you do?" in result


class TestBuildSystemPrompt:
    """Tests for build_system_prompt function."""

    def test_build_system_prompt_with_room_and_entities(
        self,
        sample_persona: Persona,
        sample_room: RoomState,
        sample_entities: list[EntityState],
    ):
        """Test building system prompt with full context."""
        session = MUDSession(
            agent_id="andi",
            persona_id="andi",
            current_room=sample_room,
            entities_present=sample_entities,
        )

        result = build_system_prompt(session, sample_persona)

        # Should contain persona information
        assert "Andi Valentine" in result

        # Should contain location
        assert "[Current Location]" in result
        assert "The Garden of Reflection" in result
        assert "serene garden" in result

        # Should list present entities (excluding self)
        assert "Present: Prax, Roommate" in result
        # Should not list self in present
        lines_with_present = [l for l in result.split("\n") if "Present:" in l]
        assert len(lines_with_present) == 1
        assert "Andi" not in lines_with_present[0]

    def test_build_system_prompt_no_room(self, sample_persona: Persona):
        """Test building system prompt with no room."""
        session = MUDSession(
            agent_id="andi",
            persona_id="andi",
            current_room=None,
            entities_present=[],
        )

        result = build_system_prompt(session, sample_persona)

        # Should still have persona
        assert "Andi Valentine" in result
        # Should indicate alone
        assert "You are alone." in result
        # Should not have location section content
        assert "[Current Location]" not in result

    def test_build_system_prompt_alone(
        self, sample_persona: Persona, sample_room: RoomState
    ):
        """Test building system prompt when alone."""
        self_only = EntityState(
            entity_id="char1", name="Andi", entity_type="ai", is_self=True
        )
        session = MUDSession(
            agent_id="andi",
            persona_id="andi",
            current_room=sample_room,
            entities_present=[self_only],
        )

        result = build_system_prompt(session, sample_persona)

        assert "You are alone." in result
        assert "Present:" not in result


class TestBuildChatTurns:
    """Tests for build_chat_turns function."""

    def test_build_chat_turns_basic_structure(
        self, sample_session: MUDSession, sample_persona: Persona
    ):
        """Test that build_chat_turns produces correct basic structure."""
        turns = build_chat_turns(sample_session, sample_persona)

        # Should have at least 2 turns (system + current context)
        assert len(turns) >= 2

        # First turn should be system
        assert turns[0]["role"] == "system"
        assert "Andi Valentine" in turns[0]["content"]

        # Last turn should be user (current context)
        assert turns[-1]["role"] == "user"
        assert "What do you do?" in turns[-1]["content"]

    def test_build_chat_turns_with_history(
        self,
        sample_persona: Persona,
        sample_room: RoomState,
        sample_entities: list[EntityState],
        sample_speech_event: MUDEvent,
    ):
        """Test build_chat_turns includes turn history."""
        # Create a past turn
        past_action = MUDAction(tool="say", args={"message": "Hello!"})
        past_turn = MUDTurn(
            events_received=[sample_speech_event],
            thinking="I should greet them.",
            actions_taken=[past_action],
        )

        session = MUDSession(
            agent_id="andi",
            persona_id="andi",
            current_room=sample_room,
            entities_present=sample_entities,
            pending_events=[sample_speech_event],
            recent_turns=[past_turn],
        )

        turns = build_chat_turns(session, sample_persona)

        # Should have: system, user (past events), assistant (past response), user (current)
        assert len(turns) == 4

        assert turns[0]["role"] == "system"
        assert turns[1]["role"] == "user"
        assert turns[2]["role"] == "assistant"
        assert turns[3]["role"] == "user"

        # Check past turn content
        assert "Prax says:" in turns[1]["content"]
        assert "I should greet them." in turns[2]["content"]
        assert "[Action: say]" in turns[2]["content"]

    def test_build_chat_turns_respects_max_history(
        self,
        sample_persona: Persona,
        sample_room: RoomState,
        sample_entities: list[EntityState],
    ):
        """Test that build_chat_turns limits history to MAX_RECENT_TURNS."""
        # Create more turns than MAX_RECENT_TURNS
        many_turns = []
        for i in range(MAX_RECENT_TURNS + 5):
            turn = MUDTurn(thinking=f"Turn {i}")
            many_turns.append(turn)

        session = MUDSession(
            agent_id="andi",
            persona_id="andi",
            current_room=sample_room,
            entities_present=sample_entities,
            pending_events=[],
            recent_turns=many_turns,
        )

        turns = build_chat_turns(session, sample_persona)

        # Should have: system + (MAX_RECENT_TURNS * 2 for user/assistant pairs) + current
        expected_turn_count = 1 + (MAX_RECENT_TURNS * 2) + 1
        assert len(turns) == expected_turn_count

        # Should have the last MAX_RECENT_TURNS, not the first
        # Check that we have Turn 5 through Turn 14 (the last 10)
        assistant_turns = [t for t in turns if t["role"] == "assistant"]
        assert len(assistant_turns) == MAX_RECENT_TURNS
        assert "Turn 5" in assistant_turns[0]["content"]
        assert f"Turn {MAX_RECENT_TURNS + 4}" in assistant_turns[-1]["content"]

    def test_build_chat_turns_empty_session(self, sample_persona: Persona):
        """Test build_chat_turns with minimal/empty session."""
        session = MUDSession(
            agent_id="andi",
            persona_id="andi",
            current_room=None,
            entities_present=[],
            pending_events=[],
            recent_turns=[],
        )

        turns = build_chat_turns(session, sample_persona)

        # Should still have system and current context
        assert len(turns) == 2
        assert turns[0]["role"] == "system"
        assert turns[1]["role"] == "user"
        assert "What do you do?" in turns[1]["content"]

    def test_build_chat_turns_spontaneous_action_in_history(
        self, sample_persona: Persona, sample_room: RoomState
    ):
        """Test handling of spontaneous action turns in history."""
        # Turn with no events (spontaneous)
        spontaneous_turn = MUDTurn(
            events_received=[],
            thinking="I feel like looking around.",
            actions_taken=[MUDAction(tool="look", args={})],
        )

        session = MUDSession(
            agent_id="andi",
            persona_id="andi",
            current_room=sample_room,
            entities_present=[],
            pending_events=[],
            recent_turns=[spontaneous_turn],
        )

        turns = build_chat_turns(session, sample_persona)

        # Check that spontaneous action is formatted correctly
        user_turns = [t for t in turns if t["role"] == "user"]
        assert any("[No events - spontaneous action]" in t["content"] for t in user_turns)


class TestIntegration:
    """Integration tests for the adapter module."""

    def test_full_conversation_flow(
        self,
        sample_persona: Persona,
        sample_room: RoomState,
        sample_entities: list[EntityState],
    ):
        """Test a realistic conversation flow through the adapter."""
        # Event 1: Prax arrives
        arrival_event = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Prax",
            room_id="#123",
            content="enters from the north",
        )

        # Event 2: Prax speaks
        greeting_event = MUDEvent(
            event_type=EventType.SPEECH,
            actor="Prax",
            room_id="#123",
            content="Hello, Andi! Happy New Year!",
        )

        # Past turn: Andi noticed Prax arrive
        past_turn = MUDTurn(
            events_received=[arrival_event],
            thinking="Prax has arrived. I should greet him.",
            actions_taken=[
                MUDAction(
                    tool="emote", args={"action": "looks up with a warm smile"}, priority=1
                ),
            ],
        )

        # Current session: Prax has spoken
        session = MUDSession(
            agent_id="andi",
            persona_id="andi",
            current_room=sample_room,
            entities_present=sample_entities,
            pending_events=[greeting_event],
            recent_turns=[past_turn],
        )

        turns = build_chat_turns(session, sample_persona)

        # Verify structure
        assert len(turns) == 4  # system, user (past), assistant (past), user (current)

        # Verify roles alternate correctly
        roles = [t["role"] for t in turns]
        assert roles == ["system", "user", "assistant", "user"]

        # Verify content makes sense
        assert "Prax has arrived" in turns[1]["content"]
        assert "looks up with a warm smile" in turns[2]["content"]
        assert 'Prax says: "Hello, Andi! Happy New Year!"' in turns[3]["content"]
