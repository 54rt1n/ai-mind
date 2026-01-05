# tests/unit/mud/test_adapter.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for MUD event-to-chat-turn adapter."""

import pytest
from datetime import datetime, timezone

from andimud_worker.adapter import (
    MAX_RECENT_TURNS,
    build_system_prompt,
    build_current_context,
    format_event,
    format_turn_events,
    format_turn_response,
    entries_to_chat_turns,
)
from andimud_worker.conversation import MUDConversationEntry
from aim.constants import DOC_MUD_WORLD, DOC_MUD_AGENT
from andimud_worker.session import (
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
        EntityState(
            entity_id="obj1",
            name="Fountain",
            entity_type="object",
            description="A softly bubbling silver fountain.",
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
        assert result == 'Prax says, "Hello, Andi! Happy New Year!"'

    def test_format_emote_event(self, sample_emote_event: MUDEvent):
        """Test formatting an emote event."""
        result = format_event(sample_emote_event)
        assert result == "*Prax smiles warmly*"

    def test_format_emote_with_quotes(self):
        """Test formatting an emote with quoted speech splits action and speech."""
        event = MUDEvent(
            event_type=EventType.EMOTE,
            actor="Prax",
            room_id="#123",
            content='looks at you. "Hello, Andi!"',
        )
        result = format_event(event)
        assert result == '*Prax looks at you.* "Hello, Andi!"'

    def test_format_emote_with_quotes_at_start(self):
        """Test formatting an emote where quote is at the very start."""
        event = MUDEvent(
            event_type=EventType.EMOTE,
            actor="Prax",
            room_id="#123",
            content='"Hello!" she said.',
        )
        result = format_event(event)
        # When quote is at start, just wrap actor
        assert result == '*Prax* "Hello!" she said.'

    def test_format_movement_enter_event(self):
        """Test formatting a movement event for arrival."""
        event = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Prax",
            room_id="#123",
            content="enters from the north",
        )
        result = format_event(event)
        assert result == "*You see Prax has arrived.*"

    def test_format_movement_arrive_event(self):
        """Test formatting a movement event with 'arrive' in content."""
        event = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Prax",
            room_id="#123",
            content="has arrived",
        )
        result = format_event(event)
        assert result == "*You see Prax has arrived.*"

    def test_format_movement_leave_event(self):
        """Test formatting a movement event for departure."""
        event = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Prax",
            room_id="#123",
            content="leaves to the north",
        )
        result = format_event(event)
        assert result == "*You watch Prax leave.*"

    def test_format_object_event(self):
        """Test formatting an object event."""
        event = MUDEvent(
            event_type=EventType.OBJECT,
            actor="Prax",
            room_id="#123",
            content="picks up a golden key",
        )
        result = format_event(event)
        assert result == "*Prax picks up a golden key*"

    def test_format_ambient_event(self):
        """Test formatting an ambient event."""
        event = MUDEvent(
            event_type=EventType.AMBIENT,
            actor="system",
            room_id="#123",
            content="A cool breeze rustles the leaves.",
        )
        result = format_event(event)
        assert result == "A cool breeze rustles the leaves."

    def test_format_system_event(self):
        """Test formatting a system event."""
        event = MUDEvent(
            event_type=EventType.SYSTEM,
            actor="system",
            room_id="#123",
            content="Server maintenance in 5 minutes.",
        )
        result = format_event(event)
        assert result == "[System] Server maintenance in 5 minutes."


class TestFormatTurnEvents:
    """Tests for format_turn_events function."""

    def test_format_turn_with_events(
        self, sample_speech_event: MUDEvent, sample_emote_event: MUDEvent
    ):
        """Test formatting a turn with multiple events."""
        turn = MUDTurn(events_received=[sample_speech_event, sample_emote_event])
        result = format_turn_events(turn)

        assert 'Prax says, "Hello, Andi! Happy New Year!"' in result
        assert "*Prax smiles warmly*" in result
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
        action = MUDAction(tool="pose", args={"action": "smiles softly"})
        turn = MUDTurn(actions_taken=[action])
        result = format_turn_response(turn)
        assert result == "[Action: pose] pose smiles softly"

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

        # Check events (default include_events=True) - now formatted directly
        assert 'Prax says, "Hello, Andi! Happy New Year!"' in result

        # Check formatting guidance
        assert "[~~ Link Format Guidance ~~]" in result

    def test_build_current_context_exclude_events(
        self,
        sample_room: RoomState,
        sample_entities: list[EntityState],
        sample_speech_event: MUDEvent,
    ):
        """Test building current context with include_events=False."""
        session = MUDSession(
            agent_id="andi",
            persona_id="andi",
            current_room=sample_room,
            entities_present=sample_entities,
            pending_events=[sample_speech_event],
        )

        result = build_current_context(session, include_events=False)

        # Events should NOT be included
        assert "Hello, Andi!" not in result

        # Format guidance should still be there
        assert "[~~ Link Format Guidance ~~]" in result

    def test_build_current_context_exclude_events_idle_mode(self):
        """Test that idle guidance is NOT added when include_events=False."""
        session = MUDSession(
            agent_id="andi",
            persona_id="andi",
            current_room=None,
            entities_present=[],
            pending_events=[],
        )

        result = build_current_context(session, idle_mode=True, include_events=False)

        # Idle guidance is part of events section, so should be excluded
        assert "You don't see anything of note occuring" not in result
        # Format guidance should still be there
        assert "[~~ Link Format Guidance ~~]" in result

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

        # Should just have format guidance when empty
        assert "[~~ Link Format Guidance ~~]" in result

    def test_build_current_context_no_exits(self):
        """Test building current context with room but no exits (idle mode)."""
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

        result = build_current_context(session, idle_mode=True)

        # With no events and idle_mode, should have idle guidance
        assert "You don't see anything of note occuring" in result
        assert "[~~ Link Format Guidance ~~]" in result

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
        assert "<objects>" not in result

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


class TestBuildSystemPrompt:
    """Tests for build_system_prompt function."""

    def test_build_system_prompt_with_room_and_entities(
        self,
        sample_persona: Persona,
        sample_room: RoomState,
        sample_entities: list[EntityState],
    ):
        """Test building system prompt returns only persona prompt."""
        session = MUDSession(
            agent_id="andi",
            persona_id="andi",
            current_room=sample_room,
            entities_present=sample_entities,
        )

        result = build_system_prompt(session, sample_persona)

        # Should contain persona information
        assert "Andi Valentine" in result

        # Should NOT contain world state (that goes in memory block now)
        assert "[Current Location]" not in result
        assert "The Garden of Reflection" not in result
        assert "Present:" not in result

    def test_build_system_prompt_no_room(self, sample_persona: Persona):
        """Test building system prompt with no room still returns persona."""
        session = MUDSession(
            agent_id="andi",
            persona_id="andi",
            current_room=None,
            entities_present=[],
        )

        result = build_system_prompt(session, sample_persona)

        # Should still have persona
        assert "Andi Valentine" in result
        # Should NOT have world state sections
        assert "[Current Location]" not in result
        assert "You are alone." not in result

    def test_build_system_prompt_alone(
        self, sample_persona: Persona, sample_room: RoomState
    ):
        """Test building system prompt when alone still returns only persona."""
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

        # Should have persona
        assert "Andi Valentine" in result
        # Should NOT have world state
        assert "You are alone." not in result
        assert "Present:" not in result


class TestEntriesToChatTurns:
    """Tests for entries_to_chat_turns function."""

    def test_entries_to_chat_turns_empty(self):
        """Test converting empty list of entries."""
        result = entries_to_chat_turns([])
        assert result == []

    def test_entries_to_chat_turns_single_user(self):
        """Test converting a single user entry."""
        # Content is now pure prose (no XML wrapper)
        entry = MUDConversationEntry(
            role="user",
            content='Prax says, "Hello!"',
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv_1",
            sequence_no=0,
        )
        result = entries_to_chat_turns([entry])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Prax says" in result[0]["content"]

    def test_entries_to_chat_turns_multiple_entries(self):
        """Test converting multiple entries in alternating roles."""
        # Content is now pure prose (no XML wrapper)
        entries = [
            MUDConversationEntry(
                role="user",
                content='Prax says, "Hello!"',
                tokens=10,
                document_type=DOC_MUD_WORLD,
                conversation_id="test_conv_1",
                sequence_no=0,
            ),
            MUDConversationEntry(
                role="assistant",
                content="Hello Prax! It's wonderful to see you.",
                tokens=15,
                document_type=DOC_MUD_AGENT,
                conversation_id="test_conv_1",
                sequence_no=1,
            ),
            MUDConversationEntry(
                role="user",
                content="Prax smiles warmly",
                tokens=8,
                document_type=DOC_MUD_WORLD,
                conversation_id="test_conv_1",
                sequence_no=2,
            ),
        ]

        result = entries_to_chat_turns(entries)

        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"

        # Verify content preserved
        assert "Prax says" in result[0]["content"]
        assert "wonderful to see you" in result[1]["content"]
        assert "smiles warmly" in result[2]["content"]

    def test_entries_to_chat_turns_preserves_order(self):
        """Test that entries are returned in the same order as input."""
        entries = [
            MUDConversationEntry(
                role="user",
                content="First message",
                tokens=5,
                document_type=DOC_MUD_WORLD,
                conversation_id="test_conv_1",
                sequence_no=0,
            ),
            MUDConversationEntry(
                role="assistant",
                content="Second message",
                tokens=5,
                document_type=DOC_MUD_AGENT,
                conversation_id="test_conv_1",
                sequence_no=1,
            ),
            MUDConversationEntry(
                role="user",
                content="Third message",
                tokens=5,
                document_type=DOC_MUD_WORLD,
                conversation_id="test_conv_1",
                sequence_no=2,
            ),
        ]

        result = entries_to_chat_turns(entries)

        assert result[0]["content"] == "First message"
        assert result[1]["content"] == "Second message"
        assert result[2]["content"] == "Third message"
