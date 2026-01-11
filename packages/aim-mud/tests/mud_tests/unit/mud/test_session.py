# tests/unit/mud/test_session.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for MUD session models."""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from aim_mud_types import (
    EventType,
    ActorType,
    RoomState,
    EntityState,
    MUDEvent,
    MUDAction,
    MUDTurn,
    MUDSession,
)


class TestEventType:
    """Tests for EventType enum."""

    def test_event_type_values(self):
        """Test that all expected event types exist."""
        assert EventType.SPEECH == "speech"
        assert EventType.EMOTE == "emote"
        assert EventType.MOVEMENT == "movement"
        assert EventType.OBJECT == "object"
        assert EventType.AMBIENT == "ambient"
        assert EventType.SYSTEM == "system"


class TestActorType:
    """Tests for ActorType enum."""

    def test_actor_type_values(self):
        """Test that all expected actor types exist."""
        assert ActorType.PLAYER == "player"
        assert ActorType.AI == "ai"
        assert ActorType.NPC == "npc"
        assert ActorType.SYSTEM == "system"


class TestRoomState:
    """Tests for RoomState model."""

    def test_room_state_required_fields(self):
        """Test RoomState with required fields only."""
        room = RoomState(room_id="#123", name="The Garden")

        assert room.room_id == "#123"
        assert room.name == "The Garden"
        assert room.description == ""
        assert room.exits == {}

    def test_room_state_complete(self):
        """Test RoomState with all fields."""
        room = RoomState(
            room_id="#123",
            name="The Garden of Reflection",
            description="A serene garden with a fountain.",
            exits={"north": "#124", "south": "#122"},
        )

        assert room.room_id == "#123"
        assert room.name == "The Garden of Reflection"
        assert room.description == "A serene garden with a fountain."
        assert room.exits == {"north": "#124", "south": "#122"}

    def test_room_state_model_validate(self):
        """Test RoomState.model_validate factory method."""
        data = {
            "room_id": "#123",
            "name": "The Garden",
            "description": "A garden",
            "exits": {"north": "#124"},
        }
        room = RoomState.model_validate(data)

        assert room.room_id == "#123"
        assert room.name == "The Garden"
        assert room.description == "A garden"
        assert room.exits == {"north": "#124"}

    def test_room_state_model_validate_missing_fields(self):
        """Test RoomState.model_validate with missing optional fields."""
        data = {"room_id": "#123", "name": "Minimal Room"}
        room = RoomState.model_validate(data)

        assert room.room_id == "#123"
        assert room.name == "Minimal Room"
        assert room.description == ""
        assert room.exits == {}


class TestEntityState:
    """Tests for EntityState model."""

    def test_entity_state_required_fields(self):
        """Test EntityState with required fields only."""
        entity = EntityState(entity_id="obj123", name="Prax")

        assert entity.entity_id == "obj123"
        assert entity.name == "Prax"
        assert entity.entity_type == "object"
        assert entity.description == ""
        assert entity.is_self is False

    def test_entity_state_complete(self):
        """Test EntityState with all fields."""
        entity = EntityState(
            entity_id="char123",
            name="Andi",
            entity_type="ai",
            description="A young woman with a silver band.",
            is_self=True,
        )

        assert entity.entity_id == "char123"
        assert entity.name == "Andi"
        assert entity.entity_type == "ai"
        assert entity.description == "A young woman with a silver band."
        assert entity.is_self is True

    def test_entity_state_model_validate(self):
        """Test EntityState.model_validate factory method."""
        data = {
            "entity_id": "char123",
            "name": "Prax",
            "entity_type": "player",
            "description": "The creator.",
            "is_self": False,
        }
        entity = EntityState.model_validate(data)

        assert entity.entity_id == "char123"
        assert entity.name == "Prax"
        assert entity.entity_type == "player"
        assert entity.is_self is False

    def test_entity_state_model_validate_missing_fields(self):
        """Test EntityState.model_validate with missing optional fields."""
        data = {"entity_id": "obj1", "name": "Key"}
        entity = EntityState.model_validate(data)

        assert entity.entity_id == "obj1"
        assert entity.name == "Key"
        assert entity.entity_type == "object"
        assert entity.is_self is False


class TestMUDEvent:
    """Tests for MUDEvent model."""

    def test_event_required_fields(self):
        """Test MUDEvent with required fields only."""
        event = MUDEvent(
            event_type=EventType.SPEECH,
            actor="Prax",
            room_id="#123",
        )

        assert event.event_type == EventType.SPEECH
        assert event.actor == "Prax"
        assert event.room_id == "#123"
        assert event.event_id == ""
        assert event.actor_type == ActorType.PLAYER
        assert event.content == ""
        assert event.target is None

    def test_event_complete(self):
        """Test MUDEvent with all fields."""
        now = datetime.now(timezone.utc)
        event = MUDEvent(
            event_id="1704096000000-0",
            event_type=EventType.SPEECH,
            actor="Prax",
            actor_type=ActorType.PLAYER,
            room_id="#123",
            room_name="The Garden",
            content="Hello, Andi!",
            target="Andi",
            timestamp=now,
            metadata={"emote": False},
        )

        assert event.event_id == "1704096000000-0"
        assert event.event_type == EventType.SPEECH
        assert event.actor == "Prax"
        assert event.actor_type == ActorType.PLAYER
        assert event.room_id == "#123"
        assert event.room_name == "The Garden"
        assert event.content == "Hello, Andi!"
        assert event.target == "Andi"
        assert event.timestamp == now
        assert event.metadata == {"emote": False}

    def test_event_from_dict(self):
        """Test MUDEvent.from_dict factory method."""
        data = {
            "id": "1704096000000-0",
            "type": "speech",
            "actor": "Prax",
            "actor_type": "player",
            "room_id": "#123",
            "room_name": "The Garden",
            "content": "Hello!",
            "timestamp": "2025-01-01T04:00:00Z",
        }
        event = MUDEvent.from_dict(data)

        assert event.event_id == "1704096000000-0"
        assert event.event_type == EventType.SPEECH
        assert event.actor == "Prax"
        assert event.room_id == "#123"
        assert event.content == "Hello!"

    def test_event_from_dict_enriched(self):
        """Test MUDEvent.from_dict with enrichment data."""
        data = {
            "type": "speech",
            "actor": "Prax",
            "room_id": "#123",
            "content": "Hello!",
            "timestamp": "2025-01-01T04:00:00+00:00",
            "room_state": {"room_id": "#123", "name": "Garden"},
            "entities_present": [
                {"entity_id": "1", "name": "Andi"},
            ],
        }
        event = MUDEvent.from_dict(data)

        assert event.room_state == {"room_id": "#123", "name": "Garden"}
        assert len(event.entities_present) == 1
        assert event.entities_present[0]["name"] == "Andi"
        assert event.world_state is None

    def test_event_serialization(self):
        """Test MUDEvent JSON serialization."""
        now = datetime.now(timezone.utc)
        event = MUDEvent(
            event_type=EventType.EMOTE,
            actor="Andi",
            room_id="#123",
            timestamp=now,
        )

        json_data = event.model_dump(mode="json")
        assert json_data["event_type"] == "emote"
        assert json_data["actor"] == "Andi"
        assert "timestamp" in json_data

    def test_event_from_dict_empty_timestamp(self):
        """Test MUDEvent.from_dict with empty timestamp string."""
        data = {
            "type": "speech",
            "actor": "Prax",
            "room_id": "#123",
            "content": "Hello!",
            "timestamp": "",  # Empty string
        }
        event = MUDEvent.from_dict(data)

        # Should default to current time
        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)

    def test_event_from_dict_none_timestamp(self):
        """Test MUDEvent.from_dict with None timestamp."""
        data = {
            "type": "speech",
            "actor": "Prax",
            "room_id": "#123",
            "content": "Hello!",
            "timestamp": None,
        }
        event = MUDEvent.from_dict(data)

        # Should default to current time
        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)

    def test_event_from_dict_missing_timestamp(self):
        """Test MUDEvent.from_dict with missing timestamp field."""
        data = {
            "type": "speech",
            "actor": "Prax",
            "room_id": "#123",
            "content": "Hello!",
            # No timestamp field at all
        }
        event = MUDEvent.from_dict(data)

        # Should default to current time
        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)


class TestMUDAction:
    """Tests for MUDAction model."""

    def test_action_required_fields(self):
        """Test MUDAction with required field only."""
        action = MUDAction(tool="say")

        assert action.tool == "say"
        assert action.args == {}
        assert action.priority == 5

    def test_action_complete(self):
        """Test MUDAction with all fields."""
        action = MUDAction(
            tool="say",
            args={"message": "Hello!"},
            priority=1,
        )

        assert action.tool == "say"
        assert action.args == {"message": "Hello!"}
        assert action.priority == 1

    def test_action_to_command_say(self):
        """Test say action to command conversion."""
        action = MUDAction(tool="say", args={"message": "Hello, Papa!"})
        assert action.to_command() == "say Hello, Papa!"

    def test_action_to_command_emote(self):
        """Test emote action to command conversion."""
        action = MUDAction(tool="emote", args={"action": "smiles warmly"})
        assert action.to_command() == "emote smiles warmly"

    def test_action_to_command_whisper(self):
        """Test whisper action to command conversion."""
        action = MUDAction(
            tool="whisper",
            args={"target": "Prax", "message": "I love you"},
        )
        assert action.to_command() == "whisper Prax = I love you"

    def test_action_to_command_pose(self):
        """Test pose action to command conversion."""
        action = MUDAction(tool="pose", args={"action": "smiles softly"})
        assert action.to_command() == "pose smiles softly"

    def test_action_to_command_home(self):
        """Test home action to command conversion."""
        action = MUDAction(tool="home", args={})
        assert action.to_command() == "home"

    def test_action_to_command_setdesc(self):
        """Test setdesc action to command conversion."""
        action = MUDAction(tool="setdesc", args={"description": "A calm presence."})
        assert action.to_command() == "setdesc A calm presence."

    def test_action_to_command_move(self):
        """Test move action to command conversion."""
        action = MUDAction(tool="move", args={"location": "north"})
        assert action.to_command() == "north"

    def test_action_to_command_get(self):
        """Test get action to command conversion."""
        action = MUDAction(tool="get", args={"object": "key"})
        assert action.to_command() == "get key"

    def test_action_to_command_drop(self):
        """Test drop action to command conversion."""
        action = MUDAction(tool="drop", args={"object": "key"})
        assert action.to_command() == "drop key"

    def test_action_to_command_give(self):
        """Test give action to command conversion."""
        action = MUDAction(
            tool="give",
            args={"object": "flower", "target": "Prax"},
        )
        assert action.to_command() == "give flower = Prax"

    def test_action_to_command_use(self):
        """Test use action to command conversion."""
        action = MUDAction(tool="use", args={"object": "key"})
        assert action.to_command() == "use key"

    def test_action_to_command_speak(self):
        """Test speak action to command conversion."""
        action = MUDAction(tool="speak", args={"text": "Hello\\n\\nWorld"})
        assert action.to_command() == "act Hello\\n\\nWorld"

    def test_action_to_command_dig(self):
        """Test dig builder action to command conversion."""
        action = MUDAction(
            tool="dig",
            args={"room": "Library", "exits": "north,south"},
        )
        assert action.to_command() == "@dig Library = north,south"

    def test_action_to_command_create(self):
        """Test create builder action to command conversion."""
        action = MUDAction(tool="create", args={"object": "Golden Key"})
        assert action.to_command() == "@create Golden Key"

    def test_action_to_command_desc(self):
        """Test desc builder action to command conversion."""
        action = MUDAction(
            tool="desc",
            args={"target": "here", "description": "A bright room."},
        )
        assert action.to_command() == "@desc here = A bright room."

    def test_action_to_command_teleport(self):
        """Test teleport builder action to command conversion."""
        action = MUDAction(tool="teleport", args={"destination": "#123", "target": "me"})
        assert action.to_command() == "@teleport me = #123"


class TestMUDTurn:
    """Tests for MUDTurn model."""

    def test_turn_defaults(self):
        """Test MUDTurn with default values."""
        turn = MUDTurn()

        assert turn.events_received == []
        assert turn.room_context is None
        assert turn.entities_context == []
        assert turn.memories_retrieved == []
        assert turn.thinking == ""
        assert turn.actions_taken == []
        assert turn.doc_id is None
        assert isinstance(turn.timestamp, datetime)

    def test_turn_complete(self):
        """Test MUDTurn with all fields."""
        now = datetime.now(timezone.utc)
        room = RoomState(room_id="#123", name="Garden")
        entity = EntityState(entity_id="1", name="Prax")
        event = MUDEvent(
            event_type=EventType.SPEECH,
            actor="Prax",
            room_id="#123",
            content="Hello!",
        )
        action = MUDAction(tool="say", args={"message": "Hello!"})

        turn = MUDTurn(
            timestamp=now,
            events_received=[event],
            room_context=room,
            entities_context=[entity],
            memories_retrieved=[{"doc_id": "mem1", "content": "A memory"}],
            thinking="I should greet Prax back.",
            actions_taken=[action],
            doc_id="turn_123",
        )

        assert turn.timestamp == now
        assert len(turn.events_received) == 1
        assert turn.room_context.name == "Garden"
        assert len(turn.entities_context) == 1
        assert len(turn.memories_retrieved) == 1
        assert turn.thinking == "I should greet Prax back."
        assert len(turn.actions_taken) == 1
        assert turn.doc_id == "turn_123"

    def test_turn_serialization(self):
        """Test MUDTurn JSON serialization."""
        now = datetime.now(timezone.utc)
        turn = MUDTurn(timestamp=now, thinking="Test")

        json_data = turn.model_dump(mode="json")
        assert "timestamp" in json_data
        assert json_data["thinking"] == "Test"


class TestMUDSession:
    """Tests for MUDSession model."""

    def test_session_required_fields(self):
        """Test MUDSession with required fields only."""
        session = MUDSession(agent_id="andi", persona_id="andi")

        assert session.agent_id == "andi"
        assert session.persona_id == "andi"
        assert session.current_room is None
        assert session.entities_present == []
        assert session.pending_events == []
        assert session.recent_turns == []
        assert session.last_event_id == "0"
        assert session.last_action_time is None
        assert session.last_event_time is None

    def test_session_complete(self):
        """Test MUDSession with all fields."""
        now = datetime.now(timezone.utc)
        room = RoomState(room_id="#123", name="Garden")
        entity = EntityState(entity_id="1", name="Prax")
        event = MUDEvent(
            event_type=EventType.SPEECH,
            actor="Prax",
            room_id="#123",
        )
        turn = MUDTurn(thinking="Test turn")

        session = MUDSession(
            agent_id="andi",
            persona_id="andi",
            current_room=room,
            entities_present=[entity],
            pending_events=[event],
            recent_turns=[turn],
            last_event_id="1704096000000-0",
            last_action_time=now,
            last_event_time=now,
            created_at=now,
            updated_at=now,
        )

        assert session.current_room.name == "Garden"
        assert len(session.entities_present) == 1
        assert len(session.pending_events) == 1
        assert len(session.recent_turns) == 1
        assert session.last_event_id == "1704096000000-0"
        assert session.last_action_time == now
        assert session.last_event_time == now

    def test_session_add_turn(self):
        """Test MUDSession.add_turn method."""
        session = MUDSession(agent_id="andi", persona_id="andi")
        turn = MUDTurn(thinking="First turn")

        session.add_turn(turn)

        assert len(session.recent_turns) == 1
        assert session.recent_turns[0].thinking == "First turn"
        assert session.last_action_time is not None

    def test_session_get_last_turn(self):
        """Test MUDSession.get_last_turn method."""
        session = MUDSession(agent_id="andi", persona_id="andi")

        # Empty session returns None
        assert session.get_last_turn() is None

        # Add turns and get last
        turn1 = MUDTurn(thinking="First")
        turn2 = MUDTurn(thinking="Second")
        session.add_turn(turn1)
        session.add_turn(turn2)

        last = session.get_last_turn()
        assert last is not None
        assert last.thinking == "Second"

    def test_session_clear_pending_events(self):
        """Test MUDSession.clear_pending_events method."""
        event = MUDEvent(
            event_type=EventType.SPEECH,
            actor="Prax",
            room_id="#123",
        )
        session = MUDSession(
            agent_id="andi",
            persona_id="andi",
            pending_events=[event],
        )

        assert len(session.pending_events) == 1

        session.clear_pending_events()

        assert len(session.pending_events) == 0

    def test_session_serialization(self):
        """Test MUDSession JSON serialization."""
        now = datetime.now(timezone.utc)
        session = MUDSession(
            agent_id="andi",
            persona_id="andi",
            last_action_time=now,
            last_event_time=now,
            created_at=now,
            updated_at=now,
        )

        json_data = session.model_dump(mode="json")
        assert json_data["agent_id"] == "andi"
        assert "created_at" in json_data
        assert "last_action_time" in json_data
        assert "last_event_time" in json_data

    def test_session_serialization_none_datetime(self):
        """Test MUDSession serialization with None datetime."""
        session = MUDSession(agent_id="andi", persona_id="andi")

        json_data = session.model_dump(mode="json")
        assert json_data["last_action_time"] is None
        assert json_data["last_event_time"] is None
