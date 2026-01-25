# tests/unit/mud/test_conversation.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for MUD conversation list manager."""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from andimud_worker.conversation import MUDConversationManager
from andimud_worker.adapter import format_self_action_guidance
from aim_mud_types import (
    MUDConversationEntry,
    MUDEvent,
    EventType,
    MUDAction,
    WorldState,
    RoomState,
)
from aim.constants import DOC_MUD_WORLD, DOC_MUD_AGENT, DOC_MUD_ACTION


def _sample_event(
    event_id: str = "1704096000000-0",
    actor: str = "Prax",
    content: str = "Hello there!",
) -> MUDEvent:
    """Create a sample MUDEvent for testing."""
    return MUDEvent(
        event_id=event_id,
        event_type=EventType.SPEECH,
        actor=actor,
        room_id="#123",
        room_name="The Garden",
        content=content,
        timestamp=datetime.now(timezone.utc),
    )


def _entries_from_rpush(mock_redis: AsyncMock) -> list[MUDConversationEntry]:
    entries: list[MUDConversationEntry] = []
    for call in mock_redis.rpush.call_args_list:
        entry_json = call.args[1]
        entries.append(MUDConversationEntry.model_validate_json(entry_json))
    return entries


def _sample_world_state() -> WorldState:
    """Create a sample WorldState for testing."""
    return WorldState(
        room_state=RoomState(
            room_id="#123",
            name="The Garden",
            description="A serene garden.",
            exits={"north": "#124"},
        ),
        entities_present=[],
    )


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    redis = AsyncMock()
    redis.lrange = AsyncMock(return_value=[])
    redis.rpush = AsyncMock(return_value=1)
    redis.lpop = AsyncMock(return_value=None)
    redis.lindex = AsyncMock(return_value=None)
    redis.lset = AsyncMock(return_value=True)
    redis.llen = AsyncMock(return_value=0)
    redis.delete = AsyncMock(return_value=1)

    # Mock pipeline for batch operations
    pipe = AsyncMock()
    pipe.delete = MagicMock(return_value=None)
    pipe.rpush = MagicMock(return_value=None)
    pipe.execute = AsyncMock(return_value=[1, 1, 1, 1])  # Mock return values for pipeline commands
    redis.pipeline = MagicMock(return_value=pipe)

    return redis


@pytest.fixture
def conversation_manager(mock_redis):
    """Create a MUDConversationManager with mocked Redis."""
    return MUDConversationManager(
        redis=mock_redis,
        agent_id="test_agent",
        persona_id="andi",
        max_tokens=1000,
    )


class TestMUDConversationEntry:
    """Test MUDConversationEntry model."""

    def test_entry_serialization(self):
        """Test that entries serialize and deserialize correctly."""
        entry = MUDConversationEntry(
            role="user",
            content="Test content",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv_123",
            sequence_no=0,
            metadata={"room_id": "#123"},
            speaker_id="world",
        )

        # Serialize
        json_str = entry.model_dump_json()
        assert "Test content" in json_str

        # Deserialize
        restored = MUDConversationEntry.model_validate_json(json_str)
        assert restored.role == "user"
        assert restored.content == "Test content"
        assert restored.tokens == 10
        assert restored.document_type == DOC_MUD_WORLD
        assert restored.sequence_no == 0
        assert restored.saved is False

    def test_entry_with_think_content(self):
        """Test assistant entry with think content."""
        entry = MUDConversationEntry(
            role="assistant",
            content="Hello!",
            tokens=15,
            document_type=DOC_MUD_AGENT,
            conversation_id="test_conv_123",
            sequence_no=1,
            think="I should respond warmly.",
            speaker_id="andi",
        )

        assert entry.think == "I should respond warmly."
        assert entry.role == "assistant"

    def test_entry_with_last_event_id(self):
        """Test that last_event_id field serializes and deserializes."""
        entry = MUDConversationEntry(
            role="user",
            content="Test content",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv_123",
            sequence_no=0,
            speaker_id="world",
            last_event_id="1704096000000-5",
        )

        # Serialize
        json_str = entry.model_dump_json()
        assert "1704096000000-5" in json_str

        # Deserialize
        restored = MUDConversationEntry.model_validate_json(json_str)
        assert restored.last_event_id == "1704096000000-5"

    def test_entry_last_event_id_defaults_to_none(self):
        """Test that last_event_id defaults to None."""
        entry = MUDConversationEntry(
            role="user",
            content="Test content",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv_123",
            sequence_no=0,
            speaker_id="world",
        )

        assert entry.last_event_id is None


class TestMUDConversationManagerInit:
    """Test MUDConversationManager initialization."""

    def test_init_sets_attributes(self, mock_redis):
        """Test that __init__ properly sets attributes."""
        manager = MUDConversationManager(
            redis=mock_redis,
            agent_id="test_agent",
            persona_id="andi",
            max_tokens=2000,
        )

        assert manager.agent_id == "test_agent"
        assert manager.persona_id == "andi"
        assert manager.max_tokens == 2000
        assert manager.key == "mud:agent:test_agent:conversation"
        assert manager._sequence_no == 0
        assert manager._conversation_id is None

    def test_get_conversation_id_creates_once(self, conversation_manager):
        """Test that conversation_id is created once and reused."""
        conv_id1 = conversation_manager._get_conversation_id()
        conv_id2 = conversation_manager._get_conversation_id()

        assert conv_id1 == conv_id2
        assert conv_id1.startswith("andimud_")
        parts = conv_id1.split("_")
        assert len(parts) == 3


class TestMUDConversationManagerPushUserTurn:
    """Test MUDConversationManager.push_user_turn method."""

    @pytest.mark.asyncio
    async def test_push_user_turn_with_events(self, conversation_manager, mock_redis):
        """Test pushing a user turn with events."""
        events = [_sample_event("1", "Prax", "Hello!")]

        entry = await conversation_manager.push_user_turn(
            events=events,
            room_id="#123",
            room_name="The Garden",
        )

        assert entry.role == "user"
        assert entry.document_type == DOC_MUD_WORLD
        # Content should be pure prose, no XML wrapper
        assert 'Prax says, "Hello!"' in entry.content
        assert "<events" not in entry.content
        assert "</events>" not in entry.content
        assert entry.tokens > 0
        assert entry.saved is False
        assert entry.speaker_id == "world"
        assert entry.metadata["event_count"] == 1
        assert "Prax" in entry.metadata["actors"]

        # Verify Redis was called
        mock_redis.rpush.assert_called_once()

    @pytest.mark.asyncio
    async def test_push_user_turn_empty_events(self, conversation_manager, mock_redis):
        """Test pushing a user turn with no events."""
        entry = await conversation_manager.push_user_turn(
            events=[],
            room_id="#123",
            room_name="The Garden",
        )

        assert entry.role == "user"
        # Empty events should show plain text marker, no XML
        assert entry.content == "[No events]"
        assert "<events" not in entry.content
        assert entry.metadata["event_count"] == 0

    @pytest.mark.asyncio
    async def test_push_user_turn_with_world_state(self, conversation_manager, mock_redis):
        """Test pushing a user turn using world_state for room info."""
        events = [_sample_event()]
        world_state = _sample_world_state()

        entry = await conversation_manager.push_user_turn(
            events=events,
            world_state=world_state,
        )

        assert entry.metadata["room_id"] == "#123"
        assert entry.metadata["room_name"] == "The Garden"

    @pytest.mark.asyncio
    async def test_push_user_turn_increments_sequence(self, conversation_manager, mock_redis):
        """Test that sequence numbers increment correctly."""
        entry1 = await conversation_manager.push_user_turn(events=[_sample_event("1")])
        entry2 = await conversation_manager.push_user_turn(events=[_sample_event("2")])

        assert entry1.sequence_no == 0
        assert entry2.sequence_no == 1
        assert entry1.conversation_id == entry2.conversation_id

    @pytest.mark.asyncio
    async def test_push_user_turn_stores_last_event_id_from_event(self, conversation_manager, mock_redis):
        """Test that push_user_turn stores last_event_id from the event's event_id."""
        # Event has event_id "1704096000000-5"
        events = [_sample_event("1704096000000-5", "Prax", "Hello!")]

        entry = await conversation_manager.push_user_turn(
            events=events,
            room_id="#123",
            room_name="The Garden",
            last_event_id="1704096000000-7",  # Fallback, should NOT be used
        )

        # Should use event's event_id, not the fallback
        assert entry.last_event_id == "1704096000000-5"

        # Verify the entry pushed to Redis has the last_event_id
        mock_redis.rpush.assert_called_once()
        pushed_json = mock_redis.rpush.call_args[0][1]
        pushed_entry = MUDConversationEntry.model_validate_json(pushed_json)
        assert pushed_entry.last_event_id == "1704096000000-5"

    @pytest.mark.asyncio
    async def test_push_user_turn_last_event_id_from_group_event(self, conversation_manager, mock_redis):
        """Test that push_user_turn gets last_event_id from group's last event."""
        events = [_sample_event("1704096000000-1", "Prax", "Hello!")]

        entry = await conversation_manager.push_user_turn(
            events=events,
            room_id="#123",
            room_name="The Garden",
            # No last_event_id fallback provided
        )

        # Should use the event's event_id
        assert entry.last_event_id == "1704096000000-1"

    @pytest.mark.asyncio
    async def test_push_user_turn_formats_self_actions_first_person(self, conversation_manager, mock_redis):
        """Test that self-actions preserve their pre-formatted guidance box content."""
        # Regular event from another actor
        regular_event = MUDEvent(
            event_id="1",
            event_type=EventType.SPEECH,
            actor="Prax",
            room_id="#123",
            room_name="The Garden",
            content="Hello there!",
            timestamp=datetime.now(timezone.utc),
        )

        # Self-action: agent moved to a new room
        # Create the event first to pass to the formatter
        self_movement_raw = MUDEvent(
            event_id="2",
            event_type=EventType.MOVEMENT,
            actor="Andi",
            room_id="#124",
            room_name="The Kitchen",
            content="arrives from the garden",  # This will be replaced
            timestamp=datetime.now(timezone.utc),
            metadata={"is_self_action": True},
        )

        # Self-actions arrive already formatted with guidance box
        self_action_guidance = format_self_action_guidance([self_movement_raw])

        # Now create the event with the pre-formatted content
        self_movement = MUDEvent(
            event_id="2",
            event_type=EventType.MOVEMENT,
            actor="Andi",
            room_id="#124",
            room_name="The Kitchen",
            content=self_action_guidance,
            timestamp=datetime.now(timezone.utc),
            metadata={"is_self_action": True},
        )

        entry = await conversation_manager.push_user_turn(
            events=[regular_event, self_movement],
            room_id="#124",
            room_name="The Kitchen",
        )

        entries = _entries_from_rpush(mock_redis)
        assert len(entries) == 2

        world_entry = entries[0]
        action_entry = entries[1]

        # Regular event should be in third person
        assert 'Prax says, "Hello there!"' in world_entry.content

        # Self-action guidance should be preserved as-is (new format)
        assert "Now you are at" in action_entry.content
        assert "[== Your Turn ==]" in action_entry.content

        # Should NOT contain third-person self-action
        assert "*You see Andi has arrived.*" not in action_entry.content

    @pytest.mark.asyncio
    async def test_push_user_turn_formats_self_object_actions(self, conversation_manager, mock_redis):
        """Test that self object actions preserve their pre-formatted guidance box content."""
        # Self-action: agent picked up an object
        # Create the event first to pass to the formatter
        self_pickup_raw = MUDEvent(
            event_id="1",
            event_type=EventType.OBJECT,
            actor="Andi",
            room_id="#123",
            room_name="The Garden",
            content="picks up a flower",  # This will be replaced
            target="flower",
            timestamp=datetime.now(timezone.utc),
            metadata={"is_self_action": True},
        )

        # Self-actions arrive already formatted with guidance box
        self_action_guidance = format_self_action_guidance([self_pickup_raw])

        # Now create the event with the pre-formatted content
        self_pickup = MUDEvent(
            event_id="1",
            event_type=EventType.OBJECT,
            actor="Andi",
            room_id="#123",
            room_name="The Garden",
            content=self_action_guidance,
            target="flower",
            timestamp=datetime.now(timezone.utc),
            metadata={"is_self_action": True},
        )

        entry = await conversation_manager.push_user_turn(
            events=[self_pickup],
            room_id="#123",
            room_name="The Garden",
        )

        # Self-action guidance should be preserved as-is (new format)
        assert "When you decided to pick up: flower" in entry.content
        assert "[== Your Turn ==]" in entry.content

        # Should NOT contain third-person format
        assert "*Andi picks up a flower*" not in entry.content


class TestMUDConversationManagerSelfSpeechRecording:
    """Test MUDConversationManager self-speech event recording (Phase 5).

    As of Phase 5, self-speech events are NO LONGER filtered. They are the
    canonical source of truth for agent speech, creating DOC_MUD_ACTION entries
    with first-person formatting.
    """

    @pytest.mark.asyncio
    async def test_push_user_turn_keeps_self_speech(self, conversation_manager, mock_redis):
        """Self-speech events are now kept and recorded as DOC_MUD_AGENT."""
        # Create a self-speech event and a regular event
        self_speech = MUDEvent(
            event_id="1",
            event_type=EventType.SPEECH,
            actor="Andi",
            actor_id="ai_andi",
            room_id="#123",
            room_name="Test Room",
            content="Hello world",
            timestamp=datetime.now(timezone.utc),
            metadata={"is_self_action": True}
        )

        other_speech = MUDEvent(
            event_id="2",
            event_type=EventType.SPEECH,
            actor="Bob",
            actor_id="player_bob",
            room_id="#123",
            room_name="Test Room",
            content="Hi Andi!",
            timestamp=datetime.now(timezone.utc),
            metadata={"is_self_action": False}
        )

        # Push both events
        await conversation_manager.push_user_turn(
            events=[self_speech, other_speech],
            room_id="#123",
            room_name="Test Room",
        )

        entries = _entries_from_rpush(mock_redis)

        # Should have 2 entries: one DOC_MUD_AGENT for self-speech, one DOC_MUD_WORLD for other
        assert len(entries) == 2

        agent_entries = [e for e in entries if e.document_type == DOC_MUD_AGENT]
        world_entries = [e for e in entries if e.document_type == DOC_MUD_WORLD]

        assert len(agent_entries) == 1
        assert len(world_entries) == 1

        # Self-speech should be in agent entry (first-person, raw content)
        assert "Hello world" in agent_entries[0].content
        # Other speech should be in world entry (third-person)
        assert "Hi Andi!" in world_entries[0].content

    @pytest.mark.asyncio
    async def test_push_user_turn_all_self_speech_creates_agent_entry(self, conversation_manager, mock_redis):
        """When all events are self-speech, they create DOC_MUD_AGENT entries."""
        # Create only self-speech events
        self_speech1 = MUDEvent(
            event_id="1",
            event_type=EventType.SPEECH,
            actor="Andi",
            actor_id="ai_andi",
            room_id="#123",
            room_name="Test Room",
            content="Hello world",
            timestamp=datetime.now(timezone.utc),
            metadata={"is_self_action": True}
        )

        self_speech2 = MUDEvent(
            event_id="2",
            event_type=EventType.SPEECH,
            actor="Andi",
            actor_id="ai_andi",
            room_id="#123",
            room_name="Test Room",
            content="How are you?",
            timestamp=datetime.now(timezone.utc),
            metadata={"is_self_action": True}
        )

        entry = await conversation_manager.push_user_turn(
            events=[self_speech1, self_speech2],
            room_id="#123",
            room_name="Test Room",
        )

        # Entry should be DOC_MUD_AGENT with both speeches (grouped by same actor)
        assert entry.document_type == DOC_MUD_AGENT
        assert "Hello world" in entry.content
        assert "How are you?" in entry.content
        assert entry.metadata["event_count"] == 2

    @pytest.mark.asyncio
    async def test_push_user_turn_self_speech_mixed_with_other_events(self, conversation_manager, mock_redis):
        """Self-speech and other events create separate entries by actor grouping."""
        # Self-speech
        self_speech = MUDEvent(
            event_id="1",
            event_type=EventType.SPEECH,
            actor="Andi",
            actor_id="ai_andi",
            room_id="#123",
            room_name="Test Room",
            content="Hello",
            timestamp=datetime.now(timezone.utc),
            metadata={"is_self_action": True}
        )

        # Self-emote (same actor, will be grouped with speech)
        self_emote = MUDEvent(
            event_id="2",
            event_type=EventType.EMOTE,
            actor="Andi",
            actor_id="ai_andi",
            room_id="#123",
            room_name="Test Room",
            content="waves enthusiastically",
            timestamp=datetime.now(timezone.utc),
            metadata={"is_self_action": True}
        )

        # Other's speech
        other_speech = MUDEvent(
            event_id="3",
            event_type=EventType.SPEECH,
            actor="Charlie",
            actor_id="player_charlie",
            room_id="#123",
            room_name="Test Room",
            content="Hi everyone!",
            timestamp=datetime.now(timezone.utc),
            metadata={"is_self_action": False}
        )

        await conversation_manager.push_user_turn(
            events=[self_speech, self_emote, other_speech],
            room_id="#123",
            room_name="Test Room",
        )

        entries = _entries_from_rpush(mock_redis)

        # Should have 2 entries: DOC_MUD_AGENT for self-speech group, DOC_MUD_WORLD for other
        assert len(entries) == 2

        agent_entries = [e for e in entries if e.document_type == DOC_MUD_AGENT]
        world_entries = [e for e in entries if e.document_type == DOC_MUD_WORLD]

        assert len(agent_entries) == 1
        assert len(world_entries) == 1

        agent_entry = agent_entries[0]
        world_entry = world_entries[0]

        # Self events should be in agent entry (first-person, raw content)
        assert "Hello" in agent_entry.content
        assert "waves enthusiastically" in agent_entry.content
        assert agent_entry.metadata["event_count"] == 2

        # Other speech in world entry (third-person)
        assert "Hi everyone!" in world_entry.content
        assert world_entry.metadata["event_count"] == 1

    @pytest.mark.asyncio
    async def test_push_user_turn_self_speech_appears_in_actor_metadata(self, conversation_manager, mock_redis):
        """Self-speech actor appears in metadata since events are no longer filtered."""
        # Self-speech
        self_speech = MUDEvent(
            event_id="1",
            event_type=EventType.SPEECH,
            actor="Andi",
            actor_id="ai_andi",
            room_id="#123",
            room_name="Test Room",
            content="Hello",
            timestamp=datetime.now(timezone.utc),
            metadata={"is_self_action": True}
        )

        # Other's speech
        other_speech = MUDEvent(
            event_id="2",
            event_type=EventType.SPEECH,
            actor="Dave",
            actor_id="player_dave",
            room_id="#123",
            room_name="Test Room",
            content="Hey there",
            timestamp=datetime.now(timezone.utc),
            metadata={"is_self_action": False}
        )

        await conversation_manager.push_user_turn(
            events=[self_speech, other_speech],
            room_id="#123",
            room_name="Test Room",
        )

        entries = _entries_from_rpush(mock_redis)

        # Find the self-speech entry (DOC_MUD_AGENT for speech)
        agent_entries = [e for e in entries if e.document_type == DOC_MUD_AGENT]
        assert len(agent_entries) == 1

        # Andi should appear in the agent entry's actors
        assert "Andi" in agent_entries[0].metadata["actors"]

    @pytest.mark.asyncio
    async def test_push_user_turn_self_speech_event_ids_in_metadata(self, conversation_manager, mock_redis):
        """Self-speech event IDs appear in metadata since events are no longer filtered."""
        # Self-speech
        self_speech = MUDEvent(
            event_id="self-speech-123",
            event_type=EventType.SPEECH,
            actor="Andi",
            actor_id="ai_andi",
            room_id="#123",
            room_name="Test Room",
            content="Hello",
            timestamp=datetime.now(timezone.utc),
            metadata={"is_self_action": True}
        )

        # Other event
        other_event = MUDEvent(
            event_id="other-event-456",
            event_type=EventType.EMOTE,
            actor="Eve",
            actor_id="player_eve",
            room_id="#123",
            room_name="Test Room",
            content="smiles",
            timestamp=datetime.now(timezone.utc),
            metadata={"is_self_action": False}
        )

        await conversation_manager.push_user_turn(
            events=[self_speech, other_event],
            room_id="#123",
            room_name="Test Room",
        )

        entries = _entries_from_rpush(mock_redis)

        # Find entries by type
        agent_entries = [e for e in entries if e.document_type == DOC_MUD_AGENT]
        world_entries = [e for e in entries if e.document_type == DOC_MUD_WORLD]

        # Self-speech event ID should be in agent entry
        assert "self-speech-123" in agent_entries[0].metadata["event_ids"]

        # Other event ID should be in world entry
        assert "other-event-456" in world_entries[0].metadata["event_ids"]


class TestMUDConversationManagerPushAssistantTurn:
    """Test MUDConversationManager.push_assistant_turn method."""

    @pytest.mark.asyncio
    async def test_push_assistant_turn(self, conversation_manager, mock_redis):
        """Test pushing an assistant turn."""
        actions = [MUDAction(tool="speak", args={"text": "Hello back!"})]

        entry = await conversation_manager.push_assistant_turn(
            content="Hello back!",
            think="I should greet them.",
            actions=actions,
        )

        assert entry.role == "assistant"
        assert entry.document_type == DOC_MUD_AGENT
        assert entry.content == "Hello back!"
        assert entry.think == "I should greet them."
        assert entry.tokens > 0
        assert entry.saved is False
        assert entry.speaker_id == "andi"
        assert entry.metadata["action_count"] == 1

        mock_redis.rpush.assert_called_once()

    @pytest.mark.asyncio
    async def test_push_assistant_turn_no_think(self, conversation_manager, mock_redis):
        """Test pushing an assistant turn without think content."""
        entry = await conversation_manager.push_assistant_turn(
            content="A simple response.",
            actions=[],
        )

        assert entry.think is None
        assert entry.content == "A simple response."

    @pytest.mark.asyncio
    async def test_push_assistant_turn_counts_think_tokens(self, conversation_manager, mock_redis):
        """Test that think content is included in token count."""
        think_content = "This is a long thinking process " * 20

        entry = await conversation_manager.push_assistant_turn(
            content="Short response.",
            think=think_content,
            actions=[],
        )

        # Token count should be higher due to think content
        assert entry.tokens > 10


class TestMUDConversationManagerGetHistory:
    """Test MUDConversationManager.get_history method."""

    @pytest.mark.asyncio
    async def test_get_history_empty(self, conversation_manager, mock_redis):
        """Test getting history from empty list."""
        mock_redis.lrange.return_value = []

        history = await conversation_manager.get_history(token_budget=1000)

        assert history == []

    @pytest.mark.asyncio
    async def test_get_history_respects_token_budget(self, conversation_manager, mock_redis):
        """Test that get_history respects token budget."""
        # Create entries with known token counts
        entry1 = MUDConversationEntry(
            role="user",
            content="First message",
            tokens=50,
            document_type=DOC_MUD_WORLD,
            conversation_id="test",
            sequence_no=0,
            speaker_id="world",
        )
        entry2 = MUDConversationEntry(
            role="assistant",
            content="Second message",
            tokens=50,
            document_type=DOC_MUD_AGENT,
            conversation_id="test",
            sequence_no=1,
            speaker_id="andi",
        )
        entry3 = MUDConversationEntry(
            role="user",
            content="Third message",
            tokens=50,
            document_type=DOC_MUD_WORLD,
            conversation_id="test",
            sequence_no=2,
            speaker_id="world",
        )

        mock_redis.lrange.return_value = [
            entry1.model_dump_json().encode(),
            entry2.model_dump_json().encode(),
            entry3.model_dump_json().encode(),
        ]

        # Budget of 75 should only get the newest entry
        history = await conversation_manager.get_history(token_budget=75)
        assert len(history) == 1
        assert history[0].sequence_no == 2

        # Budget of 125 should get the two newest entries
        history = await conversation_manager.get_history(token_budget=125)
        assert len(history) == 2
        assert history[0].sequence_no == 1
        assert history[1].sequence_no == 2

        # Budget of 200 should get all entries
        history = await conversation_manager.get_history(token_budget=200)
        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_get_history_returns_chronological_order(self, conversation_manager, mock_redis):
        """Test that history is returned in chronological order."""
        entries = [
            MUDConversationEntry(
                role="user",
                content=f"Message {i}",
                tokens=10,
                document_type=DOC_MUD_WORLD,
                conversation_id="test",
                sequence_no=i,
                speaker_id="world",
            )
            for i in range(5)
        ]

        mock_redis.lrange.return_value = [
            e.model_dump_json().encode() for e in entries
        ]

        history = await conversation_manager.get_history(token_budget=1000)

        assert len(history) == 5
        for i, entry in enumerate(history):
            assert entry.sequence_no == i


class TestMUDConversationManagerFlushToCVM:
    """Test MUDConversationManager.flush_to_cvm method."""

    @pytest.mark.asyncio
    async def test_flush_to_cvm_marks_entries_saved(self, conversation_manager, mock_redis):
        """Test that flush_to_cvm marks entries as saved."""
        entry = MUDConversationEntry(
            role="user",
            content="Test message",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test",
            sequence_no=0,
            speaker_id="world",
            saved=False,
        )

        mock_redis.lrange.return_value = [entry.model_dump_json().encode()]

        mock_cvm = MagicMock()
        mock_cvm.insert.return_value = "doc-123"

        count = await conversation_manager.flush_to_cvm(mock_cvm)

        assert count == 1
        mock_cvm.insert.assert_called_once()

        # Verify LSET was called to update the entry
        mock_redis.lset.assert_called_once()
        call_args = mock_redis.lset.call_args
        assert call_args[0][0] == conversation_manager.key
        assert call_args[0][1] == 0

        # Verify the updated entry has saved=True
        updated_json = call_args[0][2]
        updated_entry = MUDConversationEntry.model_validate_json(updated_json)
        assert updated_entry.saved is True
        assert updated_entry.doc_id == "doc-123"

    @pytest.mark.asyncio
    async def test_flush_to_cvm_skips_already_saved(self, conversation_manager, mock_redis):
        """Test that flush_to_cvm skips already saved entries."""
        entry = MUDConversationEntry(
            role="user",
            content="Already saved",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test",
            sequence_no=0,
            speaker_id="world",
            saved=True,
            doc_id="existing-doc",
        )

        mock_redis.lrange.return_value = [entry.model_dump_json().encode()]

        mock_cvm = MagicMock()

        count = await conversation_manager.flush_to_cvm(mock_cvm)

        assert count == 0
        mock_cvm.insert.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_to_cvm_skips_skip_save_entries(self, conversation_manager, mock_redis):
        """Test that skip_save entries are marked saved but not persisted."""
        entry = MUDConversationEntry(
            role="user",
            content="Seeded content",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test",
            sequence_no=0,
            speaker_id="world",
            saved=False,
            skip_save=True,
        )

        mock_redis.lrange.return_value = [entry.model_dump_json().encode()]

        mock_cvm = MagicMock()

        count = await conversation_manager.flush_to_cvm(mock_cvm)

        assert count == 0
        mock_cvm.insert.assert_not_called()

        mock_redis.lset.assert_called_once()
        updated_json = mock_redis.lset.call_args[0][2]
        updated_entry = MUDConversationEntry.model_validate_json(updated_json)
        assert updated_entry.saved is True
        assert updated_entry.skip_save is True
        assert updated_entry.doc_id is None

    @pytest.mark.asyncio
    async def test_flush_to_cvm_empty_list(self, conversation_manager, mock_redis):
        """Test flush_to_cvm with empty list."""
        mock_redis.lrange.return_value = []

        mock_cvm = MagicMock()

        count = await conversation_manager.flush_to_cvm(mock_cvm)

        assert count == 0

    @pytest.mark.asyncio
    async def test_flush_to_cvm_creates_correct_message(self, conversation_manager, mock_redis):
        """Test that flush_to_cvm creates correct ConversationMessage."""
        entry = MUDConversationEntry(
            role="assistant",
            content="Test response",
            tokens=10,
            document_type=DOC_MUD_AGENT,
            conversation_id="test_conv",
            sequence_no=1,
            speaker_id="andi",
            think="Some thinking",
            metadata={"actions": ["act: Hello"]},
        )

        mock_redis.lrange.return_value = [entry.model_dump_json().encode()]

        mock_cvm = MagicMock()
        mock_cvm.insert.return_value = "doc-456"

        await conversation_manager.flush_to_cvm(mock_cvm)

        # Verify the ConversationMessage was created correctly
        call_args = mock_cvm.insert.call_args[0][0]
        assert call_args.role == "assistant"
        assert call_args.content == "Test response"
        assert call_args.document_type == DOC_MUD_AGENT
        assert call_args.conversation_id == "test_conv"
        assert call_args.sequence_no == 1
        assert call_args.speaker_id == "andi"
        assert call_args.think == "Some thinking"


class TestMUDConversationManagerAutoTrim:
    """Test MUDConversationManager auto-trim functionality."""

    @pytest.mark.asyncio
    async def test_auto_trim_removes_old_saved_entries(self, mock_redis):
        """Test that auto-trim removes old saved entries when over budget."""
        manager = MUDConversationManager(
            redis=mock_redis,
            agent_id="test",
            persona_id="andi",
            max_tokens=100,  # Low budget
        )

        # Create a saved entry that will be trimmed
        old_saved = MUDConversationEntry(
            role="user",
            content="Old message",
            tokens=60,
            document_type=DOC_MUD_WORLD,
            conversation_id="test",
            sequence_no=0,
            speaker_id="world",
            saved=True,
            doc_id="old-doc",
        )

        # New entry that will be pushed (will have ~50 tokens)
        new_entry = MUDConversationEntry(
            role="user",
            content="New message with events",
            tokens=60,
            document_type=DOC_MUD_WORLD,
            conversation_id="test",
            sequence_no=1,
            speaker_id="world",
        )

        # Track whether we've popped
        popped = [False]

        # Simulate list state: after push we have both entries (120 tokens > 100 budget)
        # After lpop we have only the new entry (60 tokens < 100)
        async def lrange_effect(*args):
            if popped[0]:
                return [new_entry.model_dump_json().encode()]
            return [
                old_saved.model_dump_json().encode(),
                new_entry.model_dump_json().encode(),
            ]

        async def lindex_effect(key, idx):
            if idx == 0:
                return old_saved.model_dump_json().encode()
            return None

        async def lpop_effect(key):
            popped[0] = True
            return old_saved.model_dump_json().encode()

        mock_redis.lrange.side_effect = lrange_effect
        mock_redis.lindex.side_effect = lindex_effect
        mock_redis.lpop.side_effect = lpop_effect

        # Push a new entry that exceeds budget
        events = [_sample_event()]
        await manager.push_user_turn(events=events)

        # Verify lpop was called (old entry was trimmed)
        assert mock_redis.lpop.called

    @pytest.mark.asyncio
    async def test_auto_trim_never_removes_unsaved_entries(self, mock_redis):
        """Test that auto-trim never removes unsaved entries."""
        manager = MUDConversationManager(
            redis=mock_redis,
            agent_id="test",
            persona_id="andi",
            max_tokens=50,  # Very low budget
        )

        # Create an unsaved entry
        old_unsaved = MUDConversationEntry(
            role="user",
            content="Unsaved message",
            tokens=60,
            document_type=DOC_MUD_WORLD,
            conversation_id="test",
            sequence_no=0,
            speaker_id="world",
            saved=False,  # NOT saved
        )

        async def lindex_effect(key, idx):
            if idx == 0:
                return old_unsaved.model_dump_json().encode()
            return None

        async def lrange_effect(*args):
            # Return high token count to trigger trim attempt
            return [old_unsaved.model_dump_json().encode()]

        mock_redis.lrange.side_effect = lrange_effect
        mock_redis.lindex.side_effect = lindex_effect

        # Push a new entry
        events = [_sample_event()]
        await manager.push_user_turn(events=events)

        # lpop should NOT be called for unsaved entries
        mock_redis.lpop.assert_not_called()


class TestMUDConversationManagerClear:
    """Test MUDConversationManager.clear method."""

    @pytest.mark.asyncio
    async def test_clear_deletes_key(self, conversation_manager, mock_redis):
        """Test that clear deletes the Redis key."""
        # Set some state
        conversation_manager._sequence_no = 5
        conversation_manager._conversation_id = "test_conv"

        await conversation_manager.clear()

        mock_redis.delete.assert_called_once_with(conversation_manager.key)
        assert conversation_manager._sequence_no == 0
        assert conversation_manager._conversation_id is None


class TestMUDConversationManagerGetTotalTokens:
    """Test MUDConversationManager.get_total_tokens method."""

    @pytest.mark.asyncio
    async def test_get_total_tokens(self, conversation_manager, mock_redis):
        """Test getting total tokens across all entries."""
        entries = [
            MUDConversationEntry(
                role="user",
                content=f"Message {i}",
                tokens=10 * (i + 1),  # 10, 20, 30
                document_type=DOC_MUD_WORLD,
                conversation_id="test",
                sequence_no=i,
                speaker_id="world",
            )
            for i in range(3)
        ]

        mock_redis.lrange.return_value = [
            e.model_dump_json().encode() for e in entries
        ]

        total = await conversation_manager.get_total_tokens()

        assert total == 60  # 10 + 20 + 30

    @pytest.mark.asyncio
    async def test_get_total_tokens_empty(self, conversation_manager, mock_redis):
        """Test getting total tokens from empty list."""
        mock_redis.lrange.return_value = []

        total = await conversation_manager.get_total_tokens()

        assert total == 0


class TestMUDConversationManagerConversationID:
    """Test MUDConversationManager conversation_id management methods."""

    def test_get_current_conversation_id_before_creation(self, conversation_manager):
        """Test that get_current_conversation_id returns None before creation."""
        conv_id = conversation_manager.get_current_conversation_id()
        assert conv_id is None

    def test_get_current_conversation_id_after_creation(self, conversation_manager):
        """Test that get_current_conversation_id returns ID after creation."""
        # Create a conversation_id by calling _get_conversation_id
        created_id = conversation_manager._get_conversation_id()

        # get_current_conversation_id should return the same ID
        current_id = conversation_manager.get_current_conversation_id()
        assert current_id == created_id
        assert current_id.startswith("andimud_")

    def test_set_conversation_id(self, conversation_manager):
        """Test that set_conversation_id updates the instance variable."""
        new_id = "andimud_test_12345_abcdef"

        conversation_manager.set_conversation_id(new_id)

        current_id = conversation_manager.get_current_conversation_id()
        assert current_id == new_id

    @pytest.mark.asyncio
    async def test_retag_unsaved_entries(self, conversation_manager, mock_redis):
        """Test that retag_unsaved_entries updates unsaved entries."""
        # Create a mix of saved and unsaved entries
        entry1 = MUDConversationEntry(
            role="user",
            content="Unsaved message 1",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="old_conv_123",
            sequence_no=0,
            speaker_id="world",
            saved=False,
        )
        entry2 = MUDConversationEntry(
            role="assistant",
            content="Saved message",
            tokens=10,
            document_type=DOC_MUD_AGENT,
            conversation_id="old_conv_123",
            sequence_no=1,
            speaker_id="andi",
            saved=True,
            doc_id="doc-123",
        )
        entry3 = MUDConversationEntry(
            role="user",
            content="Unsaved message 2",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="old_conv_123",
            sequence_no=2,
            speaker_id="world",
            saved=False,
        )

        mock_redis.lrange.return_value = [
            entry1.model_dump_json().encode(),
            entry2.model_dump_json().encode(),
            entry3.model_dump_json().encode(),
        ]

        new_conv_id = "new_conv_456"
        retagged = await conversation_manager.retag_unsaved_entries(new_conv_id)

        # Should have retagged 2 unsaved entries
        assert retagged == 2

        # Verify pipeline operations: delete + 3 rpush
        assert mock_redis.pipeline.called
        pipe = mock_redis.pipeline.return_value
        pipe.delete.assert_called_once()
        assert pipe.rpush.call_count == 3

        # Verify sequence counter was updated
        assert conversation_manager._sequence_no == 2  # Two unsaved entries renumbered from 0

    @pytest.mark.asyncio
    async def test_retag_only_unsaved_entries(self, conversation_manager, mock_redis):
        """Test that retag_unsaved_entries preserves saved entries unchanged."""
        # Create entries with saved entry in the middle
        entry1 = MUDConversationEntry(
            role="user",
            content="Unsaved 1",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="old_conv",
            sequence_no=0,
            speaker_id="world",
            saved=False,
        )
        entry2 = MUDConversationEntry(
            role="assistant",
            content="Saved in CVM",
            tokens=10,
            document_type=DOC_MUD_AGENT,
            conversation_id="old_conv",
            sequence_no=1,
            speaker_id="andi",
            saved=True,
            doc_id="doc-saved",
        )
        entry3 = MUDConversationEntry(
            role="user",
            content="Unsaved 2",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="old_conv",
            sequence_no=2,
            speaker_id="world",
            saved=False,
        )

        mock_redis.lrange.return_value = [
            entry1.model_dump_json().encode(),
            entry2.model_dump_json().encode(),
            entry3.model_dump_json().encode(),
        ]

        new_conv_id = "new_conv_xyz"
        retagged = await conversation_manager.retag_unsaved_entries(new_conv_id)

        assert retagged == 2

        # Capture the rpush calls to verify entry contents
        pipe = mock_redis.pipeline.return_value
        rpush_calls = pipe.rpush.call_args_list

        # Parse the updated entries
        updated_entries = []
        for call in rpush_calls:
            entry_json = call[0][1]  # Second argument to rpush
            updated_entries.append(MUDConversationEntry.model_validate_json(entry_json))

        # First unsaved entry should be renumbered to 0 with new conv_id
        assert updated_entries[0].conversation_id == new_conv_id
        assert updated_entries[0].sequence_no == 0
        assert updated_entries[0].saved is False

        # Saved entry should be unchanged
        assert updated_entries[1].conversation_id == "old_conv"
        assert updated_entries[1].sequence_no == 1
        assert updated_entries[1].saved is True
        assert updated_entries[1].doc_id == "doc-saved"

        # Second unsaved entry should be renumbered to 1 with new conv_id
        assert updated_entries[2].conversation_id == new_conv_id
        assert updated_entries[2].sequence_no == 1
        assert updated_entries[2].saved is False

    @pytest.mark.asyncio
    async def test_retag_unsaved_entries_empty_list(self, conversation_manager, mock_redis):
        """Test retag_unsaved_entries with empty list."""
        mock_redis.lrange.return_value = []

        retagged = await conversation_manager.retag_unsaved_entries("new_conv")

        assert retagged == 0
        # Pipeline should not be used
        mock_redis.pipeline.assert_not_called()


class TestMUDConversationManagerGetLastEventId:
    """Test MUDConversationManager.get_last_event_id method.

    The method now scans ALL entries and returns the MAX event_id across all entries.
    Event IDs are Redis stream IDs in format "timestamp-sequence" which can be
    compared lexicographically.
    """

    @pytest.mark.asyncio
    async def test_get_last_event_id_returns_max_across_all_entries(self, conversation_manager, mock_redis):
        """Test that get_last_event_id returns the maximum event_id across all entries."""
        entry1 = MUDConversationEntry(
            role="user",
            content="First message",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test",
            sequence_no=0,
            speaker_id="world",
            last_event_id="1704096000000-10",
        )
        entry2 = MUDConversationEntry(
            role="user",
            content="Second message",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test",
            sequence_no=1,
            speaker_id="world",
            last_event_id="1704096000000-42",  # Max
        )
        entry3 = MUDConversationEntry(
            role="user",
            content="Third message",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test",
            sequence_no=2,
            speaker_id="world",
            last_event_id="1704096000000-5",  # Lower than entry2
        )

        mock_redis.lrange.return_value = [
            entry1.model_dump_json().encode(),
            entry2.model_dump_json().encode(),
            entry3.model_dump_json().encode(),
        ]

        result = await conversation_manager.get_last_event_id()

        # Should return the MAX event_id (1704096000000-42), not the most recent entry's
        assert result == "1704096000000-42"

    @pytest.mark.asyncio
    async def test_get_last_event_id_returns_none_for_empty_list(self, conversation_manager, mock_redis):
        """Test that get_last_event_id returns None when conversation is empty."""
        mock_redis.lrange.return_value = []

        result = await conversation_manager.get_last_event_id()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_last_event_id_returns_none_when_all_entries_have_no_last_event_id(self, conversation_manager, mock_redis):
        """Test that get_last_event_id returns None when no entries have last_event_id."""
        entry = MUDConversationEntry(
            role="user",
            content="Test message",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test",
            sequence_no=0,
            speaker_id="world",
            last_event_id=None,
        )

        mock_redis.lrange.return_value = [entry.model_dump_json().encode()]

        result = await conversation_manager.get_last_event_id()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_last_event_id_handles_parse_error_gracefully(self, conversation_manager, mock_redis):
        """Test that get_last_event_id skips invalid entries and returns max from valid ones."""
        entry = MUDConversationEntry(
            role="user",
            content="Valid entry",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test",
            sequence_no=0,
            speaker_id="world",
            last_event_id="1704096000000-99",
        )

        mock_redis.lrange.return_value = [
            b"invalid json",
            entry.model_dump_json().encode(),
        ]

        result = await conversation_manager.get_last_event_id()

        # Should return the max from valid entries
        assert result == "1704096000000-99"

    @pytest.mark.asyncio
    async def test_get_last_event_id_handles_bytes_decoding(self, conversation_manager, mock_redis):
        """Test that get_last_event_id correctly handles bytes from Redis."""
        entry = MUDConversationEntry(
            role="user",
            content="Test message",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test",
            sequence_no=0,
            speaker_id="world",
            last_event_id="1704096000000-99",
        )

        # Return as bytes (as Redis actually does)
        mock_redis.lrange.return_value = [entry.model_dump_json().encode("utf-8")]

        result = await conversation_manager.get_last_event_id()

        assert result == "1704096000000-99"
