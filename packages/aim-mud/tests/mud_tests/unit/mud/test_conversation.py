# tests/unit/mud/test_conversation.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for MUD conversation list manager."""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from andimud_worker.conversation import MUDConversationManager
from aim_mud_types import (
    MUDConversationEntry,
    MUDEvent,
    EventType,
    MUDAction,
    WorldState,
    RoomState,
)
from aim.constants import DOC_MUD_WORLD, DOC_MUD_AGENT


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
    async def test_push_user_turn_formats_self_actions_first_person(self, conversation_manager, mock_redis):
        """Test that self-actions are formatted in first person."""
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
        self_movement = MUDEvent(
            event_id="2",
            event_type=EventType.MOVEMENT,
            actor="Andi",
            room_id="#124",
            room_name="The Kitchen",
            content="arrives from the garden",
            timestamp=datetime.now(timezone.utc),
            metadata={"is_self_action": True},
        )

        entry = await conversation_manager.push_user_turn(
            events=[regular_event, self_movement],
            room_id="#124",
            room_name="The Kitchen",
        )

        # Regular event should be in third person
        assert 'Prax says, "Hello there!"' in entry.content

        # Self-action should be in first person
        assert "You moved to The Kitchen" in entry.content

        # Should NOT contain third-person self-action
        assert "*You see Andi has arrived.*" not in entry.content

    @pytest.mark.asyncio
    async def test_push_user_turn_formats_self_object_actions(self, conversation_manager, mock_redis):
        """Test that self object actions are formatted in first person."""
        # Self-action: agent picked up an object
        self_pickup = MUDEvent(
            event_id="1",
            event_type=EventType.OBJECT,
            actor="Andi",
            room_id="#123",
            room_name="The Garden",
            content="picks up a flower",
            target="flower",
            timestamp=datetime.now(timezone.utc),
            metadata={"is_self_action": True},
        )

        entry = await conversation_manager.push_user_turn(
            events=[self_pickup],
            room_id="#123",
            room_name="The Garden",
        )

        # Self-action should be in first person
        assert "You picked up flower" in entry.content

        # Should NOT contain third-person format
        assert "*Andi picks up a flower*" not in entry.content


class TestMUDConversationManagerSelfSpeechFiltering:
    """Test MUDConversationManager self-speech echo filtering."""

    @pytest.mark.asyncio
    async def test_push_user_turn_filters_self_speech(self, conversation_manager, mock_redis):
        """Self-speech events should be filtered from conversation."""
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
        entry = await conversation_manager.push_user_turn(
            events=[self_speech, other_speech],
            room_id="#123",
            room_name="Test Room",
        )

        # Entry should only contain other_speech, not self_speech
        assert "Hi Andi!" in entry.content
        assert "Hello world" not in entry.content
        assert "You:" not in entry.content  # No "You: Hello world"

        # Should have event_count=1 (only the non-self-speech event)
        assert entry.metadata["event_count"] == 1

    @pytest.mark.asyncio
    async def test_push_user_turn_all_self_speech_filtered(self, conversation_manager, mock_redis):
        """When all events are self-speech, entry should say [No events]."""
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

        # Entry content should be [No events] since all were filtered
        assert entry.content == "[No events]"
        assert entry.metadata["event_count"] == 0

    @pytest.mark.asyncio
    async def test_push_user_turn_self_speech_mixed_with_other_events(self, conversation_manager, mock_redis):
        """Self-speech should be filtered but other event types from self preserved."""
        # Self-speech (should be filtered)
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

        # Self-emote (should NOT be filtered)
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

        # Other's speech (should NOT be filtered)
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

        entry = await conversation_manager.push_user_turn(
            events=[self_speech, self_emote, other_speech],
            room_id="#123",
            room_name="Test Room",
        )

        # Entry should contain emote and other's speech, but NOT self-speech
        assert "waves enthusiastically" in entry.content
        assert "Hi everyone!" in entry.content
        assert "Hello" not in entry.content
        assert "You: Hello" not in entry.content

        # Should have event_count=2 (emote + other's speech, not self-speech)
        assert entry.metadata["event_count"] == 2

    @pytest.mark.asyncio
    async def test_push_user_turn_self_speech_doesnt_affect_actor_metadata(self, conversation_manager, mock_redis):
        """Filtered self-speech should not appear in actors list."""
        # Self-speech (will be filtered)
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

        entry = await conversation_manager.push_user_turn(
            events=[self_speech, other_speech],
            room_id="#123",
            room_name="Test Room",
        )

        # Actors list should only contain Dave, not Andi (from filtered self-speech)
        assert "Dave" in entry.metadata["actors"]
        assert "Andi" not in entry.metadata["actors"]
        assert len(entry.metadata["actors"]) == 1

    @pytest.mark.asyncio
    async def test_push_user_turn_self_speech_doesnt_affect_event_ids(self, conversation_manager, mock_redis):
        """Filtered self-speech event IDs should not appear in metadata."""
        # Self-speech (will be filtered)
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

        entry = await conversation_manager.push_user_turn(
            events=[self_speech, other_event],
            room_id="#123",
            room_name="Test Room",
        )

        # Event IDs should only contain the other event, not self-speech
        assert "other-event-456" in entry.metadata["event_ids"]
        assert "self-speech-123" not in entry.metadata["event_ids"]
        assert len(entry.metadata["event_ids"]) == 1


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
