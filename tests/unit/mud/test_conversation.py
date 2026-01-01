# tests/unit/mud/test_conversation.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for MUD conversation list manager."""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from aim.app.mud.conversation import (
    MUDConversationEntry,
    MUDConversationManager,
)
from aim.app.mud.session import MUDEvent, EventType
from aim_mud_types import MUDAction, WorldState, RoomState
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
