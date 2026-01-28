# packages/aim-mud/tests/mud_tests/unit/mediator/test_compiler.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for the CompilerMixin.

Tests event compilation, event grouping, and conversation entry creation.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from andimud_mediator.mixins.compiler import CompilerMixin, PendingEventBatch
from andimud_mediator.config import MediatorConfig
from aim_mud_types import EventType, MUDEvent, MUDConversationEntry
from aim_mud_types.helper import _utc_now
from aim.constants import DOC_MUD_WORLD, DOC_MUD_AGENT, DOC_CODE_ACTION, DOC_CODE_FILE


class MockMediatorWithCompiler(CompilerMixin):
    """Test harness that provides the service interface for CompilerMixin."""

    def __init__(self, mock_redis, config):
        self.redis = mock_redis
        self.config = config
        self._init_compiler()


@pytest.fixture
def mediator_config():
    """Create a test mediator configuration."""
    return MediatorConfig(
        redis_url="redis://localhost:6379",
        embedding_model="test-model",
        embedding_device="cpu",
    )


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    redis = AsyncMock()

    # Storage for simulating Redis operations
    redis_hashes = {}
    redis_lists = {}
    redis_sequence = {"mud:sequence": 0}

    async def mock_hgetall(key):
        return redis_hashes.get(key, {})

    async def mock_hset(key, mapping=None, **kwargs):
        if key not in redis_hashes:
            redis_hashes[key] = {}
        if mapping:
            redis_hashes[key].update(mapping)
        return 1

    async def mock_rpush(key, value):
        if key not in redis_lists:
            redis_lists[key] = []
        redis_lists[key].append(value)
        return len(redis_lists[key])

    async def mock_llen(key):
        return len(redis_lists.get(key, []))

    async def mock_incr(key):
        redis_sequence[key] = redis_sequence.get(key, 0) + 1
        return redis_sequence[key]

    async def mock_eval(script, num_keys, *args):
        return 1  # CAS success

    redis.hgetall = AsyncMock(side_effect=mock_hgetall)
    redis.hset = AsyncMock(side_effect=mock_hset)
    redis.rpush = AsyncMock(side_effect=mock_rpush)
    redis.llen = AsyncMock(side_effect=mock_llen)
    redis.incr = AsyncMock(side_effect=mock_incr)
    redis.eval = AsyncMock(side_effect=mock_eval)

    # Store references for assertions
    redis._hashes = redis_hashes
    redis._lists = redis_lists

    return redis


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    model = MagicMock()
    # Return a fixed embedding vector
    model.return_value = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    return model


def make_event(
    event_type: EventType = EventType.SPEECH,
    actor: str = "Player",
    actor_id: str = "player_1",
    content: str = "Hello",
    room_id: str = "room_1",
    room_name: str = "Test Room",
    event_id: str = "event_1",
) -> MUDEvent:
    """Create a test MUDEvent."""
    return MUDEvent(
        event_id=event_id,
        event_type=event_type,
        actor=actor,
        actor_id=actor_id,
        content=content,
        room_id=room_id,
        room_name=room_name,
        timestamp=_utc_now(),
    )


class TestPendingEventBatch:
    """Test PendingEventBatch dataclass."""

    def test_batch_initialization(self):
        """Batch should initialize with empty collections."""
        batch = PendingEventBatch(room_id="room_1")
        assert batch.room_id == "room_1"
        assert batch.events == []
        assert batch.observer_agents == set()
        assert batch.self_action_events == []

    def test_batch_with_events(self):
        """Batch should store events, observers, and self-actions."""
        event = make_event()
        batch = PendingEventBatch(
            room_id="room_1",
            events=[event],
            observer_agents={"andi", "nova"},
            self_action_events=[("prax", event)],
        )
        assert len(batch.events) == 1
        assert batch.events[0] == event
        assert "andi" in batch.observer_agents
        assert "nova" in batch.observer_agents
        assert len(batch.self_action_events) == 1


class TestCompilerMixinInit:
    """Test compiler initialization."""

    def test_init_compiler(self, mock_redis, mediator_config):
        """_init_compiler should initialize state."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)
        assert mediator._pending_batches == {}
        assert mediator._embedding_model is None


class TestQueueEventForCompilation:
    """Test event queueing and immediate compilation."""

    @pytest.mark.asyncio
    async def test_queue_event_compiles_immediately(self, mock_redis, mediator_config):
        """Event should be compiled immediately."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)
        event = make_event()

        compile_called = []

        async def mock_compile(room_id):
            compile_called.append(room_id)

        with patch.object(mediator, "_compile_batch", side_effect=mock_compile):
            await mediator.queue_event_for_compilation(
                event, observer_agent_ids=["andi"], self_action_agent_id=None
            )

        # Should have compiled immediately
        assert "room_1" in compile_called

    @pytest.mark.asyncio
    async def test_queue_event_with_self_action(self, mock_redis, mediator_config):
        """Event with self-action should track separately."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)
        event = make_event()

        # Track what gets compiled
        compiled_batches = []

        original_compile = mediator._compile_batch

        async def capture_compile(room_id):
            batch = mediator._pending_batches.get(room_id)
            if batch:
                compiled_batches.append(batch)
            await original_compile(room_id)

        with patch.object(mediator, "_compile_batch", side_effect=capture_compile):
            with patch.object(mediator, "_compute_embedding", return_value="test_embedding"):
                with patch.object(mediator, "_get_agent_conversation_id", return_value="conv_1"):
                    await mediator.queue_event_for_compilation(
                        event, observer_agent_ids=["nova"], self_action_agent_id="andi"
                    )

        assert len(compiled_batches) == 1
        batch = compiled_batches[0]
        assert "nova" in batch.observer_agents
        assert "andi" not in batch.observer_agents
        assert len(batch.self_action_events) == 1
        assert batch.self_action_events[0] == ("andi", event)

    @pytest.mark.asyncio
    async def test_queue_events_different_rooms(self, mock_redis, mediator_config):
        """Events in different rooms should compile separately."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        event1 = make_event(room_id="room_1", event_id="e1")
        event2 = make_event(room_id="room_2", event_id="e2")

        compile_called = []

        async def mock_compile(room_id):
            compile_called.append(room_id)

        with patch.object(mediator, "_compile_batch", side_effect=mock_compile):
            await mediator.queue_event_for_compilation(
                event1, observer_agent_ids=["andi"], self_action_agent_id=None
            )
            await mediator.queue_event_for_compilation(
                event2, observer_agent_ids=["nova"], self_action_agent_id=None
            )

        assert "room_1" in compile_called
        assert "room_2" in compile_called


class TestEventGroupingThirdPerson:
    """Test event grouping logic for third-person (observers)."""

    @pytest.mark.asyncio
    async def test_consecutive_events_same_actor_grouped(
        self, mock_redis, mediator_config, mock_embedding_model
    ):
        """Consecutive events from same actor should be grouped."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        events = [
            make_event(actor="Player", actor_id="p1", content="Hello", event_id="e1"),
            make_event(actor="Player", actor_id="p1", content="How are you?", event_id="e2"),
        ]

        with patch.object(mediator, "_compute_embedding", return_value="test_embedding"):
            entries = await mediator._compile_events_to_entries_third_person(events)

        # Should produce one entry (grouped)
        assert len(entries) == 1
        assert "Hello" in entries[0].content
        assert "How are you?" in entries[0].content

    @pytest.mark.asyncio
    async def test_different_actors_create_separate_entries(
        self, mock_redis, mediator_config, mock_embedding_model
    ):
        """Events from different actors should create separate entries."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        events = [
            make_event(actor="Alice", actor_id="a1", content="Hello", event_id="e1"),
            make_event(actor="Bob", actor_id="b1", content="Hi there", event_id="e2"),
        ]

        with patch.object(mediator, "_compute_embedding", return_value="test_embedding"):
            entries = await mediator._compile_events_to_entries_third_person(events)

        # Should produce two entries (different actors)
        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_code_events_always_separate(
        self, mock_redis, mediator_config, mock_embedding_model
    ):
        """Code events should always get their own entry."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        events = [
            make_event(actor="Player", actor_id="p1", content="Hello", event_id="e1"),
            make_event(
                event_type=EventType.CODE_ACTION,
                actor="System",
                actor_id="sys",
                content="Output from code",
                event_id="e2",
            ),
            make_event(actor="Player", actor_id="p1", content="Goodbye", event_id="e3"),
        ]

        with patch.object(mediator, "_compute_embedding", return_value="test_embedding"):
            entries = await mediator._compile_events_to_entries_third_person(events)

        # Should produce three entries (code event always separate)
        assert len(entries) == 3


class TestEventGroupingFirstPerson:
    """Test event grouping logic for first-person (self-action)."""

    @pytest.mark.asyncio
    async def test_first_person_consecutive_events_grouped(
        self, mock_redis, mediator_config
    ):
        """Consecutive self-action events should be grouped."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        events = [
            make_event(actor="Andi", actor_id="andi_1", content="I say hello", event_id="e1"),
            make_event(actor="Andi", actor_id="andi_1", content="I wave", event_id="e2"),
        ]

        with patch.object(mediator, "_compute_embedding", return_value="test_embedding"):
            with patch.object(mediator, "_get_agent_conversation_id", return_value="conv_1"):
                entries = await mediator._compile_events_to_entries_first_person(events, "andi")

        # Should produce one entry (grouped)
        assert len(entries) == 1
        assert entries[0].document_type == DOC_MUD_AGENT
        assert entries[0].speaker_id == "andi"

    @pytest.mark.asyncio
    async def test_first_person_entry_has_your_turn_banner(
        self, mock_redis, mediator_config
    ):
        """First-person entries should use format_self_action_guidance with visual banner."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        events = [
            make_event(
                event_type=EventType.MOVEMENT,
                actor="Andi",
                actor_id="andi_1",
                content="Andi moves south",
                event_id="e1",
            ),
        ]
        # Add metadata for movement formatting
        events[0].metadata = {
            "source_room_name": "Kitchen",
            "destination_room_name": "Living Room",
        }

        with patch.object(mediator, "_compute_embedding", return_value="test_embedding"):
            with patch.object(mediator, "_get_agent_conversation_id", return_value="conv_1"):
                entries = await mediator._compile_events_to_entries_first_person(events, "andi")

        assert len(entries) == 1
        content = entries[0].content
        # Should have the visual banner from format_self_action_guidance
        assert "[== Your Turn ==]" in content
        assert "═" * 60 in content
        assert "You started at Kitchen" in content
        assert "Now you are at Living Room" in content


class TestDocumentTypeAssignment:
    """Test document type assignment for entries."""

    @pytest.mark.asyncio
    async def test_third_person_events_get_doc_mud_world(
        self, mock_redis, mediator_config
    ):
        """Third-person events should get DOC_MUD_WORLD type."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        events = [make_event(content="Hello from player", event_id="e1")]

        with patch.object(mediator, "_compute_embedding", return_value="test_embedding"):
            entries = await mediator._compile_events_to_entries_third_person(events)

        assert entries[0].document_type == DOC_MUD_WORLD

    @pytest.mark.asyncio
    async def test_first_person_events_get_doc_mud_agent(
        self, mock_redis, mediator_config
    ):
        """First-person events should get DOC_MUD_AGENT type."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        events = [make_event(event_type=EventType.SPEECH, content="I say hello", event_id="e1")]

        with patch.object(mediator, "_compute_embedding", return_value="test_embedding"):
            with patch.object(mediator, "_get_agent_conversation_id", return_value="conv_1"):
                entries = await mediator._compile_events_to_entries_first_person(events, "andi")

        assert entries[0].document_type == DOC_MUD_AGENT

    @pytest.mark.asyncio
    async def test_code_action_events_get_doc_code_action(
        self, mock_redis, mediator_config
    ):
        """CODE_ACTION events should get DOC_CODE_ACTION type."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        events = [
            make_event(
                event_type=EventType.CODE_ACTION,
                content="ls -la output",
                event_id="e1",
            )
        ]

        with patch.object(mediator, "_compute_embedding", return_value="test_embedding"):
            entries = await mediator._compile_events_to_entries_third_person(events)

        assert entries[0].document_type == DOC_CODE_ACTION

    @pytest.mark.asyncio
    async def test_code_file_events_get_doc_code_file(
        self, mock_redis, mediator_config
    ):
        """CODE_FILE events should get DOC_CODE_FILE type."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        events = [
            make_event(
                event_type=EventType.CODE_FILE,
                content="file contents here",
                event_id="e1",
            )
        ]

        with patch.object(mediator, "_compute_embedding", return_value="test_embedding"):
            entries = await mediator._compile_events_to_entries_third_person(events)

        assert entries[0].document_type == DOC_CODE_FILE


class TestSkipSave:
    """Test skip_save flag assignment for CVM indexing."""

    @pytest.mark.asyncio
    async def test_third_person_world_events_written_without_embedding(
        self, mock_redis, mediator_config
    ):
        """DOC_MUD_WORLD entries get written to CVM but without embeddings."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        events = [make_event(content="Hello from player", event_id="e1")]

        with patch.object(mediator, "_compute_embedding", return_value="test_embedding"):
            entries = await mediator._compile_events_to_entries_third_person(events)

        assert entries[0].document_type == DOC_MUD_WORLD
        assert entries[0].skip_save is False  # Written to CVM
        assert entries[0].embedding is None  # But no embedding (not searchable)

    @pytest.mark.asyncio
    async def test_code_action_events_do_not_skip_save(
        self, mock_redis, mediator_config
    ):
        """DOC_CODE_ACTION entries should have skip_save=False (indexed to CVM)."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        events = [
            make_event(
                event_type=EventType.CODE_ACTION,
                content="ls -la output",
                event_id="e1",
            )
        ]

        with patch.object(mediator, "_compute_embedding", return_value="test_embedding"):
            entries = await mediator._compile_events_to_entries_third_person(events)

        assert entries[0].document_type == DOC_CODE_ACTION
        assert entries[0].skip_save is False

    @pytest.mark.asyncio
    async def test_code_file_events_do_not_skip_save(
        self, mock_redis, mediator_config
    ):
        """DOC_CODE_FILE entries should have skip_save=False (indexed to CVM)."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        events = [
            make_event(
                event_type=EventType.CODE_FILE,
                content="file contents",
                event_id="e1",
            )
        ]

        with patch.object(mediator, "_compute_embedding", return_value="test_embedding"):
            entries = await mediator._compile_events_to_entries_third_person(events)

        assert entries[0].document_type == DOC_CODE_FILE
        assert entries[0].skip_save is False

    @pytest.mark.asyncio
    async def test_first_person_agent_events_do_not_skip_save(
        self, mock_redis, mediator_config
    ):
        """DOC_MUD_AGENT entries should have skip_save=False (indexed to CVM)."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        events = [make_event(event_type=EventType.SPEECH, content="I say hello", event_id="e1")]

        with patch.object(mediator, "_compute_embedding", return_value="test_embedding"):
            with patch.object(mediator, "_get_agent_conversation_id", return_value="conv_1"):
                entries = await mediator._compile_events_to_entries_first_person(events, "andi")

        assert entries[0].document_type == DOC_MUD_AGENT
        assert entries[0].skip_save is False


class TestEntryMetadata:
    """Test metadata in conversation entries."""

    @pytest.mark.asyncio
    async def test_entry_contains_event_metadata(
        self, mock_redis, mediator_config
    ):
        """Entry metadata should contain event information."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        events = [
            make_event(
                actor="Player",
                actor_id="p1",
                room_id="room_123",
                room_name="Living Room",
                event_id="e1",
            ),
            make_event(
                actor="Player",
                actor_id="p1",
                room_id="room_123",
                room_name="Living Room",
                event_id="e2",
            ),
        ]

        with patch.object(mediator, "_compute_embedding", return_value="test_embedding"):
            entries = await mediator._compile_events_to_entries_third_person(events)

        assert len(entries) == 1
        metadata = entries[0].metadata
        assert metadata["room_id"] == "room_123"
        assert metadata["room_name"] == "Living Room"
        assert metadata["event_count"] == 2
        assert "Player" in metadata["actors"]
        assert "p1" in metadata["actor_ids"]
        assert "e1" in metadata["event_ids"]
        assert "e2" in metadata["event_ids"]


class TestSpeakerId:
    """Test speaker_id assignment."""

    @pytest.mark.asyncio
    async def test_third_person_events_have_world_speaker(
        self, mock_redis, mediator_config
    ):
        """Third-person events should have speaker_id='world'."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        events = [make_event(event_id="e1")]

        with patch.object(mediator, "_compute_embedding", return_value="test_embedding"):
            entries = await mediator._compile_events_to_entries_third_person(events)

        assert entries[0].speaker_id == "world"

    @pytest.mark.asyncio
    async def test_first_person_events_have_agent_speaker(
        self, mock_redis, mediator_config
    ):
        """First-person events should have speaker_id=agent_id."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        events = [make_event(event_id="e1")]

        with patch.object(mediator, "_compute_embedding", return_value="test_embedding"):
            with patch.object(mediator, "_get_agent_conversation_id", return_value="conv_1"):
                entries = await mediator._compile_events_to_entries_first_person(events, "andi")

        assert entries[0].speaker_id == "andi"


class TestConversationIdGeneration:
    """Test conversation ID retrieval and generation."""

    @pytest.mark.asyncio
    async def test_uses_existing_conversation_id(self, mock_redis, mediator_config):
        """Should use existing conversation_id from profile."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        # Set up existing profile with conversation_id
        # Key format is agent:{id} per RedisKeys.agent_profile()
        mock_redis._hashes["agent:andi"] = {
            "agent_id": "andi",
            "conversation_id": "existing_conv_123",
        }

        conv_id = await mediator._get_agent_conversation_id("andi")
        assert conv_id == "existing_conv_123"

    @pytest.mark.asyncio
    async def test_generates_new_conversation_id(self, mock_redis, mediator_config):
        """Should generate new conversation_id if none exists."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        # No profile exists
        conv_id = await mediator._get_agent_conversation_id("andi")

        assert conv_id.startswith("andimud_")
        # Format: andimud_{timestamp_ms}_{random_hex}
        parts = conv_id.split("_")
        assert len(parts) == 3


class TestEmbeddingComputation:
    """Test embedding computation."""

    @pytest.mark.asyncio
    async def test_compute_embedding_uses_model(self, mock_redis, mediator_config):
        """_compute_embedding should use the embedding model."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        mock_model = MagicMock()
        mock_model.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        mediator._embedding_model = mock_model

        result = await mediator._compute_embedding("test content")

        mock_model.assert_called_once_with("test content")
        # Result should be base64 encoded
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_embedding_is_base64_encoded(self, mock_redis, mediator_config):
        """Embedding should be base64 encoded."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        mock_model = MagicMock()
        mock_model.return_value = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        mediator._embedding_model = mock_model

        result = await mediator._compute_embedding("test")

        # Verify we can decode it back
        vector = MUDConversationEntry(
            role="user",
            content="test",
            tokens=1,
            document_type=DOC_MUD_WORLD,
            conversation_id="test",
            sequence_no=0,
            embedding=result,
        ).get_embedding_vector()

        assert vector is not None
        np.testing.assert_array_almost_equal(
            vector, np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        )


class TestEmbeddingSharing:
    """Test that embeddings are computed once and shared to multiple observers."""

    @pytest.mark.asyncio
    async def test_embedding_computed_once_for_multiple_observers(
        self, mock_redis, mediator_config
    ):
        """Third-person embedding should be computed once for all observers."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        event = make_event(event_id="e1")

        # Track embedding computation calls
        embedding_calls = []

        async def mock_compute_embedding(content):
            embedding_calls.append(content)
            return "test_embedding_b64"

        # Mock Redis list operations for conversation tracking
        list_lengths = {"conversation:andi": 0, "conversation:nova": 0, "conversation:val": 0}
        async def mock_llen(key):
            return list_lengths.get(key, 0)

        async def mock_rpush(key, value):
            if key not in list_lengths:
                list_lengths[key] = 0
            list_lengths[key] += 1
            return list_lengths[key]

        mock_redis.llen = AsyncMock(side_effect=mock_llen)
        mock_redis.rpush = AsyncMock(side_effect=mock_rpush)

        with patch.object(mediator, "_compute_embedding", side_effect=mock_compute_embedding):
            with patch.object(mediator, "_get_agent_conversation_id", return_value="conv_1"):
                # Queue event for 3 observers - compiles immediately
                await mediator.queue_event_for_compilation(
                    event,
                    observer_agent_ids=["andi", "nova", "val"],
                    self_action_agent_id=None,
                )

        # Third-person (world) entries don't get embeddings computed
        # (only agent's own actions are searchable)
        assert len(embedding_calls) == 0

    @pytest.mark.asyncio
    async def test_first_person_gets_separate_embedding(
        self, mock_redis, mediator_config
    ):
        """First-person entry should get its own embedding (different content)."""
        mediator = MockMediatorWithCompiler(mock_redis, mediator_config)

        event = make_event(event_id="e1", actor="Andi", actor_id="andi_1")

        # Track embedding computation calls
        embedding_calls = []

        async def mock_compute_embedding(content):
            embedding_calls.append(content)
            return "test_embedding_b64"

        # Mock Redis operations
        list_lengths = {}
        async def mock_llen(key):
            return list_lengths.get(key, 0)

        async def mock_rpush(key, value):
            if key not in list_lengths:
                list_lengths[key] = 0
            list_lengths[key] += 1
            return list_lengths[key]

        mock_redis.llen = AsyncMock(side_effect=mock_llen)
        mock_redis.rpush = AsyncMock(side_effect=mock_rpush)

        with patch.object(mediator, "_compute_embedding", side_effect=mock_compute_embedding):
            with patch.object(mediator, "_get_agent_conversation_id", return_value="conv_1"):
                # Queue event with 2 observers AND a self-action - compiles immediately
                await mediator.queue_event_for_compilation(
                    event,
                    observer_agent_ids=["nova", "val"],
                    self_action_agent_id="andi",
                )

        # Should have computed embedding only ONCE:
        # - Third-person (observers/world) entries don't get embeddings
        # - First-person (self-action) gets embedding (agent's own actions are searchable)
        assert len(embedding_calls) == 1
