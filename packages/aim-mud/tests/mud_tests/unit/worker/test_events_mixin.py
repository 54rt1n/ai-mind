# tests/unit/worker/test_events_mixin.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for EventsMixin conversation entry methods."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aim.constants import DOC_MUD_WORLD, DOC_MUD_AGENT, DOC_MUD_ACTION
from aim_mud_types import MUDConversationEntry, MUDSession


class MockWorker:
    """Mock worker for testing EventsMixin methods."""

    def __init__(self, redis_client, agent_id="test_agent"):
        self.redis = redis_client
        self.config = MagicMock()
        self.config.agent_id = agent_id
        self.session = MUDSession(
            agent_id=agent_id,
            persona_id="test_persona",
        )


class TestGetNewConversationEntries:
    """Tests for get_new_conversation_entries method."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis = AsyncMock()
        redis.lrange = AsyncMock(return_value=[])
        redis.llen = AsyncMock(return_value=0)
        return redis

    @pytest.fixture
    def mock_worker(self, mock_redis):
        """Create mock worker with EventsMixin."""
        from andimud_worker.mixins.datastore.events import EventsMixin

        worker = MockWorker(mock_redis)
        # Apply mixin methods to worker instance
        worker.get_new_conversation_entries = EventsMixin.get_new_conversation_entries.__get__(
            worker, type(worker)
        )
        worker._read_entries_once = EventsMixin._read_entries_once.__get__(
            worker, type(worker)
        )
        return worker

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_entries(self, mock_worker, mock_redis):
        """Should return empty list when no conversation entries exist."""
        mock_redis.lrange.return_value = []

        entries = await mock_worker.get_new_conversation_entries()

        assert entries == []
        assert mock_worker.session.last_conversation_index == 0

    @pytest.mark.asyncio
    async def test_reads_entries_from_last_position(self, mock_worker, mock_redis):
        """Should read entries starting from last_conversation_index."""
        mock_worker.session.last_conversation_index = 5
        entry = MUDConversationEntry(
            role="user",
            content="Test content",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=5,
            speaker_id="world",
        )
        mock_redis.lrange.return_value = [entry.model_dump_json().encode()]

        entries = await mock_worker.get_new_conversation_entries()

        assert len(entries) == 1
        assert entries[0].content == "Test content"
        # lrange called with start=5 (last_read position)
        mock_redis.lrange.assert_called_once()
        call_args = mock_redis.lrange.call_args[0]
        assert call_args[1] == 5  # start index

    @pytest.mark.asyncio
    async def test_updates_last_conversation_index(self, mock_worker, mock_redis):
        """Should update session.last_conversation_index after reading."""
        entry1 = MUDConversationEntry(
            role="user",
            content="Entry 1",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=0,
            speaker_id="world",
        )
        entry2 = MUDConversationEntry(
            role="user",
            content="Entry 2",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=1,
            speaker_id="world",
        )
        mock_redis.lrange.return_value = [
            entry1.model_dump_json().encode(),
            entry2.model_dump_json().encode(),
        ]

        entries = await mock_worker.get_new_conversation_entries()

        assert len(entries) == 2
        assert mock_worker.session.last_conversation_index == 2

    @pytest.mark.asyncio
    async def test_handles_parse_errors_gracefully(self, mock_worker, mock_redis):
        """Should skip entries that fail to parse."""
        valid_entry = MUDConversationEntry(
            role="user",
            content="Valid entry",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=0,
            speaker_id="world",
        )
        mock_redis.lrange.return_value = [
            b"invalid json {{{",
            valid_entry.model_dump_json().encode(),
        ]

        entries = await mock_worker.get_new_conversation_entries()

        assert len(entries) == 1
        assert entries[0].content == "Valid entry"

    @pytest.mark.asyncio
    async def test_always_reads_all_entries(self, mock_worker, mock_redis):
        """Should always read all available entries (end=-1)."""
        entry = MUDConversationEntry(
            role="user",
            content="Test content",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=0,
            speaker_id="world",
        )
        mock_redis.lrange.return_value = [entry.model_dump_json().encode()]

        await mock_worker.get_new_conversation_entries()

        # lrange end should be -1 (read all)
        mock_redis.lrange.assert_called_once()
        call_args = mock_redis.lrange.call_args[0]
        assert call_args[2] == -1


class TestGetNewConversationEntriesWithSettling:
    """Tests for get_new_conversation_entries with settle=True."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis = AsyncMock()
        redis.lrange = AsyncMock(return_value=[])
        redis.llen = AsyncMock(return_value=0)
        return redis

    @pytest.fixture
    def mock_worker(self, mock_redis):
        """Create mock worker with EventsMixin."""
        from andimud_worker.mixins.datastore.events import EventsMixin

        worker = MockWorker(mock_redis)
        # Need event_settle_seconds config
        worker.config.event_settle_seconds = 0.1
        # Apply mixin methods to worker instance
        worker.get_new_conversation_entries = EventsMixin.get_new_conversation_entries.__get__(
            worker, type(worker)
        )
        worker._read_entries_once = EventsMixin._read_entries_once.__get__(
            worker, type(worker)
        )
        return worker

    @pytest.mark.asyncio
    async def test_settle_returns_empty_on_first_empty_read(self, mock_worker, mock_redis):
        """With settle=True, should return empty immediately if first read is empty."""
        mock_redis.lrange.return_value = []

        entries = await mock_worker.get_new_conversation_entries(settle=True)

        assert entries == []
        # Should only read once
        assert mock_redis.lrange.call_count == 1

    @pytest.mark.asyncio
    async def test_settle_accumulates_entries_across_reads(self, mock_worker, mock_redis):
        """With settle=True, should accumulate entries from multiple reads."""
        entry1 = MUDConversationEntry(
            role="user",
            content="First entry",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=0,
            speaker_id="world",
        )
        entry2 = MUDConversationEntry(
            role="user",
            content="Second entry",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=1,
            speaker_id="world",
        )

        # First read returns entry1, second returns entry2, then two empty reads to settle
        call_count = [0]

        async def mock_lrange(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [entry1.model_dump_json().encode()]
            elif call_count[0] == 2:
                return [entry2.model_dump_json().encode()]
            else:
                return []

        mock_redis.lrange.side_effect = mock_lrange

        entries = await mock_worker.get_new_conversation_entries(settle=True)

        # Should have accumulated both entries
        assert len(entries) == 2
        assert entries[0].content == "First entry"
        assert entries[1].content == "Second entry"
        # Two empty reads needed to confirm settling
        assert call_count[0] == 4

    @pytest.mark.asyncio
    async def test_settle_waits_between_reads(self, mock_worker, mock_redis):
        """With settle=True, should wait between reads."""
        entry = MUDConversationEntry(
            role="user",
            content="Test entry",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=0,
            speaker_id="world",
        )

        # One entry, then empty reads
        call_count = [0]

        async def mock_lrange(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [entry.model_dump_json().encode()]
            return []

        mock_redis.lrange.side_effect = mock_lrange

        with patch("andimud_worker.mixins.datastore.events.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            entries = await mock_worker.get_new_conversation_entries(settle=True)

            # Should have slept between reads
            assert mock_sleep.call_count >= 1

        assert len(entries) == 1


class TestCollapseConsecutiveEntries:
    """Tests for collapse_consecutive_entries method."""

    @pytest.fixture
    def mock_worker(self):
        """Create mock worker with EventsMixin."""
        from andimud_worker.mixins.datastore.events import EventsMixin

        worker = MockWorker(AsyncMock())
        # Apply mixin methods to worker instance
        worker.collapse_consecutive_entries = EventsMixin.collapse_consecutive_entries.__get__(
            worker, type(worker)
        )
        worker._merge_entries = EventsMixin._merge_entries.__get__(
            worker, type(worker)
        )
        return worker

    def test_returns_empty_for_empty_input(self, mock_worker):
        """Should return empty list for empty input."""
        result = mock_worker.collapse_consecutive_entries([])
        assert result == []

    def test_single_entry_unchanged(self, mock_worker):
        """Single entry should be returned unchanged."""
        entry = MUDConversationEntry(
            role="user",
            content="Single entry",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=0,
            speaker_id="world",
        )

        result = mock_worker.collapse_consecutive_entries([entry])

        assert len(result) == 1
        assert result[0].content == "Single entry"

    def test_collapses_consecutive_world_entries_same_speaker(self, mock_worker):
        """Should collapse consecutive DOC_MUD_WORLD entries with same speaker."""
        entry1 = MUDConversationEntry(
            role="user",
            content="First message",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=0,
            speaker_id="world",
            metadata={"event_ids": ["e1"]},
        )
        entry2 = MUDConversationEntry(
            role="user",
            content="Second message",
            tokens=15,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=1,
            speaker_id="world",
            metadata={"event_ids": ["e2"]},
        )

        result = mock_worker.collapse_consecutive_entries([entry1, entry2])

        assert len(result) == 1
        assert "First message" in result[0].content
        assert "Second message" in result[0].content
        assert result[0].tokens == 25

    def test_does_not_collapse_different_speakers(self, mock_worker):
        """Should not collapse entries with different speaker_ids."""
        entry1 = MUDConversationEntry(
            role="user",
            content="From speaker A",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=0,
            speaker_id="speaker_a",
        )
        entry2 = MUDConversationEntry(
            role="user",
            content="From speaker B",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=1,
            speaker_id="speaker_b",
        )

        result = mock_worker.collapse_consecutive_entries([entry1, entry2])

        assert len(result) == 2

    def test_does_not_collapse_non_world_entries(self, mock_worker):
        """Should not collapse DOC_MUD_AGENT or DOC_MUD_ACTION entries."""
        world_entry = MUDConversationEntry(
            role="user",
            content="World event",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=0,
            speaker_id="world",
        )
        agent_entry = MUDConversationEntry(
            role="assistant",
            content="Agent response",
            tokens=10,
            document_type=DOC_MUD_AGENT,
            conversation_id="test_conv",
            sequence_no=1,
            speaker_id="test_agent",
        )
        world_entry2 = MUDConversationEntry(
            role="user",
            content="Another world event",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=2,
            speaker_id="world",
        )

        result = mock_worker.collapse_consecutive_entries([world_entry, agent_entry, world_entry2])

        # Agent entry breaks the sequence, so all 3 are separate
        assert len(result) == 3

    def test_merges_event_ids_from_metadata(self, mock_worker):
        """Should merge event_ids from metadata when collapsing."""
        entry1 = MUDConversationEntry(
            role="user",
            content="First",
            tokens=5,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=0,
            speaker_id="world",
            metadata={"event_ids": ["e1", "e2"]},
        )
        entry2 = MUDConversationEntry(
            role="user",
            content="Second",
            tokens=5,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=1,
            speaker_id="world",
            metadata={"event_ids": ["e3"]},
        )

        result = mock_worker.collapse_consecutive_entries([entry1, entry2])

        assert len(result) == 1
        assert result[0].metadata["event_ids"] == ["e1", "e2", "e3"]
        assert result[0].metadata["merged_entry_count"] == 2

    def test_uses_first_entry_embedding(self, mock_worker):
        """Should use first entry's embedding in merged result."""
        entry1 = MUDConversationEntry(
            role="user",
            content="First",
            tokens=5,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=0,
            speaker_id="world",
            embedding="base64_embedding_1",
        )
        entry2 = MUDConversationEntry(
            role="user",
            content="Second",
            tokens=5,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=1,
            speaker_id="world",
            embedding="base64_embedding_2",
        )

        result = mock_worker.collapse_consecutive_entries([entry1, entry2])

        assert result[0].embedding == "base64_embedding_1"

    def test_uses_last_entry_last_event_id(self, mock_worker):
        """Should use last entry's last_event_id in merged result."""
        entry1 = MUDConversationEntry(
            role="user",
            content="First",
            tokens=5,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=0,
            speaker_id="world",
            last_event_id="event-1",
        )
        entry2 = MUDConversationEntry(
            role="user",
            content="Second",
            tokens=5,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=1,
            speaker_id="world",
            last_event_id="event-2",
        )

        result = mock_worker.collapse_consecutive_entries([entry1, entry2])

        assert result[0].last_event_id == "event-2"


class TestMergeEntries:
    """Tests for _merge_entries helper method."""

    @pytest.fixture
    def mock_worker(self):
        """Create mock worker with EventsMixin."""
        from andimud_worker.mixins.datastore.events import EventsMixin

        worker = MockWorker(AsyncMock())
        worker._merge_entries = EventsMixin._merge_entries.__get__(
            worker, type(worker)
        )
        return worker

    def test_single_entry_returned_unchanged(self, mock_worker):
        """Single entry should be returned as-is."""
        entry = MUDConversationEntry(
            role="user",
            content="Single entry",
            tokens=10,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=0,
            speaker_id="world",
        )

        result = mock_worker._merge_entries([entry])

        assert result is entry

    def test_joins_content_with_double_newline(self, mock_worker):
        """Should join content with double newlines."""
        entry1 = MUDConversationEntry(
            role="user",
            content="Line 1",
            tokens=5,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=0,
            speaker_id="world",
        )
        entry2 = MUDConversationEntry(
            role="user",
            content="Line 2",
            tokens=5,
            document_type=DOC_MUD_WORLD,
            conversation_id="test_conv",
            sequence_no=1,
            speaker_id="world",
        )

        result = mock_worker._merge_entries([entry1, entry2])

        assert result.content == "Line 1\n\nLine 2"
