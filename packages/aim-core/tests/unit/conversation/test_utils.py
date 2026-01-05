# tests/unit/conversation/test_utils.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

"""Tests for conversation utilities."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from aim.conversation.utils import rebuild_agent_index, _get_chunk_stats
from aim.conversation.message import ConversationMessage
from aim.constants import CHUNK_LEVEL_FULL, CHUNK_LEVEL_768, CHUNK_LEVEL_256


@pytest.fixture
def sample_messages():
    """Sample conversation messages for testing."""
    return [
        ConversationMessage(
            doc_id="msg_1",
            conversation_id="conv_1",
            user_id="user_1",
            persona_id="test_agent",
            role="user",
            content="Hello, how are you?",
            document_type="conversation",
            timestamp=1000000,
            sequence_no=1,
            branch=0,
            speaker_id="user_1",
            listener_id="test_agent"
        ),
        ConversationMessage(
            doc_id="msg_2",
            conversation_id="conv_1",
            user_id="user_1",
            persona_id="test_agent",
            role="assistant",
            content="I'm doing well, thank you!",
            document_type="conversation",
            timestamp=1000001,
            sequence_no=2,
            branch=0,
            speaker_id="test_agent",
            listener_id="user_1"
        )
    ]


@pytest.fixture
def mock_loader(sample_messages):
    """Mock ConversationLoader that returns sample messages."""
    loader = MagicMock()
    loader.load_all.return_value = sample_messages
    return loader


@pytest.fixture
def mock_index():
    """Mock SearchIndex with stats."""
    index = MagicMock()

    # Mock searcher and results for chunk stats
    searcher = MagicMock()
    index.index.searcher.return_value = searcher

    # Mock query parsing and search results
    def mock_search(query, limit):
        result = MagicMock()
        # Return different counts based on chunk level
        if hasattr(query, '_level'):
            if query._level == CHUNK_LEVEL_FULL:
                result.count = 2
            elif query._level == CHUNK_LEVEL_768:
                result.count = 5
            elif query._level == CHUNK_LEVEL_256:
                result.count = 10
        else:
            result.count = 0
        return result

    searcher.search.side_effect = mock_search

    # Mock parse_query to attach level info
    def mock_parse_query(query, default_field_names):
        parsed = MagicMock()
        parsed._level = query
        return parsed

    index.index.parse_query.side_effect = mock_parse_query

    return index


def test_rebuild_agent_index_full(tmp_path, sample_messages, mock_loader, mock_index):
    """Test full rebuild of agent index."""
    # Setup test directories
    memory_base = tmp_path / "memory"
    agent_id = "test_agent"
    conversations_dir = memory_base / agent_id / "conversations"
    conversations_dir.mkdir(parents=True)

    # Mock the loader and index
    with patch('aim.conversation.utils.ConversationLoader', return_value=mock_loader):
        with patch('aim.conversation.utils.SearchIndex', return_value=mock_index):
            result = rebuild_agent_index(
                agent_id=agent_id,
                embedding_model="test-model",
                device="cpu",
                full=True,
                memory_base=str(memory_base)
            )

    # Verify rebuild was called
    mock_index.rebuild.assert_called_once()

    # Verify result structure for full rebuild
    assert result["mode"] == "full"
    assert result["total_messages"] == 2
    assert "total_entries" in result
    assert "chunk_stats" in result
    assert result["chunk_stats"]["full"] == 2
    assert result["chunk_stats"]["768"] == 5
    assert result["chunk_stats"]["256"] == 10


def test_rebuild_agent_index_incremental(tmp_path, sample_messages, mock_loader, mock_index):
    """Test incremental update of agent index."""
    # Setup test directories with existing index
    memory_base = tmp_path / "memory"
    agent_id = "test_agent"
    conversations_dir = memory_base / agent_id / "conversations"
    index_dir = memory_base / agent_id / "indices"
    conversations_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)

    # Create a dummy file to make index appear non-empty
    (index_dir / "dummy.txt").touch()

    # Mock incremental_update to return counts
    mock_index.incremental_update.return_value = (1, 2, 0)  # added, updated, deleted

    # Mock the loader and index
    with patch('aim.conversation.utils.ConversationLoader', return_value=mock_loader):
        with patch('aim.conversation.utils.SearchIndex', return_value=mock_index):
            result = rebuild_agent_index(
                agent_id=agent_id,
                embedding_model="test-model",
                device="cpu",
                full=False,
                memory_base=str(memory_base)
            )

    # Verify incremental_update was called
    mock_index.incremental_update.assert_called_once()

    # Verify result structure for incremental
    assert result["mode"] == "incremental"
    assert result["added"] == 1
    assert result["updated"] == 2
    assert result["deleted"] == 0
    assert result["total_messages"] == 2
    assert "total_entries" in result
    assert "chunk_stats" in result


def test_rebuild_agent_index_missing_conversations_dir(tmp_path):
    """Test rebuild fails gracefully when conversations directory doesn't exist."""
    memory_base = tmp_path / "memory"
    agent_id = "test_agent"

    with pytest.raises(FileNotFoundError, match="Conversations directory not found"):
        rebuild_agent_index(
            agent_id=agent_id,
            embedding_model="test-model",
            device="cpu",
            memory_base=str(memory_base)
        )


def test_rebuild_agent_index_no_messages(tmp_path, mock_loader):
    """Test rebuild fails when no messages are found."""
    # Setup test directories
    memory_base = tmp_path / "memory"
    agent_id = "test_agent"
    conversations_dir = memory_base / agent_id / "conversations"
    conversations_dir.mkdir(parents=True)

    # Mock loader to return empty list
    mock_loader.load_all.return_value = []

    with patch('aim.conversation.utils.ConversationLoader', return_value=mock_loader):
        with pytest.raises(ValueError, match="No messages found"):
            rebuild_agent_index(
                agent_id=agent_id,
                embedding_model="test-model",
                device="cpu",
                memory_base=str(memory_base)
            )


def test_get_chunk_stats(mock_index):
    """Test chunk statistics extraction."""
    stats = _get_chunk_stats(mock_index)

    assert "full" in stats
    assert "768" in stats
    assert "256" in stats
    assert stats["full"] == 2
    assert stats["768"] == 5
    assert stats["256"] == 10


def test_rebuild_with_custom_batch_size(tmp_path, sample_messages, mock_loader, mock_index):
    """Test rebuild respects custom batch size parameter."""
    # Setup test directories with existing index
    memory_base = tmp_path / "memory"
    agent_id = "test_agent"
    conversations_dir = memory_base / agent_id / "conversations"
    index_dir = memory_base / agent_id / "indices"
    conversations_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)
    (index_dir / "dummy.txt").touch()

    mock_index.incremental_update.return_value = (0, 0, 0)

    with patch('aim.conversation.utils.ConversationLoader', return_value=mock_loader):
        with patch('aim.conversation.utils.SearchIndex', return_value=mock_index):
            rebuild_agent_index(
                agent_id=agent_id,
                embedding_model="test-model",
                device="cpu",
                batch_size=128,
                full=False,
                memory_base=str(memory_base)
            )

    # Verify batch_size was passed to incremental_update
    call_kwargs = mock_index.incremental_update.call_args[1]
    assert call_kwargs["batch_size"] == 128


def test_rebuild_with_custom_device(tmp_path, sample_messages, mock_loader, mock_index):
    """Test rebuild respects custom device parameter."""
    memory_base = tmp_path / "memory"
    agent_id = "test_agent"
    conversations_dir = memory_base / agent_id / "conversations"
    conversations_dir.mkdir(parents=True)

    with patch('aim.conversation.utils.ConversationLoader', return_value=mock_loader):
        with patch('aim.conversation.utils.SearchIndex', return_value=mock_index) as mock_search_index:
            rebuild_agent_index(
                agent_id=agent_id,
                embedding_model="test-model",
                device="cuda",
                full=True,
                memory_base=str(memory_base)
            )

    # Verify device was passed to SearchIndex
    call_kwargs = mock_search_index.call_args[1]
    assert call_kwargs["device"] == "cuda"
