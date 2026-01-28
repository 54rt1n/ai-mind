# tests/core_tests/unit/conversation/test_precomputed_embedding.py
"""Tests for pre-computed embedding support in SearchIndex and ConversationModel."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

from aim.conversation.index import SearchIndex
from aim.conversation.model import ConversationModel
from aim.constants import CHUNK_LEVEL_FULL, CHUNK_LEVEL_256, CHUNK_LEVEL_768


class TestSearchIndexPrecomputedEmbedding:
    """Tests for SearchIndex.add_document() with pre-computed embeddings."""

    @pytest.fixture
    def mock_vectorizer(self):
        """Create a mock vectorizer that returns predictable embeddings."""
        vectorizer = MagicMock()
        # Return a 384-dim vector (typical for MiniLM)
        vectorizer.return_value = np.random.randn(384).astype(np.float32)
        vectorizer.transform.return_value = [np.random.randn(384).astype(np.float32)]
        return vectorizer

    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing."""
        return {
            "doc_id": "test-doc-001",
            "content": "This is test content for embedding.",
            "conversation_id": "conv-001",
            "user_id": "user-001",
            "persona_id": "persona-001",
            "role": "user",
            "document_type": "conversation",
            "timestamp": 1704067200,
            "sequence_no": 1,
            "branch": 0,
        }

    @pytest.fixture
    def precomputed_embedding(self):
        """Create a pre-computed embedding vector."""
        return np.random.randn(384).astype(np.float32)

    @patch('aim.conversation.index.Index')
    @patch('aim.conversation.index.HuggingFaceEmbedding')
    def test_add_document_with_precomputed_embedding(
        self, MockEmbedding, MockIndex, mock_vectorizer, sample_document, precomputed_embedding
    ):
        """Test that add_document ignores pre-computed embedding and computes all internally (Phase 2)."""
        MockEmbedding.return_value = mock_vectorizer
        mock_writer = MagicMock()
        MockIndex.return_value.writer.return_value = mock_writer

        index = SearchIndex(Path("/tmp/test_index"), embedding_model="test-model")

        # Load vectorizer before adding documents (Phase 2 requirement)
        index.load_vectorizer()

        index.add_document(sample_document, embedding=precomputed_embedding)

        # Release vectorizer after adding
        index.release_vectorizer()

        # Phase 2: Pre-computed embedding is IGNORED, all embeddings computed internally
        # Vectorizer should be called for ALL entries (full + chunks)
        call_count = mock_vectorizer.call_count

        # The document will be expanded to full + chunk_768 + chunk_256 entries
        # All entries should have embeddings computed via vectorizer
        entries = index._expand_document_to_entries(sample_document)

        # Vectorizer should be called exactly once per entry
        assert call_count == len(entries)

    @patch('aim.conversation.index.Index')
    @patch('aim.conversation.index.HuggingFaceEmbedding')
    def test_add_document_without_embedding_computes_all(
        self, MockEmbedding, MockIndex, mock_vectorizer, sample_document
    ):
        """Test that add_document computes embeddings for all entries when none provided."""
        MockEmbedding.return_value = mock_vectorizer
        mock_writer = MagicMock()
        MockIndex.return_value.writer.return_value = mock_writer

        index = SearchIndex(Path("/tmp/test_index"), embedding_model="test-model")

        # Load vectorizer before adding documents (Phase 2 requirement)
        index.load_vectorizer()

        index.add_document(sample_document)  # No embedding parameter

        # Release vectorizer after adding
        index.release_vectorizer()

        # Vectorizer should be called for ALL entries (full + chunks)
        entries = index._expand_document_to_entries(sample_document)
        assert mock_vectorizer.call_count == len(entries)

    @patch('aim.conversation.index.Index')
    @patch('aim.conversation.index.HuggingFaceEmbedding')
    def test_add_document_with_none_embedding_computes_all(
        self, MockEmbedding, MockIndex, mock_vectorizer, sample_document
    ):
        """Test that add_document computes embeddings when embedding=None is explicitly passed."""
        MockEmbedding.return_value = mock_vectorizer
        mock_writer = MagicMock()
        MockIndex.return_value.writer.return_value = mock_writer

        index = SearchIndex(Path("/tmp/test_index"), embedding_model="test-model")

        # Load vectorizer before adding documents (Phase 2 requirement)
        index.load_vectorizer()

        index.add_document(sample_document, embedding=None)

        # Release vectorizer after adding
        index.release_vectorizer()

        # Vectorizer should be called for ALL entries
        entries = index._expand_document_to_entries(sample_document)
        assert mock_vectorizer.call_count == len(entries)


class TestConversationModelPrecomputedQueryEmbedding:
    """Tests for ConversationModel.query() with pre-computed query embeddings."""

    @pytest.fixture
    def mock_search_index(self):
        """Create a mock SearchIndex with vectorizer."""
        mock_index = MagicMock()
        mock_index.vectorizer = MagicMock()
        mock_index.vectorizer.transform.return_value = [np.random.randn(384).astype(np.float32)]
        return mock_index

    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results DataFrame."""
        import pandas as pd
        return pd.DataFrame({
            "doc_id": ["doc1", "doc2", "doc3"],
            "content": ["content one", "content two", "content three"],
            "conversation_id": ["conv1", "conv1", "conv2"],
            "user_id": ["user1", "user1", "user2"],
            "persona_id": ["persona1", "persona1", "persona1"],
            "role": ["user", "assistant", "user"],
            "document_type": ["conversation", "conversation", "conversation"],
            "timestamp": [1704067200, 1704067300, 1704067400],
            "sequence_no": [1, 2, 3],
            "branch": [0, 0, 0],
            "weight": [1.0, 1.0, 1.0],
            "importance": [1.0, 1.0, 1.0],
            "hits": [1, 1, 1],
            "distance": [0.5, 0.7, 0.9],
            "index_a": [
                np.random.randn(384).astype(np.float32),
                np.random.randn(384).astype(np.float32),
                np.random.randn(384).astype(np.float32),
            ],
            "parent_doc_id": ["doc1", "doc2", "doc3"],
            "chunk_level": ["full", "full", "full"],
            "chunk_index": [0, 0, 0],
            "chunk_start": [0, 0, 0],
            "chunk_end": [10, 10, 10],
            "chunk_count": [1, 1, 1],
        })

    @pytest.fixture
    def query_embedding(self):
        """Create a pre-computed query embedding."""
        return np.random.randn(384).astype(np.float32)

    @patch('aim.conversation.model.SearchIndex')
    @patch('aim.conversation.model.ConversationLoader')
    @patch('aim.conversation.model.ConversationModel.maybe_init_folders')
    def test_query_with_precomputed_embedding(
        self, mock_init, MockLoader, MockIndex, mock_search_index, sample_search_results, query_embedding
    ):
        """Test that query uses pre-computed embedding for FAISS reranking."""
        MockIndex.return_value = mock_search_index
        mock_search_index.search.return_value = sample_search_results

        model = ConversationModel(
            memory_path="/tmp/test_memory",
            embedding_model="test-model",
            user_timezone="UTC"
        )

        result = model.query(
            query_texts=["test query"],
            top_n=10,
            query_embedding=query_embedding
        )

        # Vectorizer.transform should NOT be called when pre-computed embedding is provided
        mock_search_index.vectorizer.transform.assert_not_called()

        # Result should still be a DataFrame
        assert result is not None
        assert len(result) > 0

    @patch('aim.conversation.model.SearchIndex')
    @patch('aim.conversation.model.ConversationLoader')
    @patch('aim.conversation.model.ConversationModel.maybe_init_folders')
    def test_query_without_embedding_computes_from_text(
        self, mock_init, MockLoader, MockIndex, mock_search_index, sample_search_results
    ):
        """Test that query computes embedding from text when none provided."""
        MockIndex.return_value = mock_search_index
        mock_search_index.search.return_value = sample_search_results

        model = ConversationModel(
            memory_path="/tmp/test_memory",
            embedding_model="test-model",
            user_timezone="UTC"
        )

        result = model.query(
            query_texts=["test query"],
            top_n=10
            # No query_embedding parameter
        )

        # Vectorizer.transform SHOULD be called to compute embedding from query text
        mock_search_index.vectorizer.transform.assert_called_once()

        # Result should still be a DataFrame
        assert result is not None
        assert len(result) > 0

    @patch('aim.conversation.model.SearchIndex')
    @patch('aim.conversation.model.ConversationLoader')
    @patch('aim.conversation.model.ConversationModel.maybe_init_folders')
    def test_query_with_none_embedding_computes_from_text(
        self, mock_init, MockLoader, MockIndex, mock_search_index, sample_search_results
    ):
        """Test that query computes embedding when query_embedding=None is explicitly passed."""
        MockIndex.return_value = mock_search_index
        mock_search_index.search.return_value = sample_search_results

        model = ConversationModel(
            memory_path="/tmp/test_memory",
            embedding_model="test-model",
            user_timezone="UTC"
        )

        result = model.query(
            query_texts=["test query"],
            top_n=10,
            query_embedding=None  # Explicitly None
        )

        # Vectorizer.transform SHOULD be called when embedding is None
        mock_search_index.vectorizer.transform.assert_called_once()

    @patch('aim.conversation.model.SearchIndex')
    @patch('aim.conversation.model.ConversationLoader')
    @patch('aim.conversation.model.ConversationModel.maybe_init_folders')
    def test_query_empty_results_with_precomputed_embedding(
        self, mock_init, MockLoader, MockIndex, mock_search_index, query_embedding
    ):
        """Test that query handles empty results gracefully with pre-computed embedding."""
        import pandas as pd
        MockIndex.return_value = mock_search_index
        # Return empty DataFrame
        mock_search_index.search.return_value = pd.DataFrame()

        model = ConversationModel(
            memory_path="/tmp/test_memory",
            embedding_model="test-model",
            user_timezone="UTC"
        )

        result = model.query(
            query_texts=["test query"],
            top_n=10,
            query_embedding=query_embedding
        )

        # Should handle empty results gracefully
        assert result is not None
        assert len(result) == 0
