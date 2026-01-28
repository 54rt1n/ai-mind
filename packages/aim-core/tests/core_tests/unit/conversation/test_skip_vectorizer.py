# tests/core_tests/unit/conversation/test_skip_vectorizer.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for skip_vectorizer functionality in SearchIndex and ConversationModel.

These tests verify that the skip_vectorizer option correctly prevents loading
the embedding model while maintaining functionality when pre-computed embeddings
are provided.
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestSearchIndexSkipVectorizer:
    """Tests for SearchIndex with skip_vectorizer=True."""

    @pytest.fixture
    def temp_index_path(self, tmp_path):
        """Create a temporary path for the index."""
        return tmp_path / "test_index"

    @pytest.fixture
    def sample_embedding(self):
        """Create a sample embedding vector (384-dim to match mock vectorizer)."""
        return np.random.randn(384).astype(np.float32)

    @pytest.fixture
    def sample_document(self):
        """Create a sample document dictionary."""
        return {
            "doc_id": "test-doc-001",
            "content": "This is a test document for skip_vectorizer testing.",
            "conversation_id": "test-convo",
            "user_id": "test-user",
            "persona_id": "test-persona",
            "role": "user",
            "timestamp": 1700000000,
            "sequence_no": 1,
            "branch": 0,
        }

    def test_skip_vectorizer_does_not_load_model(self, temp_index_path):
        """SearchIndex with skip_vectorizer=True should not load embedding model."""
        from aim.conversation.index import SearchIndex

        with patch("aim.conversation.index.HuggingFaceEmbedding") as mock_embedding:
            index = SearchIndex(
                index_path=temp_index_path,
                embedding_model="test-model",
                device="cpu",
                skip_vectorizer=True,
            )

            # Should NOT have called HuggingFaceEmbedding
            mock_embedding.assert_not_called()
            assert index.vectorizer is None
            assert index.skip_vectorizer is True

    def test_skip_vectorizer_false_loads_model(self, temp_index_path):
        """SearchIndex with skip_vectorizer=False should load embedding model."""
        from aim.conversation.index import SearchIndex

        with patch("aim.conversation.index.HuggingFaceEmbedding") as mock_embedding:
            mock_embedding.return_value = MagicMock()

            index = SearchIndex(
                index_path=temp_index_path,
                embedding_model="test-model",
                device="cpu",
                skip_vectorizer=False,
            )

            # Should have called HuggingFaceEmbedding
            mock_embedding.assert_called_once_with(model_name="test-model", device="cpu")
            assert index.vectorizer is not None
            assert index.skip_vectorizer is False

    def test_add_document_with_precomputed_embedding(
        self, temp_index_path, sample_document, sample_embedding
    ):
        """add_document() should work with pre-computed embedding when skip_vectorizer=True."""
        from aim.conversation.index import SearchIndex

        with patch("aim.conversation.embedding.HuggingFaceEmbedding") as mock_embedding:
            mock_embedding.return_value = MagicMock()

            index = SearchIndex(
                index_path=temp_index_path,
                embedding_model="test-model",
                device="cpu",
                skip_vectorizer=True,
            )

            # Load vectorizer before adding documents (Phase 2 requirement)
            index.load_vectorizer()

            # Should succeed with pre-computed embedding
            index.add_document(sample_document, embedding=sample_embedding)

            # Release vectorizer after batch writes
            index.release_vectorizer()

            # Verify document was indexed
            result = index.get_document(sample_document["doc_id"])
            assert result is not None
            assert result["content"] == sample_document["content"]

    def test_add_document_without_embedding_when_skip_vectorizer(
        self, temp_index_path, sample_document, caplog
    ):
        """add_document() without embedding when skip_vectorizer=True computes all embeddings (Phase 2)."""
        from aim.conversation.index import SearchIndex
        import logging

        with patch("aim.conversation.embedding.HuggingFaceEmbedding") as mock_embedding:
            mock_vectorizer = MagicMock()
            mock_vectorizer.return_value = np.random.randn(384).astype(np.float32)
            mock_embedding.return_value = mock_vectorizer

            index = SearchIndex(
                index_path=temp_index_path,
                embedding_model="test-model",
                device="cpu",
                skip_vectorizer=True,
            )

            # Load vectorizer before adding documents (Phase 2 requirement)
            index.load_vectorizer()

            # Phase 2: embedding=None means compute all embeddings, not skip
            with caplog.at_level(logging.DEBUG):
                index.add_document(sample_document, embedding=None)

            # Release vectorizer after batch writes
            index.release_vectorizer()

            # Phase 2: Document SHOULD be in the index with computed embeddings
            result = index.get_document(sample_document["doc_id"])
            assert result is not None
            assert result["content"] == sample_document["content"]

    def test_add_documents_raises_when_skip_vectorizer(self, temp_index_path, sample_document):
        """add_documents() should raise RuntimeError when skip_vectorizer=True."""
        from aim.conversation.index import SearchIndex

        with patch("aim.conversation.index.HuggingFaceEmbedding"):
            index = SearchIndex(
                index_path=temp_index_path,
                embedding_model="test-model",
                device="cpu",
                skip_vectorizer=True,
            )

            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="add_documents.*requires vectorizer"):
                index.add_documents([sample_document])

    def test_rebuild_raises_when_skip_vectorizer(
        self, temp_index_path, sample_document, sample_embedding
    ):
        """rebuild() should raise RuntimeError when skip_vectorizer=True."""
        from aim.conversation.index import SearchIndex

        with patch("aim.conversation.embedding.HuggingFaceEmbedding") as mock_embedding:
            mock_embedding.return_value = MagicMock()

            index = SearchIndex(
                index_path=temp_index_path,
                embedding_model="test-model",
                device="cpu",
                skip_vectorizer=True,
            )

            # Load vectorizer before adding documents (Phase 2 requirement)
            index.load_vectorizer()

            # Add a document first
            index.add_document(sample_document, embedding=sample_embedding)

            # Release vectorizer after adding
            index.release_vectorizer()

            # rebuild() should raise because it calls add_documents internally
            with pytest.raises(RuntimeError, match="add_documents.*requires vectorizer"):
                index.rebuild([sample_document])


class TestConversationModelSkipVectorizer:
    """Tests for ConversationModel with skip_vectorizer=True."""

    @pytest.fixture
    def temp_memory_path(self, tmp_path):
        """Create a temporary path for memory storage."""
        memory_path = tmp_path / "memory" / "test-persona"
        memory_path.mkdir(parents=True)
        (memory_path / "conversations").mkdir()
        (memory_path / "indices").mkdir()
        return memory_path

    @pytest.fixture
    def sample_embedding(self):
        """Create a sample embedding vector (384-dim to match mock vectorizer)."""
        return np.random.randn(384).astype(np.float32)

    @pytest.fixture
    def mock_chat_config(self, temp_memory_path):
        """Create a mock ChatConfig."""
        from aim.config import ChatConfig

        return ChatConfig(
            memory_path=str(temp_memory_path.parent),
            persona_id="test-persona",
            embedding_model="test-model",
            embedding_device="cpu",
        )

    def test_from_config_with_skip_vectorizer(self, mock_chat_config):
        """ConversationModel.from_config() should pass skip_vectorizer to SearchIndex."""
        from aim.conversation.model import ConversationModel

        with patch("aim.conversation.index.HuggingFaceEmbedding") as mock_embedding:
            cvm = ConversationModel.from_config(mock_chat_config, skip_vectorizer=True)

            # Should NOT have loaded embedding model
            mock_embedding.assert_not_called()
            assert cvm.index.vectorizer is None

    def test_from_config_without_skip_vectorizer(self, mock_chat_config):
        """ConversationModel.from_config() should load vectorizer by default."""
        from aim.conversation.model import ConversationModel

        with patch("aim.conversation.index.HuggingFaceEmbedding") as mock_embedding:
            mock_embedding.return_value = MagicMock()

            cvm = ConversationModel.from_config(mock_chat_config, skip_vectorizer=False)

            # Should have loaded embedding model
            mock_embedding.assert_called_once()
            assert cvm.index.vectorizer is not None

    def test_query_with_precomputed_embedding(self, mock_chat_config, sample_embedding):
        """query() should work with pre-computed query_embedding when skip_vectorizer=True."""
        from aim.conversation.model import ConversationModel
        from aim.conversation.message import ConversationMessage
        from aim.constants import DOC_CONVERSATION, LISTENER_ALL
        import time

        with patch("aim.conversation.embedding.HuggingFaceEmbedding") as mock_embedding_class:
            # Create proper mock vectorizer that returns numpy arrays
            mock_vectorizer = MagicMock()
            mock_vectorizer.return_value = np.random.randn(384).astype(np.float32)
            mock_embedding_class.return_value = mock_vectorizer

            cvm = ConversationModel.from_config(mock_chat_config, skip_vectorizer=True)

            # Load vectorizer before inserting documents (Phase 2 requirement)
            cvm.load_vectorizer()

            # Insert a document with pre-computed embedding
            message = ConversationMessage(
                doc_id="test-doc-001",
                content="Test content for query testing",
                conversation_id="test-convo",
                user_id="test-user",
                persona_id="test-persona",
                document_type=DOC_CONVERSATION,
                speaker_id="test-user",
                listener_id=LISTENER_ALL,
                role="user",
                timestamp=int(time.time()),
                sequence_no=1,
                branch=0,
            )
            cvm.insert(message, embedding=sample_embedding)

            # Release vectorizer after batch writes
            cvm.release_vectorizer()

            # Query with pre-computed embedding should work
            results = cvm.query(
                query_texts=["test"],
                top_n=10,
                query_embedding=sample_embedding,
            )

            # Should return results (may be empty if BM25 doesn't match, but no error)
            assert results is not None

    def test_query_without_embedding_uses_bm25_only(self, mock_chat_config, sample_embedding, caplog):
        """query() without query_embedding should skip FAISS reranking when skip_vectorizer=True."""
        from aim.conversation.model import ConversationModel
        from aim.conversation.message import ConversationMessage
        from aim.constants import DOC_CONVERSATION, LISTENER_ALL
        import time
        import logging

        with patch("aim.conversation.embedding.HuggingFaceEmbedding") as mock_embedding_class:
            # Create proper mock vectorizer that returns numpy arrays
            mock_vectorizer = MagicMock()
            mock_vectorizer.return_value = np.random.randn(384).astype(np.float32)
            mock_embedding_class.return_value = mock_vectorizer

            cvm = ConversationModel.from_config(mock_chat_config, skip_vectorizer=True)

            # Load vectorizer before inserting documents (Phase 2 requirement)
            cvm.load_vectorizer()

            # Insert a document with pre-computed embedding
            message = ConversationMessage(
                doc_id="test-doc-001",
                content="Test content for query testing",
                conversation_id="test-convo",
                user_id="test-user",
                persona_id="test-persona",
                document_type=DOC_CONVERSATION,
                speaker_id="test-user",
                listener_id=LISTENER_ALL,
                role="user",
                timestamp=int(time.time()),
                sequence_no=1,
                branch=0,
            )
            cvm.insert(message, embedding=sample_embedding)

            # Release vectorizer after batch writes
            cvm.release_vectorizer()

            # Query without pre-computed embedding - should use BM25 only
            with caplog.at_level(logging.DEBUG):
                results = cvm.query(
                    query_texts=["test content"],
                    top_n=10,
                    query_embedding=None,  # No pre-computed embedding
                )

            # Should return results with neutral rerank score
            assert results is not None
            # Check that we logged skipping FAISS
            assert any(
                "skip_vectorizer=True" in record.message or "Skipping FAISS" in record.message
                for record in caplog.records
            )


class TestChatConfigSkipVectorizer:
    """Tests for ChatConfig.skip_vectorizer field."""

    def test_skip_vectorizer_default_false(self):
        """ChatConfig.skip_vectorizer should default to False."""
        from aim.config import ChatConfig

        config = ChatConfig()
        assert config.skip_vectorizer is False

    def test_skip_vectorizer_can_be_set_true(self):
        """ChatConfig.skip_vectorizer can be set to True."""
        from aim.config import ChatConfig

        config = ChatConfig(skip_vectorizer=True)
        assert config.skip_vectorizer is True

    def test_skip_vectorizer_in_to_dict(self):
        """ChatConfig.to_dict() should include skip_vectorizer."""
        from aim.config import ChatConfig

        config = ChatConfig(skip_vectorizer=True)
        config_dict = config.to_dict()
        assert "skip_vectorizer" in config_dict
        assert config_dict["skip_vectorizer"] is True
