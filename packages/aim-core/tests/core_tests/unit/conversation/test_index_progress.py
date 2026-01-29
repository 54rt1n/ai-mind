# tests/core_tests/unit/conversation/test_index_progress.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

"""Tests for tqdm progress tracking in SearchIndex."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from aim.conversation.index import SearchIndex
from aim.constants import CHUNK_LEVEL_FULL, CHUNK_LEVEL_768, CHUNK_LEVEL_256


@pytest.fixture
def temp_index_dir(tmp_path):
    """Create a temporary index directory."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    return index_dir


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    docs = []
    for i in range(5):
        doc = {
            "doc_id": f"doc_{i}",
            "content": f"This is test document {i}. " * 50,  # Make content long enough to chunk
            "conversation_id": f"conv_{i}",
            "user_id": "test_user",
            "persona_id": "test_persona",
            "role": "user",
            "document_type": "conversation",
            "timestamp": 1700000000 + i,
            "sequence_no": i,
            "branch": 0
        }
        docs.append(doc)
    return docs


@pytest.fixture
def mock_vectorizer():
    """Create a mock vectorizer that returns fake embeddings."""
    vectorizer = MagicMock()

    def mock_call(text):
        # Return a fake embedding vector
        return np.random.rand(512).astype(np.float32)

    def mock_transform(texts):
        # Return fake embeddings for batch
        return [np.random.rand(512).astype(np.float32) for _ in texts]

    vectorizer.side_effect = mock_call
    vectorizer.transform.side_effect = mock_transform
    return vectorizer


class TestExpandDocumentProgress:
    """Tests for document expansion progress tracking."""

    def test_expand_document_entries_created(self, temp_index_dir, sample_documents, mock_vectorizer):
        """Test that document expansion creates entries at all chunk levels."""
        with patch('aim.conversation.index.HuggingFaceEmbedding', return_value=mock_vectorizer):
            index = SearchIndex(
                index_path=temp_index_dir,
                embedding_model="test-model",
                device="cpu"
            )

            doc = sample_documents[0]
            entries = index._expand_document_to_entries(doc, use_tqdm=False)

            # Should have at least: 1 full + some 768-chunks + some 256-chunks
            assert len(entries) > 1

            # Check that all chunk levels are represented
            chunk_levels = {entry['chunk_level'] for entry in entries}
            assert CHUNK_LEVEL_FULL in chunk_levels
            assert CHUNK_LEVEL_768 in chunk_levels
            assert CHUNK_LEVEL_256 in chunk_levels


class TestAddDocumentsProgress:
    """Tests for add_documents() progress tracking."""

    def test_add_documents_with_tqdm_enabled(self, temp_index_dir, sample_documents, mock_vectorizer):
        """Test that add_documents creates progress bars when use_tqdm=True."""
        with patch('aim.conversation.index.HuggingFaceEmbedding', return_value=mock_vectorizer):
            index = SearchIndex(
                index_path=temp_index_dir,
                embedding_model="test-model",
                device="cpu"
            )

            # Mock tqdm at the point of use (inside the function)
            with patch('tqdm.tqdm') as mock_tqdm:
                # Setup mocks for both progress bars
                mock_expand_progress = MagicMock()
                mock_batch_progress = MagicMock()

                # First call is for document expansion
                # Second call is for batch processing
                mock_tqdm.side_effect = [mock_expand_progress, mock_batch_progress]

                mock_expand_progress.__iter__ = MagicMock(return_value=iter(sample_documents))
                mock_batch_progress.__iter__ = MagicMock(return_value=iter(range(1)))

                index.add_documents(sample_documents, use_tqdm=True, batch_size=64)

                # Verify expansion progress bar was created
                expansion_call = mock_tqdm.call_args_list[0]
                assert expansion_call[0][0] == sample_documents  # First positional arg
                assert expansion_call[1]['desc'] == "Expanding documents to chunks"
                assert expansion_call[1]['unit'] == "doc"
                assert expansion_call[1]['position'] == 0
                assert expansion_call[1]['leave'] is True

                # Verify postfix was updated with entry count
                assert mock_expand_progress.set_postfix.called

                # Verify batch processing progress bar was created
                batch_call = mock_tqdm.call_args_list[1]
                assert batch_call[1]['desc'] == "Vectorizing and writing batches"
                assert batch_call[1]['unit'] == "batch"
                assert batch_call[1]['position'] == 0
                assert batch_call[1]['leave'] is True

    def test_add_documents_with_tqdm_disabled(self, temp_index_dir, sample_documents, mock_vectorizer):
        """Test that add_documents does not create progress bars when use_tqdm=False."""
        with patch('aim.conversation.index.HuggingFaceEmbedding', return_value=mock_vectorizer):
            index = SearchIndex(
                index_path=temp_index_dir,
                embedding_model="test-model",
                device="cpu"
            )

            with patch('tqdm.tqdm') as mock_tqdm:
                index.add_documents(sample_documents, use_tqdm=False, batch_size=64)

                # Verify tqdm was NOT called
                mock_tqdm.assert_not_called()

    def test_add_documents_postfix_updates(self, temp_index_dir, sample_documents, mock_vectorizer):
        """Test that expansion progress bar postfix shows entry count."""
        with patch('aim.conversation.index.HuggingFaceEmbedding', return_value=mock_vectorizer):
            index = SearchIndex(
                index_path=temp_index_dir,
                embedding_model="test-model",
                device="cpu"
            )

            with patch('tqdm.tqdm') as mock_tqdm:
                mock_expand_progress = MagicMock()
                mock_batch_progress = MagicMock()

                mock_tqdm.side_effect = [mock_expand_progress, mock_batch_progress]
                mock_expand_progress.__iter__ = MagicMock(return_value=iter(sample_documents))
                mock_batch_progress.__iter__ = MagicMock(return_value=iter(range(1)))

                index.add_documents(sample_documents, use_tqdm=True, batch_size=64)

                # Verify postfix was called for each document
                assert mock_expand_progress.set_postfix.call_count == len(sample_documents)

                # Check that postfix shows increasing entry count
                first_call = mock_expand_progress.set_postfix.call_args_list[0]
                last_call = mock_expand_progress.set_postfix.call_args_list[-1]

                # Last call should have more entries than first
                assert last_call[0][0]['total_entries'] > first_call[0][0]['total_entries']

    def test_add_documents_batch_progress(self, temp_index_dir, sample_documents, mock_vectorizer):
        """Test that batch processing progress bar shows correct number of batches."""
        with patch('aim.conversation.index.HuggingFaceEmbedding', return_value=mock_vectorizer):
            index = SearchIndex(
                index_path=temp_index_dir,
                embedding_model="test-model",
                device="cpu"
            )

            # Use small batch size to create multiple batches
            batch_size = 2

            with patch('tqdm.tqdm') as mock_tqdm:
                mock_expand_progress = MagicMock()
                mock_batch_progress = MagicMock()

                # Collect entry count from expansion phase
                all_entries = []
                for doc in sample_documents:
                    entries = index._expand_document_to_entries(doc, use_tqdm=False)
                    all_entries.extend(entries)

                num_entries = len(all_entries)
                expected_batches = (num_entries + batch_size - 1) // batch_size

                mock_tqdm.side_effect = [mock_expand_progress, mock_batch_progress]
                mock_expand_progress.__iter__ = MagicMock(return_value=iter(sample_documents))
                mock_batch_progress.__iter__ = MagicMock(return_value=iter(range(expected_batches)))

                index.add_documents(sample_documents, use_tqdm=True, batch_size=batch_size)

                # Verify batch progress bar was created with correct total
                batch_call = mock_tqdm.call_args_list[1]
                assert batch_call[1]['total'] == expected_batches


class TestRebuildProgress:
    """Tests for rebuild() progress tracking."""

    def test_rebuild_with_tqdm_enabled(self, temp_index_dir, sample_documents, mock_vectorizer):
        """Test that rebuild uses progress bars."""
        with patch('aim.conversation.index.HuggingFaceEmbedding', return_value=mock_vectorizer):
            index = SearchIndex(
                index_path=temp_index_dir,
                embedding_model="test-model",
                device="cpu"
            )

            # Mock add_documents to verify it's called with use_tqdm=True
            with patch.object(index, 'add_documents') as mock_add_docs:
                index.rebuild(sample_documents, use_tqdm=True)

                # Verify add_documents was called with use_tqdm=True
                mock_add_docs.assert_called_once()
                call_args = mock_add_docs.call_args
                assert call_args[0][0] == sample_documents
                assert call_args[1]['use_tqdm'] is True

    def test_rebuild_with_tqdm_disabled(self, temp_index_dir, sample_documents, mock_vectorizer):
        """Test that rebuild respects use_tqdm=False."""
        with patch('aim.conversation.index.HuggingFaceEmbedding', return_value=mock_vectorizer):
            index = SearchIndex(
                index_path=temp_index_dir,
                embedding_model="test-model",
                device="cpu"
            )

            with patch.object(index, 'add_documents') as mock_add_docs:
                index.rebuild(sample_documents, use_tqdm=False)

                # Verify add_documents was called with use_tqdm=False
                mock_add_docs.assert_called_once()
                call_args = mock_add_docs.call_args
                assert call_args[1]['use_tqdm'] is False


class TestIncrementalUpdateProgress:
    """Tests for incremental_update() progress tracking."""

    def test_incremental_update_with_tqdm_enabled(self, temp_index_dir, sample_documents, mock_vectorizer):
        """Test that incremental_update uses progress bars."""
        with patch('aim.conversation.index.HuggingFaceEmbedding', return_value=mock_vectorizer):
            index = SearchIndex(
                index_path=temp_index_dir,
                embedding_model="test-model",
                device="cpu"
            )

            # First add some documents to create initial state
            index.add_documents(sample_documents[:3], use_tqdm=False, batch_size=64)

            # Now do incremental update with all documents (including existing ones)
            with patch('tqdm.tqdm') as mock_tqdm:
                mock_progress = MagicMock()
                mock_progress.__iter__ = MagicMock(return_value=iter([]))
                mock_tqdm.return_value = mock_progress

                # Use all documents - 3 existing + 2 new
                added, updated, deleted = index.incremental_update(
                    sample_documents,
                    use_tqdm=True,
                    batch_size=64
                )

                # Should have added only the new documents
                assert added == 2  # Only the last 2 are new
                assert updated == 0  # No changes to existing
                assert deleted == 0  # No deleted documents

    def test_incremental_update_with_tqdm_disabled(self, temp_index_dir, sample_documents, mock_vectorizer):
        """Test that incremental_update respects use_tqdm=False."""
        with patch('aim.conversation.index.HuggingFaceEmbedding', return_value=mock_vectorizer):
            index = SearchIndex(
                index_path=temp_index_dir,
                embedding_model="test-model",
                device="cpu"
            )

            # First add some documents
            index.add_documents(sample_documents[:3], use_tqdm=False, batch_size=64)

            with patch('tqdm.tqdm') as mock_tqdm:
                new_docs = sample_documents[3:]
                added, updated, deleted = index.incremental_update(
                    new_docs,
                    use_tqdm=False,
                    batch_size=64
                )

                # Verify tqdm was NOT called
                mock_tqdm.assert_not_called()


class TestProgressParameterDefaults:
    """Tests for default parameter values."""

    def test_add_documents_default_use_tqdm(self, temp_index_dir, sample_documents, mock_vectorizer):
        """Test that add_documents defaults to use_tqdm=False."""
        with patch('aim.conversation.index.HuggingFaceEmbedding', return_value=mock_vectorizer):
            index = SearchIndex(
                index_path=temp_index_dir,
                embedding_model="test-model",
                device="cpu"
            )

            with patch('tqdm.tqdm') as mock_tqdm:
                # Call without specifying use_tqdm
                index.add_documents(sample_documents, batch_size=64)

                # Should not use tqdm by default
                mock_tqdm.assert_not_called()

    def test_rebuild_default_use_tqdm(self, temp_index_dir, sample_documents, mock_vectorizer):
        """Test that rebuild defaults to use_tqdm=True."""
        with patch('aim.conversation.index.HuggingFaceEmbedding', return_value=mock_vectorizer):
            index = SearchIndex(
                index_path=temp_index_dir,
                embedding_model="test-model",
                device="cpu"
            )

            # Mock add_documents before calling rebuild
            with patch.object(index, 'add_documents') as mock_add_docs:
                # Mock shutil.rmtree to avoid cleanup issues (patch at import location)
                with patch('shutil.rmtree'):
                    # Call without specifying use_tqdm
                    index.rebuild(sample_documents)

                    # Should use tqdm by default (use_tqdm=True)
                    call_args = mock_add_docs.call_args
                    assert call_args[1]['use_tqdm'] is True

    def test_incremental_update_default_use_tqdm(self, temp_index_dir, sample_documents, mock_vectorizer):
        """Test that incremental_update defaults to use_tqdm=False."""
        with patch('aim.conversation.index.HuggingFaceEmbedding', return_value=mock_vectorizer):
            index = SearchIndex(
                index_path=temp_index_dir,
                embedding_model="test-model",
                device="cpu"
            )

            with patch('tqdm.tqdm') as mock_tqdm:
                # Call without specifying use_tqdm
                index.incremental_update(sample_documents, batch_size=64)

                # Should not use tqdm by default
                mock_tqdm.assert_not_called()
