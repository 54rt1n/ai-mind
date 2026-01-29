# tests/core_tests/unit/conversation/test_loader_progress.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

"""Tests for tqdm progress tracking in ConversationLoader."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from aim.conversation.loader import ConversationLoader
from aim.conversation.message import ConversationMessage, ROLE_USER, ROLE_ASSISTANT


@pytest.fixture
def sample_message_data():
    """Generate sample message data."""
    return {
        "doc_id": "doc_1",
        "document_type": "conversation",
        "user_id": "test_user",
        "persona_id": "test_persona",
        "conversation_id": "conv_123",
        "branch": 0,
        "sequence_no": 0,
        "role": ROLE_USER,
        "content": "Test message",
        "timestamp": 1700000000
    }


@pytest.fixture
def temp_conversations_dir(tmp_path):
    """Create a temporary conversations directory with test files."""
    conv_dir = tmp_path / "conversations"
    conv_dir.mkdir()
    return conv_dir


def create_jsonl_file(path: Path, num_messages: int, conversation_id: str = "conv_test"):
    """Create a JSONL file with specified number of messages."""
    with open(path, 'w') as f:
        for i in range(num_messages):
            msg = {
                "doc_id": f"doc_{i}",
                "document_type": "conversation",
                "user_id": "test_user",
                "persona_id": "test_persona",
                "conversation_id": conversation_id,
                "branch": 0,
                "sequence_no": i,
                "role": ROLE_USER if i % 2 == 0 else ROLE_ASSISTANT,
                "content": f"Message {i}",
                "timestamp": 1700000000 + i
            }
            json.dump(msg, f)
            f.write('\n')


class TestLoadAllProgress:
    """Tests for load_all() progress tracking."""

    def test_load_all_with_tqdm_enabled(self, temp_conversations_dir):
        """Test that load_all creates progress bar when use_tqdm=True."""
        # Create test files
        create_jsonl_file(temp_conversations_dir / "conv1.jsonl", 10, "conv1")
        create_jsonl_file(temp_conversations_dir / "conv2.jsonl", 20, "conv2")

        loader = ConversationLoader(str(temp_conversations_dir))

        with patch('aim.conversation.loader.tqdm') as mock_tqdm:
            # Setup mock to return an iterable that behaves like tqdm
            mock_progress = MagicMock()
            mock_progress.__iter__ = MagicMock(
                return_value=iter(list(temp_conversations_dir.glob("*.jsonl")))
            )
            mock_tqdm.return_value = mock_progress

            messages = loader.load_all(use_tqdm=True)

            # Verify tqdm was called with correct parameters
            mock_tqdm.assert_called_once()
            call_args = mock_tqdm.call_args
            assert call_args[1]['desc'] == "Loading conversation files"
            assert call_args[1]['unit'] == "file"
            assert call_args[1]['position'] == 0
            assert call_args[1]['leave'] is True

            # Verify messages were loaded
            assert len(messages) == 30  # 10 + 20

            # Verify postfix was updated
            assert mock_progress.set_postfix.called

    def test_load_all_with_tqdm_disabled(self, temp_conversations_dir):
        """Test that load_all does not create progress bar when use_tqdm=False."""
        # Create test files
        create_jsonl_file(temp_conversations_dir / "conv1.jsonl", 10, "conv1")
        create_jsonl_file(temp_conversations_dir / "conv2.jsonl", 20, "conv2")

        loader = ConversationLoader(str(temp_conversations_dir))

        with patch('aim.conversation.loader.tqdm') as mock_tqdm:
            messages = loader.load_all(use_tqdm=False)

            # Verify tqdm was NOT called
            mock_tqdm.assert_not_called()

            # Verify messages were still loaded correctly
            assert len(messages) == 30

    def test_load_all_postfix_updates(self, temp_conversations_dir):
        """Test that postfix updates with total message count."""
        # Create test files with different sizes
        create_jsonl_file(temp_conversations_dir / "conv1.jsonl", 5, "conv1")
        create_jsonl_file(temp_conversations_dir / "conv2.jsonl", 15, "conv2")
        create_jsonl_file(temp_conversations_dir / "conv3.jsonl", 10, "conv3")

        loader = ConversationLoader(str(temp_conversations_dir))

        with patch('aim.conversation.loader.tqdm') as mock_tqdm:
            mock_progress = MagicMock()
            files = list(temp_conversations_dir.glob("*.jsonl"))
            mock_progress.__iter__ = MagicMock(return_value=iter(files))
            mock_tqdm.return_value = mock_progress

            messages = loader.load_all(use_tqdm=True)

            # Verify postfix was called for each file
            assert mock_progress.set_postfix.call_count == len(files)

            # Check the final postfix call has the total count
            final_call = mock_progress.set_postfix.call_args_list[-1]
            assert final_call[0][0]['total_messages'] == 30

    def test_load_all_empty_directory(self, temp_conversations_dir):
        """Test load_all with empty directory."""
        loader = ConversationLoader(str(temp_conversations_dir))

        with patch('aim.conversation.loader.tqdm') as mock_tqdm:
            mock_progress = MagicMock()
            mock_progress.__iter__ = MagicMock(return_value=iter([]))
            mock_tqdm.return_value = mock_progress

            messages = loader.load_all(use_tqdm=True)

            assert len(messages) == 0
            # Tqdm should still be called even with no files
            mock_tqdm.assert_called_once()


class TestLoadFileProgress:
    """Tests for load_file() progress tracking."""

    def test_load_file_small_file_no_progress(self, temp_conversations_dir):
        """Test that load_file does NOT show progress for files <=100 messages."""
        file_path = temp_conversations_dir / "small.jsonl"
        create_jsonl_file(file_path, 50, "conv_small")

        loader = ConversationLoader(str(temp_conversations_dir))

        with patch('aim.conversation.loader.tqdm') as mock_tqdm:
            messages = loader.load_file(file_path, use_tqdm=True)

            # Should not create progress bar for small files
            mock_tqdm.assert_not_called()
            assert len(messages) == 50

    def test_load_file_large_file_with_progress(self, temp_conversations_dir):
        """Test that load_file shows progress for files >100 messages."""
        file_path = temp_conversations_dir / "large.jsonl"
        create_jsonl_file(file_path, 150, "conv_large")

        loader = ConversationLoader(str(temp_conversations_dir))

        with patch('aim.conversation.loader.tqdm') as mock_tqdm:
            # Setup mock to return an iterable
            mock_progress = MagicMock()
            # Read the actual file to get line count
            with open(file_path, 'r') as f:
                lines = list(enumerate(f, 1))
            mock_progress.__iter__ = MagicMock(return_value=iter(lines))
            mock_tqdm.return_value = mock_progress

            messages = loader.load_file(file_path, use_tqdm=True)

            # Verify tqdm was called with correct parameters
            mock_tqdm.assert_called_once()
            call_args = mock_tqdm.call_args
            assert call_args[1]['desc'] == f"  {file_path.name}"
            assert call_args[1]['unit'] == "msg"
            assert call_args[1]['position'] == 1
            assert call_args[1]['leave'] is False
            assert call_args[1]['total'] == 150

    def test_load_file_exactly_100_messages_no_progress(self, temp_conversations_dir):
        """Test boundary condition: exactly 100 messages should NOT show progress."""
        file_path = temp_conversations_dir / "boundary.jsonl"
        create_jsonl_file(file_path, 100, "conv_boundary")

        loader = ConversationLoader(str(temp_conversations_dir))

        with patch('aim.conversation.loader.tqdm') as mock_tqdm:
            messages = loader.load_file(file_path, use_tqdm=True)

            # Should not create progress bar for exactly 100 messages
            mock_tqdm.assert_not_called()
            assert len(messages) == 100

    def test_load_file_101_messages_with_progress(self, temp_conversations_dir):
        """Test boundary condition: 101 messages should show progress."""
        file_path = temp_conversations_dir / "boundary_plus.jsonl"
        create_jsonl_file(file_path, 101, "conv_boundary_plus")

        loader = ConversationLoader(str(temp_conversations_dir))

        with patch('aim.conversation.loader.tqdm') as mock_tqdm:
            mock_progress = MagicMock()
            with open(file_path, 'r') as f:
                lines = list(enumerate(f, 1))
            mock_progress.__iter__ = MagicMock(return_value=iter(lines))
            mock_tqdm.return_value = mock_progress

            messages = loader.load_file(file_path, use_tqdm=True)

            # Should create progress bar for 101 messages
            mock_tqdm.assert_called_once()
            assert len(messages) == 101

    def test_load_file_use_tqdm_false(self, temp_conversations_dir):
        """Test that load_file does not show progress when use_tqdm=False."""
        file_path = temp_conversations_dir / "large.jsonl"
        create_jsonl_file(file_path, 200, "conv_large")

        loader = ConversationLoader(str(temp_conversations_dir))

        with patch('aim.conversation.loader.tqdm') as mock_tqdm:
            messages = loader.load_file(file_path, use_tqdm=False)

            # Should not create progress bar even for large file
            mock_tqdm.assert_not_called()
            assert len(messages) == 200

    def test_load_file_nonexistent_file(self, temp_conversations_dir):
        """Test that load_file raises FileNotFoundError for nonexistent file."""
        file_path = temp_conversations_dir / "nonexistent.jsonl"
        loader = ConversationLoader(str(temp_conversations_dir))

        with pytest.raises(FileNotFoundError, match="not found"):
            loader.load_file(file_path, use_tqdm=True)


class TestIntegrationProgress:
    """Integration tests for progress tracking across load_all and load_file."""

    def test_nested_progress_bars(self, temp_conversations_dir):
        """Test that nested progress bars use correct position parameter."""
        # Create files with varying sizes
        create_jsonl_file(temp_conversations_dir / "small.jsonl", 50, "conv_small")
        create_jsonl_file(temp_conversations_dir / "large1.jsonl", 150, "conv_large1")
        create_jsonl_file(temp_conversations_dir / "large2.jsonl", 200, "conv_large2")

        loader = ConversationLoader(str(temp_conversations_dir))

        with patch('aim.conversation.loader.tqdm') as mock_tqdm:
            tqdm_calls = []

            def tqdm_side_effect(*args, **kwargs):
                mock_progress = MagicMock()
                if 'position' in kwargs:
                    if kwargs['position'] == 0:
                        # File-level progress bar
                        files = list(temp_conversations_dir.glob("*.jsonl"))
                        mock_progress.__iter__ = MagicMock(return_value=iter(files))
                    elif kwargs['position'] == 1:
                        # Message-level progress bar - need to read the file
                        # We'll track which file is being processed
                        tqdm_calls.append(kwargs)
                        # Return empty iterator for simplicity
                        mock_progress.__iter__ = MagicMock(return_value=iter([]))
                return mock_progress

            mock_tqdm.side_effect = tqdm_side_effect

            messages = loader.load_all(use_tqdm=True)

            # Verify we got calls at both positions
            positions = [call[1]['position'] for call in mock_tqdm.call_args_list]
            assert 0 in positions  # File-level progress
            # Message-level progress should only appear for files >100 messages

    def test_load_all_propagates_use_tqdm(self, temp_conversations_dir):
        """Test that load_all correctly propagates use_tqdm to load_file."""
        create_jsonl_file(temp_conversations_dir / "file1.jsonl", 150, "conv1")

        loader = ConversationLoader(str(temp_conversations_dir))

        with patch.object(loader, 'load_file') as mock_load_file:
            mock_load_file.return_value = []

            loader.load_all(use_tqdm=True)

            # Verify load_file was called with use_tqdm=True
            assert mock_load_file.called
            call_args = mock_load_file.call_args
            assert call_args[1]['use_tqdm'] is True

    def test_load_all_no_progress_propagation(self, temp_conversations_dir):
        """Test that load_all with use_tqdm=False propagates to load_file."""
        create_jsonl_file(temp_conversations_dir / "file1.jsonl", 150, "conv1")

        loader = ConversationLoader(str(temp_conversations_dir))

        with patch.object(loader, 'load_file') as mock_load_file:
            mock_load_file.return_value = []

            loader.load_all(use_tqdm=False)

            # Verify load_file was called with use_tqdm=False
            assert mock_load_file.called
            call_args = mock_load_file.call_args
            assert call_args[1]['use_tqdm'] is False
