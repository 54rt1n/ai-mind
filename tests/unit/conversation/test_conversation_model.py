# tests/unit/conversation/test_conversation_model.py
import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, call
import pandas as pd
import time

from aim.config import ChatConfig
from aim.conversation.model import ConversationModel
from aim.conversation.message import ConversationMessage, ROLE_USER, ROLE_ASSISTANT
from aim.constants import DOC_CONVERSATION

# Use fixtures from the existing conftest.py
# Make sure it's in the python path or discoverable by pytest

TEST_PERSONA_ID = "test_persona"
TEST_MEMORY_PATH = f"memory/{TEST_PERSONA_ID}"

@pytest.fixture
def mock_chat_config():
    "Fixture for a mock ChatConfig object."""
    config = MagicMock(spec=ChatConfig)
    config.persona_id = TEST_PERSONA_ID
    config.embedding_model = "mock_embedding_model"
    config.user_timezone = "UTC"
    config.embedding_device = None
    return config

@pytest.fixture
@patch('aim.conversation.model.SearchIndex')
@patch('aim.conversation.model.ConversationLoader')
@patch('aim.conversation.model.ConversationModel.maybe_init_folders')
def conversation_model(mock_maybe_init_folders, MockLoader, MockIndex, mock_chat_config):
    """Fixture for an initialized ConversationModel with mocked dependencies."""
    # Setup mocks for SearchIndex and ConversationLoader instances if needed
    mock_index_instance = MockIndex.return_value
    mock_loader_instance = MockLoader.return_value

    # Instantiate the model using from_config which derives memory path from persona_id
    model = ConversationModel.from_config(mock_chat_config)

    # Ensure mocks were called as expected during init
    mock_maybe_init_folders.assert_called_once_with(TEST_MEMORY_PATH)
    MockIndex.assert_called_once_with(Path('.', TEST_MEMORY_PATH, 'indices'), embedding_model="mock_embedding_model", device=None)
    MockLoader.assert_called_once_with(conversations_dir=os.path.join(TEST_MEMORY_PATH, 'conversations'))

    # Attach mocks for later use if needed in tests
    model.index = mock_index_instance
    model.loader = mock_loader_instance

    return model

def test_conversation_model_init(conversation_model, mock_chat_config):
    """Test ConversationModel initialization."""
    assert conversation_model.memory_path == TEST_MEMORY_PATH
    assert conversation_model.collection_path == Path(f'./{TEST_MEMORY_PATH}/conversations')
    assert conversation_model.index is not None
    assert conversation_model.loader is not None

@patch('aim.conversation.model.Path.mkdir')
@patch('aim.conversation.model.Path.exists')
def test_maybe_init_folders(mock_exists, mock_mkdir):
    """Test that maybe_init_folders creates directories if they don't exist."""
    # Simulate folders not existing
    mock_exists.return_value = False

    ConversationModel.maybe_init_folders(TEST_MEMORY_PATH)

    assert mock_exists.call_count == 2
    assert mock_mkdir.call_count == 2

@patch('aim.conversation.model.Path.mkdir')
@patch('aim.conversation.model.Path.exists')
def test_maybe_init_folders_existing(mock_exists, mock_mkdir):
    """Test that maybe_init_folders doesn't create directories if they exist."""
    # Simulate folders existing
    mock_exists.return_value = True

    ConversationModel.maybe_init_folders(TEST_MEMORY_PATH)

    assert mock_exists.call_count == 2
    # mkdir should not be called
    mock_mkdir.assert_not_called()

def test_fix_dataframe(conversation_model):
    """Test the _fix_dataframe helper method."""
    ts = int(time.time())
    data = {
        'timestamp': [ts, ts - 3600],
        'role': [ROLE_USER, ROLE_ASSISTANT],
        'user_id': ['user1', 'user1'],
        'persona_id': ['persona1', 'persona1'],
        'doc_id': ['doc1', 'doc2'] # Add other necessary columns for ConversationMessage
    }
    # Add other minimal columns required by ConversationMessage visible columns
    # or ensure the test doesn't rely on accessing columns not provided here.
    df = pd.DataFrame(data)
    
    fixed_df = conversation_model._fix_dataframe(df.copy()) # Pass a copy
    
    assert 'date' in fixed_df.columns
    assert 'speaker' in fixed_df.columns
    assert fixed_df.loc[0, 'speaker'] == 'user1'
    assert fixed_df.loc[1, 'speaker'] == 'persona1'
    # Check date format (optional, depends on exact strftime format)
    assert isinstance(fixed_df.loc[0, 'date'], str)

# --- Tests for insert (requires more mocking) ---

@patch('aim.conversation.model.open', new_callable=mock_open)
@patch('aim.conversation.model.Path.exists')
def test_insert_new_file(mock_exists, mock_file_open, conversation_model, message_obj_with_timestamp):
    """Test insert when the conversation file does not exist."""
    mock_exists.return_value = False # Simulate file does not exist
    
    conversation_model.insert(message_obj_with_timestamp)
    
    # Check if file existence was checked
    expected_path = conversation_model.collection_path / f"{message_obj_with_timestamp.conversation_id}.jsonl"
    assert mock_exists.call_count == 2

    # Check if file was opened in write mode first (to create/clear)
    # Check if file was opened in append mode later (for the message)
    # Check if the correct data was written
    # mock_open is tricky for multiple opens/writes
    # For simplicity here, check add_document was called on index
    conversation_model.index.add_document.assert_called_once()
    # We could inspect the call args for add_document if needed
    # arg_dict = conversation_model.index.add_document.call_args[0][0]
    # assert arg_dict['doc_id'] == message_obj_with_timestamp.doc_id

@patch('aim.conversation.model.open', new_callable=mock_open)
@patch('aim.conversation.model.Path.exists')
def test_insert_existing_file(mock_exists, mock_file_open, conversation_model, message_obj_with_timestamp):
    """Test insert when the conversation file already exists."""
    mock_exists.return_value = True # Simulate file exists
    
    conversation_model.insert(message_obj_with_timestamp)
    
    # Check if file existence was checked
    expected_path = conversation_model.collection_path / f"{message_obj_with_timestamp.conversation_id}.jsonl"
    assert mock_exists.call_count == 2

    # Check if file was opened in append mode (since it exists, _append_message is called)
    # Check if the correct data was written
    # mock_open mocking verification can be complex for append vs write modes.
    # Check add_document was called
    conversation_model.index.add_document.assert_called_once()

# --- Add more tests for load, delete, update, query (with index mocking), next_id etc. --- 

