# tests/unit/conversation/test_conversation_loader.py
import pytest
import os
import json
import logging
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock, call

from aim.conversation.loader import ConversationLoader
from aim.conversation.message import ConversationMessage, ROLE_USER

TEST_CONV_DIR = "test_conversations_dir"
TEST_CONV_ID = "conv_test_loader_123"
TEST_FILE_PATH = os.path.join(TEST_CONV_DIR, f"{TEST_CONV_ID}.jsonl")

# Sample JSON data for messages
MSG_VALID_WITH_TS = {
    "doc_id": "loader_doc_1", "document_type": "conversation", "user_id": "u1", 
    "persona_id": "p1", "conversation_id": TEST_CONV_ID, "branch": 0, 
    "sequence_no": 0, "role": ROLE_USER, "content": "Valid msg 1", "timestamp": 1700000000
}
MSG_VALID_WITHOUT_TS = {
    "doc_id": "loader_doc_2", "document_type": "conversation", "user_id": "u1", 
    "persona_id": "p1", "conversation_id": TEST_CONV_ID, "branch": 0, 
    "sequence_no": 1, "role": ROLE_USER, "content": "Valid msg 2, no timestamp"
    # No timestamp
}
MSG_VALID_WITH_TS_2 = {
    "doc_id": "loader_doc_3", "document_type": "conversation", "user_id": "u1", 
    "persona_id": "p1", "conversation_id": TEST_CONV_ID, "branch": 0, 
    "sequence_no": 2, "role": ROLE_USER, "content": "Valid msg 3", "timestamp": 1700000010
}

# Simulate file content
VALID_JSONL_CONTENT = f"{json.dumps(MSG_VALID_WITH_TS)}\n{json.dumps(MSG_VALID_WITHOUT_TS)}\n{json.dumps(MSG_VALID_WITH_TS_2)}\n"

MIXED_JSONL_CONTENT = (
    f"{json.dumps(MSG_VALID_WITH_TS)}\n"          # Valid
    f"this is not valid json\n"                 # Invalid JSON
    f"{json.dumps(MSG_VALID_WITHOUT_TS)}\n"   # Valid without TS
    f"\n"                                       # Empty line
    f'{json.dumps(MSG_VALID_WITH_TS_2)}\n'       # Valid
)

INVALID_JSON_ONLY_CONTENT = "not json line 1\nnot json line 2\n"
EMPTY_FILE_CONTENT = ""

@pytest.fixture
def loader():
    "Fixture for a ConversationLoader instance."""
    return ConversationLoader(conversations_dir=TEST_CONV_DIR)

# -------------------------------------------------
# Tests for load_or_new
# -------------------------------------------------

@patch('aim.conversation.loader.Path.exists')
@patch('aim.conversation.loader.open', new_callable=mock_open, read_data=VALID_JSONL_CONTENT)
def test_load_or_new_valid_file(mock_file, mock_exists, loader):
    """Test loading a conversation from a file with only valid JSON lines."""
    mock_exists.return_value = True
    
    messages = loader.load_or_new(TEST_CONV_ID)
    
    assert mock_exists.call_count == 2
    mock_file.assert_called_once_with(loader.conversations_dir / f"{TEST_CONV_ID}.jsonl", 'r')
    
    assert len(messages) == 3
    assert isinstance(messages[0], ConversationMessage)
    assert messages[0].doc_id == MSG_VALID_WITH_TS["doc_id"]
    assert messages[0].timestamp == MSG_VALID_WITH_TS["timestamp"]
    assert messages[1].doc_id == MSG_VALID_WITHOUT_TS["doc_id"]
    assert messages[1].timestamp == 0 # Defaulted
    assert messages[2].doc_id == MSG_VALID_WITH_TS_2["doc_id"]
    assert messages[2].timestamp == MSG_VALID_WITH_TS_2["timestamp"]

@patch('aim.conversation.loader.Path.exists')
@patch('aim.conversation.loader.open', new_callable=mock_open, read_data=MIXED_JSONL_CONTENT)
def test_load_or_new_mixed_file(mock_file, mock_exists, loader, caplog):
    """Test loading from a file with valid, invalid JSON, and empty lines."""
    mock_exists.return_value = True
    
    with caplog.at_level(logging.WARNING):
        messages = loader.load_or_new(TEST_CONV_ID)
        
    assert mock_exists.call_count == 2
    mock_file.assert_called_once_with(loader.conversations_dir / f"{TEST_CONV_ID}.jsonl", 'r')
    
    # Should load the 3 valid messages and skip the invalid/empty ones
    assert len(messages) == 3 
    assert messages[0].doc_id == MSG_VALID_WITH_TS["doc_id"]
    assert messages[1].doc_id == MSG_VALID_WITHOUT_TS["doc_id"]
    assert messages[2].doc_id == MSG_VALID_WITH_TS_2["doc_id"]
    
    # Check that a warning was logged for the invalid JSON line
    assert len(caplog.records) >= 1
    assert "JSON decode error" in caplog.text
    assert "line 1" in caplog.text # Line numbers are 0-indexed in the loop
    # Check warning for missing timestamp on MSG_VALID_WITHOUT_TS
    assert f"Timestamp missing in message data (doc_id: {MSG_VALID_WITHOUT_TS['doc_id']})" in caplog.text

@patch('aim.conversation.loader.Path.exists')
@patch('aim.conversation.loader.open', new_callable=mock_open, read_data=INVALID_JSON_ONLY_CONTENT)
def test_load_or_new_invalid_only_file(mock_file, mock_exists, loader, caplog):
    """Test loading from a file with only invalid JSON lines."""
    mock_exists.return_value = True
    
    with caplog.at_level(logging.WARNING):
        messages = loader.load_or_new(TEST_CONV_ID)
        
    assert mock_exists.call_count == 2
    mock_file.assert_called_once_with(loader.conversations_dir / f"{TEST_CONV_ID}.jsonl", 'r')
    
    assert len(messages) == 0
    assert len(caplog.records) == 2 # One warning per invalid line
    assert "JSON decode error" in caplog.text

@patch('aim.conversation.loader.Path.exists')
@patch('aim.conversation.loader.open', new_callable=mock_open, read_data=EMPTY_FILE_CONTENT)
def test_load_or_new_empty_file(mock_file, mock_exists, loader):
    """Test loading from an empty file."""
    mock_exists.return_value = True
    
    messages = loader.load_or_new(TEST_CONV_ID)
        
    assert mock_exists.call_count == 2
    mock_file.assert_called_once_with(loader.conversations_dir / f"{TEST_CONV_ID}.jsonl", 'r')
    assert len(messages) == 0

@patch('aim.conversation.loader.Path.exists')
@patch('aim.conversation.loader.open', new_callable=mock_open)
def test_load_or_new_file_not_exist(mock_file, mock_exists, loader):
    """Test loading when the conversation file does not exist."""
    mock_exists.return_value = False
    
    messages = loader.load_or_new(TEST_CONV_ID)
    
    assert mock_exists.call_count == 1
    mock_file.assert_not_called() # Should not attempt to open non-existent file
    assert len(messages) == 0

# -------------------------------------------------
# Tests for load_all (to be added)
# -------------------------------------------------

# TODO: Add tests for load_all, mocking Path.glob and multiple file reads 