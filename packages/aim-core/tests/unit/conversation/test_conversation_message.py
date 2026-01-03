# tests/unit/conversation/test_conversation_message.py
import pytest
import time
import logging
from aim.conversation.message import ConversationMessage, ROLE_USER

# Assuming conftest.py is in the same directory or a parent directory recognized by pytest
# from .conftest import raw_message_data_with_timestamp, raw_message_data_without_timestamp, minimal_message_data_keys, TEST_DOC_ID_1

def test_message_from_dict_with_timestamp(raw_message_data_with_timestamp):
    """Test that a ConversationMessage can be created from a dict with a timestamp."""
    current_time = raw_message_data_with_timestamp["timestamp"] # Get the timestamp used in fixture
    message = ConversationMessage.from_dict(raw_message_data_with_timestamp)
    assert message.doc_id == raw_message_data_with_timestamp["doc_id"]
    assert message.content == raw_message_data_with_timestamp["content"]
    assert message.timestamp == current_time
    assert message.think == raw_message_data_with_timestamp["think"]
    assert message.speaker_id == raw_message_data_with_timestamp["speaker_id"]
    assert message.listener_id == raw_message_data_with_timestamp["listener_id"]

def test_message_from_dict_without_timestamp(raw_message_data_without_timestamp, caplog):
    """Test that a ConversationMessage defaults timestamp to 0 and logs a warning if it's missing."""
    with caplog.at_level(logging.WARNING):
        message = ConversationMessage.from_dict(raw_message_data_without_timestamp)
    
    assert message.doc_id == raw_message_data_without_timestamp["doc_id"]
    assert message.timestamp == 0
    assert len(caplog.records) == 1
    assert "Timestamp missing in message data" in caplog.records[0].message
    assert f"doc_id: {raw_message_data_without_timestamp['doc_id']}" in caplog.records[0].message

def test_message_to_dict_serialization(message_obj_with_timestamp, raw_message_data_with_timestamp):
    """Test that to_dict correctly serializes the message, including the timestamp."""
    serialized_data = message_obj_with_timestamp.to_dict()
    assert serialized_data["doc_id"] == raw_message_data_with_timestamp["doc_id"]
    assert serialized_data["content"] == raw_message_data_with_timestamp["content"]
    assert serialized_data["timestamp"] == raw_message_data_with_timestamp["timestamp"]
    assert serialized_data["think"] == raw_message_data_with_timestamp["think"]
    assert serialized_data["speaker_id"] == raw_message_data_with_timestamp["speaker_id"]
    assert serialized_data["listener_id"] == raw_message_data_with_timestamp["listener_id"]
    # Check for other essential fields that should be present
    assert "role" in serialized_data
    assert "user_id" in serialized_data
    assert "persona_id" in serialized_data
    assert "conversation_id" in serialized_data

def test_message_to_dict_serialization_missing_optional_fields(minimal_message_data_keys):
    """Test to_dict handles missing optional fields by using defaults."""
    # Add timestamp as it's required by from_dict to not default to 0 for this test's purpose
    minimal_data_with_ts = minimal_message_data_keys.copy()
    minimal_data_with_ts["timestamp"] = int(time.time())

    message = ConversationMessage.from_dict(minimal_data_with_ts)
    serialized_data = message.to_dict()

    assert serialized_data["doc_id"] == minimal_data_with_ts["doc_id"]
    assert serialized_data["timestamp"] == minimal_data_with_ts["timestamp"]
    # Check that default values are present for optional fields not in minimal_message_data_keys
    assert serialized_data["think"] is None
    assert serialized_data["emotion_a"] is None
    assert serialized_data["sentiment_v"] == 0.0
    assert serialized_data["speaker_id"] == minimal_data_with_ts["user_id"] # Default for user role
    assert serialized_data["listener_id"] == minimal_data_with_ts["persona_id"] # Default for user role


def test_message_round_trip_with_timestamp(raw_message_data_with_timestamp):
    """Test from_dict -> to_dict -> from_dict data integrity with timestamp."""
    message1 = ConversationMessage.from_dict(raw_message_data_with_timestamp)
    dict1 = message1.to_dict()
    message2 = ConversationMessage.from_dict(dict1)
    dict2 = message2.to_dict()
    assert dict1 == dict2
    assert message2.timestamp == raw_message_data_with_timestamp["timestamp"]

def test_message_round_trip_without_timestamp(raw_message_data_without_timestamp):
    """Test from_dict -> to_dict -> from_dict data integrity when timestamp is initially missing."""
    message1 = ConversationMessage.from_dict(raw_message_data_without_timestamp) # timestamp becomes 0
    assert message1.timestamp == 0
    dict1 = message1.to_dict()
    assert dict1["timestamp"] == 0 # Ensure to_dict writes the defaulted 0
    message2 = ConversationMessage.from_dict(dict1)
    dict2 = message2.to_dict()
    assert dict1 == dict2
    assert message2.timestamp == 0 