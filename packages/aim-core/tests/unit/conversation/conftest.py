# tests/unit/conversation/conftest.py
import pytest
from aim.conversation.message import ConversationMessage, ROLE_USER, ROLE_ASSISTANT
import time

TEST_USER_ID = "test_user"
TEST_PERSONA_ID = "test_persona"
TEST_CONVERSATION_ID = "conv_test_123"
TEST_DOC_ID_1 = "doc_1"
TEST_DOC_ID_2 = "doc_2"

@pytest.fixture
def minimal_message_data_keys():
    return {
        "doc_id": TEST_DOC_ID_1,
        "document_type": "conversation",
        "user_id": TEST_USER_ID,
        "persona_id": TEST_PERSONA_ID,
        "conversation_id": TEST_CONVERSATION_ID,
        "branch": 0,
        "sequence_no": 0,
        "role": ROLE_USER,
        "content": "Hello world",
    }

@pytest.fixture
def raw_message_data_with_timestamp(minimal_message_data_keys):
    current_time = int(time.time())
    data = minimal_message_data_keys.copy()
    data.update({
        "timestamp": current_time,
        "speaker_id": data["user_id"], 
        "listener_id": data["persona_id"],
        "think": "This is a thought.",
        "emotion_a": "joy",
        "reference_id": data["conversation_id"],
        "observer": "test_observer",
        "inference_model": "test_model"
    })
    return data

@pytest.fixture
def raw_message_data_without_timestamp(minimal_message_data_keys):
    data = minimal_message_data_keys.copy()
    data.update({
        "speaker_id": data["user_id"],
        "listener_id": data["persona_id"],
        "think": "This is another thought.",
        # No timestamp
    })
    return data

@pytest.fixture
def raw_message_data_with_think_tag_in_content(raw_message_data_with_timestamp):
    data = raw_message_data_with_timestamp.copy()
    data["doc_id"] = TEST_DOC_ID_2
    data["think"] = None 
    data["content"] = "<think>This is a think tag content.</think>This is the actual message content."
    return data

@pytest.fixture
def raw_message_data_with_closing_think_tag_in_content(raw_message_data_with_timestamp):
    data = raw_message_data_with_timestamp.copy()
    data["doc_id"] = "doc_3_closing_tag"
    data["think"] = None
    data["content"] = "This is a think tag content without opening.</think>This is the actual message content."
    return data


@pytest.fixture
def raw_message_data_without_think_tag_in_content(raw_message_data_with_timestamp):
    data = raw_message_data_with_timestamp.copy()
    data["doc_id"] = "doc_4_no_tag"
    data["content"] = "This is just normal content without any think tags."
    return data

@pytest.fixture
def message_obj_with_timestamp(raw_message_data_with_timestamp):
    return ConversationMessage.from_dict(raw_message_data_with_timestamp)

@pytest.fixture
def message_obj_with_think_tag_in_content(raw_message_data_with_think_tag_in_content):
    return ConversationMessage.from_dict(raw_message_data_with_think_tag_in_content)

@pytest.fixture
def message_obj_with_closing_think_tag_in_content(raw_message_data_with_closing_think_tag_in_content):
    return ConversationMessage.from_dict(raw_message_data_with_closing_think_tag_in_content)

@pytest.fixture
def message_obj_without_think_tag_in_content(raw_message_data_without_think_tag_in_content):
    return ConversationMessage.from_dict(raw_message_data_without_think_tag_in_content) 