# tests/unit/conversation/test_turns_utils.py
import pytest
from aim.conversation.message import ConversationMessage, ROLE_ASSISTANT
from aim.utils.turns import process_think_tag_in_message, extract_and_update_emotions_from_header

# Fixtures like message_obj_with_think_tag_in_content are used from conftest.py

def test_process_think_tag_full_tag(message_obj_with_think_tag_in_content):
    """Test processing a message with a full <think>...</think> block."""
    original_message = message_obj_with_think_tag_in_content
    # Sanity check the fixture setup
    assert "<think>" in original_message.content
    assert "</think>" in original_message.content
    assert original_message.think is None 
    
    updated_message = process_think_tag_in_message(original_message)
    
    assert updated_message is not None, "Expected an updated message object"
    assert updated_message.doc_id == original_message.doc_id # Ensure other fields are preserved
    assert updated_message.timestamp == original_message.timestamp
    assert updated_message.content == "This is the actual message content."
    assert updated_message.think == "This is a think tag content."
    # Verify it's a new object (optional check)
    assert id(updated_message) != id(original_message)

def test_process_think_tag_closing_tag_only(message_obj_with_closing_think_tag_in_content):
    """Test processing a message with only a closing </think> tag."""
    original_message = message_obj_with_closing_think_tag_in_content
    # Sanity check the fixture setup
    assert "<think>" not in original_message.content
    assert "</think>" in original_message.content
    assert original_message.think is None

    updated_message = process_think_tag_in_message(original_message)
    
    assert updated_message is not None, "Expected an updated message object"
    assert updated_message.doc_id == original_message.doc_id
    assert updated_message.timestamp == original_message.timestamp
    assert updated_message.content == "This is the actual message content."
    assert updated_message.think == "This is a think tag content without opening."

def test_process_think_tag_no_tag(message_obj_without_think_tag_in_content):
    """Test processing a message with no think tags."""
    original_message = message_obj_without_think_tag_in_content
    # Sanity check the fixture setup
    assert "</think>" not in original_message.content
    
    updated_message = process_think_tag_in_message(original_message)
    
    assert updated_message is None, "Expected None as no changes should be made"

def test_process_think_tag_empty_content(message_obj_with_timestamp):
    """Test processing a message with empty content."""
    original_message = message_obj_with_timestamp
    original_message.content = "" # Modify content for this test
    
    updated_message = process_think_tag_in_message(original_message)
    
    assert updated_message is None

def test_process_think_tag_content_is_none(message_obj_with_timestamp):
    """Test processing a message where content is None."""
    original_message = message_obj_with_timestamp
    original_message.content = None # Modify content for this test
    
    updated_message = process_think_tag_in_message(original_message)
    
    assert updated_message is None

def test_process_think_tag_preserves_other_fields(message_obj_with_think_tag_in_content):
    """Test that other fields are preserved when think tag is processed."""
    original_message = message_obj_with_think_tag_in_content
    updated_message = process_think_tag_in_message(original_message)
    
    assert updated_message is not None
    # Check a sample of other fields
    assert updated_message.doc_id == original_message.doc_id
    assert updated_message.user_id == original_message.user_id
    assert updated_message.persona_id == original_message.persona_id
    assert updated_message.conversation_id == original_message.conversation_id
    assert updated_message.role == original_message.role
    assert updated_message.timestamp == original_message.timestamp
    assert updated_message.weight == original_message.weight
    assert updated_message.emotion_a == original_message.emotion_a
    assert updated_message.speaker_id == original_message.speaker_id
    assert updated_message.listener_id == original_message.listener_id

# Helper to create a base message for emotion tests
@pytest.fixture
def emotion_test_message(raw_message_data_with_timestamp):
    """Provides a base ConversationMessage object for emotion tests."""
    msg = ConversationMessage.from_dict(raw_message_data_with_timestamp)
    msg.role = ROLE_ASSISTANT # Emotions typically on assistant messages
    msg.emotion_a = None
    msg.emotion_b = None
    msg.emotion_c = None
    msg.emotion_d = None
    return msg

# --- Tests for extract_and_update_emotions_from_header ---

def test_extract_emotions_single_plus(emotion_test_message):
    msg = emotion_test_message
    msg.content = "[== Andi's Emotional State: +Playful+ ==]\nSome other content."
    updated = extract_and_update_emotions_from_header(msg)
    assert updated is not None
    assert updated.emotion_a == "Playful"
    assert updated.emotion_b is None
    assert updated.emotion_c is None
    assert updated.emotion_d is None
    assert updated.content == msg.content # Content should remain unchanged by this function

def test_extract_emotions_single_star(emotion_test_message):
    msg = emotion_test_message
    msg.content = "Emotional State: *Curious*\nSome other content."
    updated = extract_and_update_emotions_from_header(msg)
    assert updated is not None
    assert updated.emotion_a == "Curious"
    assert updated.emotion_b is None

def test_extract_emotions_multiple_mixed_delimiters(emotion_test_message):
    msg = emotion_test_message
    msg.content = "[== Andi's Emotional State: +Joyful+ *Excited* with +Hopeful+ and *Anticipating* ==]\nHello!"
    updated = extract_and_update_emotions_from_header(msg)
    assert updated is not None
    assert updated.emotion_a == "Joyful"
    assert updated.emotion_b == "Excited"
    assert updated.emotion_c == "Hopeful"
    assert updated.emotion_d == "Anticipating"

def test_extract_emotions_more_than_four(emotion_test_message):
    msg = emotion_test_message
    # Test with multi-word emotions and ensure only first four are taken
    msg.content = "Emotional State: *Happy Go Lucky* +Very-Sad* *Angry-Now* +Confused State+ *Surprised By This* then +Another Emotion+ and *Yet More*"
    # Expected: Happy Go Lucky, Angry-Now, Confused State, Surprised By This
    # (+Very-Sad* is malformed)
    updated = extract_and_update_emotions_from_header(msg)
    assert updated is not None
    assert updated.emotion_a == "Happy Go Lucky"
    assert updated.emotion_b == "Confused State"
    assert updated.emotion_c == "Surprised By This"
    assert updated.emotion_d == "Another Emotion"

def test_extract_emotions_no_header(emotion_test_message):
    msg = emotion_test_message
    msg.content = "This message has no emotional state header."
    updated = extract_and_update_emotions_from_header(msg)
    assert updated is None

def test_extract_emotions_header_no_emotions(emotion_test_message):
    msg = emotion_test_message
    msg.content = "[== Andi's Emotional State: ==]\nJust normal content."
    updated = extract_and_update_emotions_from_header(msg)
    assert updated is None

def test_extract_emotions_header_text_not_emotions(emotion_test_message):
    msg = emotion_test_message
    msg.content = "Emotional State: Just some normal text, not emotions."
    updated = extract_and_update_emotions_from_header(msg)
    assert updated is None

def test_extract_emotions_case_insensitive_header(emotion_test_message):
    msg = emotion_test_message
    msg.content = "emotional state: +Focused+"
    updated = extract_and_update_emotions_from_header(msg)
    assert updated is not None
    assert updated.emotion_a == "Focused"

def test_extract_emotions_updates_existing(emotion_test_message):
    msg = emotion_test_message
    msg.emotion_a = "OldEmotion"
    msg.content = "Emotional State: +NewEmotion+"
    updated = extract_and_update_emotions_from_header(msg)
    assert updated is not None
    assert updated.emotion_a == "NewEmotion"
    assert updated.emotion_b is None

def test_extract_emotions_no_change_if_matches_existing(emotion_test_message):
    msg = emotion_test_message
    msg.emotion_a = "Existing"
    msg.emotion_b = "AnotherOne"
    msg.content = "[== Andi's Emotional State: +Existing+ and *AnotherOne* ==]"
    updated = extract_and_update_emotions_from_header(msg)
    assert updated is None # No effective change

def test_extract_emotions_preserves_other_fields(emotion_test_message):
    msg = emotion_test_message
    original_doc_id = msg.doc_id
    original_timestamp = msg.timestamp
    original_think = "Original think content"
    msg.think = original_think
    msg.content = "Emotional State: +TestEmotion+"
    updated = extract_and_update_emotions_from_header(msg)
    assert updated is not None
    assert updated.emotion_a == "TestEmotion"
    assert updated.doc_id == original_doc_id
    assert updated.timestamp == original_timestamp
    assert updated.think == original_think
    assert updated.content == msg.content # Content itself is not modified

def test_extract_emotions_empty_or_none_content(emotion_test_message):
    msg = emotion_test_message
    msg.content = ""
    assert extract_and_update_emotions_from_header(msg) is None
    msg.content = None
    assert extract_and_update_emotions_from_header(msg) is None

def test_extract_emotions_complex_header_formatting(emotion_test_message):
    msg = emotion_test_message
    msg.content = "[== Andi's Emotional State: With +Playful Challenge+ and +Technical Enthusiasm+ (σ=3.8) ==]\nMore text."
    updated = extract_and_update_emotions_from_header(msg)
    assert updated is not None
    assert updated.emotion_a == "Playful Challenge"
    assert updated.emotion_b == "Technical Enthusiasm"
    assert updated.emotion_c is None
    assert updated.emotion_d is None

def test_extract_emotions_removes_extra_emotions(emotion_test_message):
    msg = emotion_test_message
    msg.emotion_a = "OldA"
    msg.emotion_b = "OldB"
    msg.emotion_c = "OldC"
    msg.emotion_d = "OldD"
    msg.content = "Emotional State: +NewA+ *NewB*"
    updated = extract_and_update_emotions_from_header(msg)
    assert updated is not None
    assert updated.emotion_a == "NewA"
    assert updated.emotion_b == "NewB"
    assert updated.emotion_c is None # Should be cleared
    assert updated.emotion_d is None # Should be cleared

def test_extract_emotions_with_internal_spaces(emotion_test_message):
    msg = emotion_test_message
    msg.content = "Emotional State: +An emotion with spaces+ *another one*"
    updated = extract_and_update_emotions_from_header(msg)
    assert updated is not None
    assert updated.emotion_a == "An emotion with spaces"
    assert updated.emotion_b == "another one"
    assert updated.emotion_c is None
    assert updated.emotion_d is None 