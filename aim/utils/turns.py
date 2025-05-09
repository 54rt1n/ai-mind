# aim/utils/turns.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import logging
from typing import Optional
import re

from aim.conversation.message import ConversationMessage

logger = logging.getLogger(__name__)


def validate_turns(turns: list[dict[str, str]]) -> None:
    """
    Validates that the turns follow a user/assistant turn structure, with an optional 'system' turn as the first.
    """
    
    if len(turns) == 0:
        raise ValueError("No turns in the list.")

    offset = 0
    if turns[0]["role"] == "system":
        offset = 1
        
    for i, turn in enumerate(turns[offset:]):
        if turn["role"] == "user":
            if i % 2 != 0: 
                logger.warning(', '.join([turn['role'] for turn in turns]))
                for turn in turns:
                    logger.debug(turn)
                raise ValueError(f"Turn {i + offset} is not a user turn: {turn['role']}")
        elif turn["role"] == "assistant":
            if i % 2 != 1:
                logger.warning(', '.join([turn['role'] for turn in turns]))
                raise ValueError(f"Turn {i + offset} is not an assistant turn: {turn['role']}")
        else:
            logger.warning(', '.join([turn['role'] for turn in turns]))
            raise ValueError(f"Turn {i + offset} is not a user or assistant turn: {turn['role']}")


def process_think_tag_in_message(message: ConversationMessage) -> Optional[ConversationMessage]:
    """
    Checks if a </think> tag exists in the message content.
    If so, extracts the content before it into the 'think' field and the content after it
    as the new 'content' field. Handles an optional leading <think> tag.
    Returns a new ConversationMessage object if changes were made, otherwise None.
    """
    content = message.content
    if content is None: # Should not happen for typical messages, but good to check
        return None

    think_tag_end = "</think>"
    think_tag_end_index = content.find(think_tag_end)

    if think_tag_end_index == -1:
        return None  # No closing think tag found, no changes needed

    think_text_raw = content[:think_tag_end_index]
    new_content_raw = content[think_tag_end_index + len(think_tag_end):]

    # Clean up think_text: remove <think> if present at the start (after potential whitespace)
    think_text_cleaned = think_text_raw
    # Use regex to remove leading "<think>" and any surrounding whitespace robustly
    think_text_cleaned = re.sub(r"^\s*<think>\s*", "", think_text_cleaned, count=1)

    think_text_final = think_text_cleaned.strip()
    new_content_final = new_content_raw.strip()

    # Create a new message object if the fields have effectively changed
    # or if the presence of </think> in content implies it was not correctly processed before.
    # Even if think_text_final is same as message.think and new_content_final is same as message.content,
    # the act of finding </think> in original content means it was "dirty".
    
    updated_data = message.to_dict()
    updated_data["think"] = think_text_final
    updated_data["content"] = new_content_final
    
    # Return a new ConversationMessage instance
    return ConversationMessage.from_dict(updated_data)
    

def extract_and_update_emotions_from_header(message: ConversationMessage) -> Optional[ConversationMessage]:
    """
    Checks for an "Emotional State:" header in the message content.
    If found, extracts emotions (e.g., ++emotion++ or **emotion**) and populates
    emotion_a, emotion_b, emotion_c, emotion_d fields.
    Returns a new ConversationMessage object if changes were made, otherwise None.
    """
    if not message.content:
        return None

    emotional_state_line_content = None
    # Find the line containing "Emotional State:"
    for line in message.content.splitlines():
        # More robustly find the relevant part of the line, ignoring prefixes like "[== ... Emotional State:"
        # and capturing the part after "Emotional State:"
        match = re.search(r"Emotional State:\s*(.*)", line, re.IGNORECASE)
        if match:
            emotional_state_line_content = match.group(1).strip()
            # Preprocess to replace double delimiters ++/** with single +/*
            if emotional_state_line_content:
                emotional_state_line_content = emotional_state_line_content.replace('++', '+').replace('**', '*')
            break
    
    if emotional_state_line_content is None:
        return None

    # Find *emotion* or +emotion+ using the final working regex
    extracted_emotions_raw = re.findall(r"(\*([\w\s-]+)\*|\+([\w\s-]+)\+)", emotional_state_line_content)

    extracted_emotions = []
    # Loop logic for the regex which returns tuples like ('*Angry*', 'Angry', '') or ('+Confused+', '', 'Confused')
    for tpl in extracted_emotions_raw:
        full_match = tpl[0]
        star_content = tpl[1].strip()
        plus_content = tpl[2].strip()
        
        if star_content:
            extracted_emotions.append(star_content)
        elif plus_content:
            extracted_emotions.append(plus_content)
        # Fallback/safety if group logic somehow failed but full match exists
        # This case should theoretically not be needed with the current regex structure
        elif len(full_match) > 2:
             content = full_match[1:-1].strip()
             if content:
                 extracted_emotions.append(content)

    logger.debug(f"Extracted emotions for doc_id {message.doc_id}: {extracted_emotions}")

    if not extracted_emotions:
        # No emotions found in the header line itself
        return None

    # Prepare new emotion values, up to 4
    new_emotion_a = extracted_emotions[0] if len(extracted_emotions) > 0 else None
    new_emotion_b = extracted_emotions[1] if len(extracted_emotions) > 1 else None
    new_emotion_c = extracted_emotions[2] if len(extracted_emotions) > 2 else None
    new_emotion_d = extracted_emotions[3] if len(extracted_emotions) > 3 else None

    # Check if any new emotion is different from the existing one,
    # or if any existing emotion was None and is now populated,
    # or if an existing emotion was populated and is now None because fewer emotions were extracted.
    # More precise change detection:
    current_emotions = [message.emotion_a, message.emotion_b, message.emotion_c, message.emotion_d]
    new_emotions = [new_emotion_a, new_emotion_b, new_emotion_c, new_emotion_d]

    # Changed if the lists of emotions are different, ignoring cases where all were/are None for a slot.
    changed = current_emotions != new_emotions
    
    # An additional check: if all new_emotions are None AND all current_emotions were already None, no change.
    # This handles the case where the header exists but has no parseable emotions,
    # and the message object already had no emotions.
    if all(e is None for e in new_emotions) and all(e is None for e in current_emotions):
        changed = False
    
    if changed:
        updated_data = message.to_dict()
        updated_data["emotion_a"] = new_emotion_a
        updated_data["emotion_b"] = new_emotion_b
        updated_data["emotion_c"] = new_emotion_c
        updated_data["emotion_d"] = new_emotion_d
        
        logger.info(f"Extracted and updated emotions for doc_id {message.doc_id}: a='{new_emotion_a}', b='{new_emotion_b}', c='{new_emotion_c}', d='{new_emotion_d}'")
        return ConversationMessage.from_dict(updated_data)
    else:
        logger.debug(f"Extracted emotions for doc_id {message.doc_id} matched existing ones. No update needed.")
        return None
    