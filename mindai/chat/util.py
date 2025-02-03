# mindai/chat/util.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import logging

logger = logging.getLogger(__name__)

def insert_at_fold(turns: list[dict[str, str]], content: str, fold_depth: int = 4) -> list[dict[str, str]]:
    """
    Inserts content at the first user turn after the specified fold depth in a list of chat turns.
    
    Args:
        turns (List[Dict[str, str]]): The list of chat turns.
        content (str): The content to insert.
        fold_depth (int): The fold depth to search for the first user turn. Defaults to 4.
    
    Returns:
        List[Dict[str, str]]: The updated list of chat turns with the content inserted.
    """
        
    # We work from the end, counting the number of user turns until we find the first one, or hit the fold depth
    depth = 0
    ix = 0
    while True:
        if ix >= len(turns) - 1:
            break
        if turns[ix]["role"] == "user":
            depth += 1
            if depth >= fold_depth:
                break
        ix += 1
        
    logger.info(f"Inserting {len(content)} at fold depth {fold_depth} at index {ix}/{len(turns)}")
    # Copy our turns
    new_turns = [*turns]
    # Get our user turn
    target = new_turns[ix]
    # Insert our content
    new_turns[ix] = {"role": target['role'], "content": f"{content}{target['content']}"}

    return new_turns
