# aim/utils/turns.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import logging

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
    