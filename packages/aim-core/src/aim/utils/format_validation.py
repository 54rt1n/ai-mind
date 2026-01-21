# aim/utils/format_validation.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Response format validation utilities.

Provides functions for validating LLM response formats, particularly
the emotional state header format used in persona-driven responses.
"""

import re


def has_emotional_state_header(response: str) -> bool:
    """Check if response starts with emotional state header after think block.

    Validates that the response begins with the expected format:
    [== <name>'s Emotional State: <emotions> ==]

    The check is performed after removing any <think>...</think> blocks,
    since models may emit thinking before the formatted response.

    Args:
        response: The response text to validate

    Returns:
        True if response has the required header format at the start
    """
    if not response:
        return False

    # Remove think block first
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    if not cleaned:
        return False

    # Check if it starts with [== ... Emotional State ... ==]
    return bool(re.match(r'\[==.*Emotional State.*==\]', cleaned, re.IGNORECASE))
