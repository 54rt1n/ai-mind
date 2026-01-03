# aim/utils/tokens.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Shared token counting utilities.

Provides a single source of truth for token counting using tiktoken,
with caching for the encoder to avoid repeated initialization.
"""

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def get_encoder() -> tiktoken.Encoding:
    """Get the tiktoken encoder (cached)."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Count tokens in a string using tiktoken.

    Args:
        text: The text to count tokens for.

    Returns:
        Number of tokens in the text.
    """
    if not text:
        return 0
    return len(get_encoder().encode(text))
