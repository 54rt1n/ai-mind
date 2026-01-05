# aim/app/mud/utils.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Utility functions for MUD agent worker."""


def sanitize_response(text: str) -> str:
    """Truncate at second </think> tag if present.

    Handles malformed responses where the model outputs multiple think blocks.
    """
    first = text.find("</think>")
    if first == -1:
        return text
    second = text.find("</think>", first + 8)
    if second == -1:
        return text
    return text[:second].strip()
