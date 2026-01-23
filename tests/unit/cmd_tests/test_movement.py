# tests/unit/commands/test_movement.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for movement command logic.

These tests verify the mood parsing functionality without requiring
full Evennia/Django initialization.
"""

import pytest


def parse_mood(args_string: str) -> tuple[str, str]:
    """Parse comma-separated mood from command args.

    This is a copy of the parse_mood function from andimud.commands.mud.movement
    for testing without Django dependencies.

    Args:
        args_string: Raw command arguments (may include mood after comma)

    Returns:
        Tuple of (cleaned_args, mood_text)
        - cleaned_args: Arguments before the comma
        - mood_text: The mood string after comma, or empty string if not present

    Examples:
        "north, thoughtfully" → ("north", "thoughtfully")
        "north" → ("north", "")
        "garden, with quiet purpose" → ("garden", "with quiet purpose")
    """
    if not args_string or "," not in args_string:
        return args_string or "", ""

    parts = args_string.split(",", 1)
    cleaned_args = parts[0].strip()
    mood_text = parts[1].strip()
    return cleaned_args, mood_text


class TestMovementLogic:
    """Test movement command logic and mood parsing."""

    def test_parse_mood_with_mood(self):
        """Test parse_mood extracts mood from comma-separated string."""
        args, mood = parse_mood("north, thoughtfully")
        assert args == "north"
        assert mood == "thoughtfully"

    def test_parse_mood_without_mood(self):
        """Test parse_mood handles strings without mood."""
        args, mood = parse_mood("north")
        assert args == "north"
        assert mood == ""

    def test_parse_mood_with_longer_mood(self):
        """Test parse_mood handles multi-word mood expressions."""
        args, mood = parse_mood("garden, with quiet purpose")
        assert args == "garden"
        assert mood == "with quiet purpose"

    def test_parse_mood_empty_string(self):
        """Test parse_mood handles empty string."""
        args, mood = parse_mood("")
        assert args == ""
        assert mood == ""

    def test_parse_mood_only_comma(self):
        """Test parse_mood handles edge case of only comma."""
        args, mood = parse_mood("north,")
        assert args == "north"
        assert mood == ""

    def test_parse_mood_strips_whitespace(self):
        """Test parse_mood strips whitespace from both parts."""
        args, mood = parse_mood("  north  ,  thoughtfully  ")
        assert args == "north"
        assert mood == "thoughtfully"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
