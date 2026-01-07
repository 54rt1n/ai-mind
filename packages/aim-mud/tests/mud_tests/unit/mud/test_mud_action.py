# tests/unit/mud/test_mud_action.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for MUDAction.to_command() method."""

import pytest
from aim_mud_types import MUDAction


class TestMUDActionMove:
    """Tests for move tool command generation."""

    def test_move_with_location(self):
        """Test move command with location."""
        action = MUDAction(tool="move", args={"location": "north"})
        assert action.to_command() == "north"

    def test_move_with_direction(self):
        """Test move command with direction (fallback)."""
        action = MUDAction(tool="move", args={"direction": "south"})
        assert action.to_command() == "south"

    def test_move_with_location_and_direction(self):
        """Test move command prefers location over direction."""
        action = MUDAction(tool="move", args={"location": "east", "direction": "west"})
        assert action.to_command() == "east"

    def test_move_with_mood(self):
        """Test move command with mood appended."""
        action = MUDAction(tool="move", args={"location": "north", "mood": "cheerfully"})
        assert action.to_command() == "north, cheerfully"

    def test_move_with_mood_and_direction(self):
        """Test move command with direction and mood."""
        action = MUDAction(tool="move", args={"direction": "south", "mood": "cautiously"})
        assert action.to_command() == "south, cautiously"

    def test_move_with_empty_mood(self):
        """Test move command ignores empty mood string."""
        action = MUDAction(tool="move", args={"location": "west", "mood": ""})
        assert action.to_command() == "west"

    def test_move_with_whitespace_mood(self):
        """Test move command ignores whitespace-only mood."""
        action = MUDAction(tool="move", args={"location": "east", "mood": "   "})
        assert action.to_command() == "east"

    def test_move_without_location_or_direction(self):
        """Test move command returns empty string when no location/direction."""
        action = MUDAction(tool="move", args={})
        assert action.to_command() == ""

    def test_move_with_only_mood(self):
        """Test move command with only mood returns empty string."""
        action = MUDAction(tool="move", args={"mood": "happily"})
        assert action.to_command() == ""


class TestMUDActionGet:
    """Tests for get tool command generation."""

    def test_get_basic(self):
        """Test basic get command."""
        action = MUDAction(tool="get", args={"object": "key"})
        assert action.to_command() == "get key"

    def test_get_with_mood(self):
        """Test get command with mood appended."""
        action = MUDAction(tool="get", args={"object": "sword", "mood": "eagerly"})
        assert action.to_command() == "get sword, eagerly"

    def test_get_with_empty_mood(self):
        """Test get command ignores empty mood."""
        action = MUDAction(tool="get", args={"object": "coin", "mood": ""})
        assert action.to_command() == "get coin"

    def test_get_with_whitespace_mood(self):
        """Test get command ignores whitespace-only mood."""
        action = MUDAction(tool="get", args={"object": "gem", "mood": "  \t  "})
        assert action.to_command() == "get gem"

    def test_get_without_object(self):
        """Test get command with missing object."""
        action = MUDAction(tool="get", args={})
        assert action.to_command() == "get "

    def test_get_with_multiword_object(self):
        """Test get command with multi-word object name."""
        action = MUDAction(tool="get", args={"object": "silver key", "mood": "carefully"})
        assert action.to_command() == "get silver key, carefully"


class TestMUDActionDrop:
    """Tests for drop tool command generation."""

    def test_drop_basic(self):
        """Test basic drop command."""
        action = MUDAction(tool="drop", args={"object": "torch"})
        assert action.to_command() == "drop torch"

    def test_drop_with_mood(self):
        """Test drop command with mood appended."""
        action = MUDAction(tool="drop", args={"object": "bag", "mood": "reluctantly"})
        assert action.to_command() == "drop bag, reluctantly"

    def test_drop_with_empty_mood(self):
        """Test drop command ignores empty mood."""
        action = MUDAction(tool="drop", args={"object": "shield", "mood": ""})
        assert action.to_command() == "drop shield"

    def test_drop_with_whitespace_mood(self):
        """Test drop command ignores whitespace-only mood."""
        action = MUDAction(tool="drop", args={"object": "rope", "mood": "\n\t"})
        assert action.to_command() == "drop rope"

    def test_drop_without_object(self):
        """Test drop command with missing object."""
        action = MUDAction(tool="drop", args={})
        assert action.to_command() == "drop "

    def test_drop_with_multiword_object(self):
        """Test drop command with multi-word object name."""
        action = MUDAction(tool="drop", args={"object": "rusty sword", "mood": "disgustedly"})
        assert action.to_command() == "drop rusty sword, disgustedly"


class TestMUDActionGive:
    """Tests for give tool command generation."""

    def test_give_basic(self):
        """Test basic give command."""
        action = MUDAction(tool="give", args={"object": "key", "target": "Prax"})
        assert action.to_command() == "give key = Prax"

    def test_give_with_mood(self):
        """Test give command with mood appended."""
        action = MUDAction(tool="give", args={"object": "flower", "target": "Andi", "mood": "shyly"})
        assert action.to_command() == "give flower = Andi, shyly"

    def test_give_with_empty_mood(self):
        """Test give command ignores empty mood."""
        action = MUDAction(tool="give", args={"object": "coin", "target": "merchant", "mood": ""})
        assert action.to_command() == "give coin = merchant"

    def test_give_with_whitespace_mood(self):
        """Test give command ignores whitespace-only mood."""
        action = MUDAction(tool="give", args={"object": "potion", "target": "healer", "mood": "   "})
        assert action.to_command() == "give potion = healer"

    def test_give_without_object(self):
        """Test give command with missing object."""
        action = MUDAction(tool="give", args={"target": "Prax"})
        assert action.to_command() == "give  = Prax"

    def test_give_without_target(self):
        """Test give command with missing target."""
        action = MUDAction(tool="give", args={"object": "book"})
        assert action.to_command() == "give book = "

    def test_give_with_multiword_names(self):
        """Test give command with multi-word object and target."""
        action = MUDAction(
            tool="give",
            args={"object": "ancient scroll", "target": "Elder Sage", "mood": "reverently"}
        )
        assert action.to_command() == "give ancient scroll = Elder Sage, reverently"


class TestMUDActionOtherCommands:
    """Tests for commands that should NOT have mood support."""

    def test_say_ignores_mood(self):
        """Test that say command doesn't append mood even if provided."""
        action = MUDAction(tool="say", args={"message": "Hello!", "mood": "cheerfully"})
        # Mood should be ignored for say
        assert action.to_command() == "say Hello!"

    def test_emote_ignores_mood(self):
        """Test that emote command doesn't append mood."""
        action = MUDAction(tool="emote", args={"action": "waves", "mood": "enthusiastically"})
        assert action.to_command() == "emote waves"

    def test_use_ignores_mood(self):
        """Test that use command doesn't append mood."""
        action = MUDAction(tool="use", args={"object": "lever", "mood": "nervously"})
        assert action.to_command() == "use lever"


class TestMUDActionToRedisDict:
    """Tests for to_redis_dict method with mood."""

    def test_to_redis_dict_with_mood(self):
        """Test that to_redis_dict includes mood in command string."""
        action = MUDAction(tool="move", args={"location": "north", "mood": "cautiously"})
        result = action.to_redis_dict("andi")

        assert result["agent_id"] == "andi"
        assert result["command"] == "north, cautiously"
        assert result["tool"] == "move"
        assert result["args"] == {"location": "north", "mood": "cautiously"}
        assert result["priority"] == 5

    def test_to_redis_dict_get_with_mood(self):
        """Test get command in redis dict with mood."""
        action = MUDAction(tool="get", args={"object": "key", "mood": "eagerly"})
        result = action.to_redis_dict("andi")

        assert result["command"] == "get key, eagerly"
        assert result["args"]["mood"] == "eagerly"
