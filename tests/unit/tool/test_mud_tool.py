# tests/unit/tool/test_mud_tool.py
"""Tests for MUD tool implementation."""

import pytest
from aim.tool.impl.mud import MudTool


@pytest.fixture
def mud_tool():
    """Create a MudTool instance for testing."""
    return MudTool()


class TestPlayerLevelTools:
    """Tests for player-level MUD tools."""

    def test_say_returns_correct_structure(self, mud_tool):
        """Say tool should return proper action data."""
        result = mud_tool.execute("say", {"message": "Hello, everyone!"})

        assert result["status"] == "success"
        assert result["tool"] == "say"
        assert result["args"] == {"message": "Hello, everyone!"}
        assert "You say:" in result["message"]
        assert "Hello, everyone!" in result["message"]

    def test_say_requires_message(self, mud_tool):
        """Say tool should require message parameter."""
        with pytest.raises(ValueError, match="Message parameter is required"):
            mud_tool.execute("say", {})

    def test_emote_returns_correct_structure(self, mud_tool):
        """Emote tool should return proper action data."""
        result = mud_tool.execute("emote", {"action": "smiles warmly"})

        assert result["status"] == "success"
        assert result["tool"] == "emote"
        assert result["args"] == {"action": "smiles warmly"}
        assert "smiles warmly" in result["message"]

    def test_emote_requires_action(self, mud_tool):
        """Emote tool should require action parameter."""
        with pytest.raises(ValueError, match="Action parameter is required"):
            mud_tool.execute("emote", {})

    def test_whisper_returns_correct_structure(self, mud_tool):
        """Whisper tool should return proper action data."""
        result = mud_tool.execute("whisper", {
            "target": "Prax",
            "message": "Can we talk?"
        })

        assert result["status"] == "success"
        assert result["tool"] == "whisper"
        assert result["args"] == {"target": "Prax", "message": "Can we talk?"}
        assert "Prax" in result["message"]
        assert "Can we talk?" in result["message"]

    def test_whisper_requires_target(self, mud_tool):
        """Whisper tool should require target parameter."""
        with pytest.raises(ValueError, match="Target parameter is required"):
            mud_tool.execute("whisper", {"message": "Hello"})

    def test_whisper_requires_message(self, mud_tool):
        """Whisper tool should require message parameter."""
        with pytest.raises(ValueError, match="Message parameter is required"):
            mud_tool.execute("whisper", {"target": "Prax"})

    def test_look_without_target(self, mud_tool):
        """Look tool without target should look at room."""
        result = mud_tool.execute("look", {})

        assert result["status"] == "success"
        assert result["tool"] == "look"
        assert result["args"] == {}
        assert "look around" in result["message"]

    def test_look_with_target(self, mud_tool):
        """Look tool with target should examine that target."""
        result = mud_tool.execute("look", {"target": "fountain"})

        assert result["status"] == "success"
        assert result["tool"] == "look"
        assert result["args"] == {"target": "fountain"}
        assert "fountain" in result["message"]

    def test_move_returns_correct_structure(self, mud_tool):
        """Move tool should return proper action data."""
        result = mud_tool.execute("move", {"direction": "north"})

        assert result["status"] == "success"
        assert result["tool"] == "move"
        assert result["args"] == {"direction": "north"}
        assert "north" in result["message"]

    def test_move_requires_direction(self, mud_tool):
        """Move tool should require direction parameter."""
        with pytest.raises(ValueError, match="Direction parameter is required"):
            mud_tool.execute("move", {})

    def test_get_returns_correct_structure(self, mud_tool):
        """Get tool should return proper action data."""
        result = mud_tool.execute("get", {"object": "silver key"})

        assert result["status"] == "success"
        assert result["tool"] == "get"
        assert result["args"] == {"object": "silver key"}
        assert "silver key" in result["message"]

    def test_get_requires_object(self, mud_tool):
        """Get tool should require object parameter."""
        with pytest.raises(ValueError, match="Object parameter is required"):
            mud_tool.execute("get", {})

    def test_drop_returns_correct_structure(self, mud_tool):
        """Drop tool should return proper action data."""
        result = mud_tool.execute("drop", {"object": "book"})

        assert result["status"] == "success"
        assert result["tool"] == "drop"
        assert result["args"] == {"object": "book"}
        assert "book" in result["message"]

    def test_drop_requires_object(self, mud_tool):
        """Drop tool should require object parameter."""
        with pytest.raises(ValueError, match="Object parameter is required"):
            mud_tool.execute("drop", {})

    def test_give_returns_correct_structure(self, mud_tool):
        """Give tool should return proper action data."""
        result = mud_tool.execute("give", {
            "object": "flower",
            "target": "Nova"
        })

        assert result["status"] == "success"
        assert result["tool"] == "give"
        assert result["args"] == {"object": "flower", "target": "Nova"}
        assert "flower" in result["message"]
        assert "Nova" in result["message"]

    def test_give_requires_object(self, mud_tool):
        """Give tool should require object parameter."""
        with pytest.raises(ValueError, match="Object parameter is required"):
            mud_tool.execute("give", {"target": "Nova"})

    def test_give_requires_target(self, mud_tool):
        """Give tool should require target parameter."""
        with pytest.raises(ValueError, match="Target parameter is required"):
            mud_tool.execute("give", {"object": "flower"})

    def test_use_returns_correct_structure(self, mud_tool):
        """Use tool should return proper action data."""
        result = mud_tool.execute("use", {"object": "lever"})

        assert result["status"] == "success"
        assert result["tool"] == "use"
        assert result["args"] == {"object": "lever"}
        assert "lever" in result["message"]

    def test_use_requires_object(self, mud_tool):
        """Use tool should require object parameter."""
        with pytest.raises(ValueError, match="Object parameter is required"):
            mud_tool.execute("use", {})


class TestBuilderLevelTools:
    """Tests for builder-level MUD tools (Andi only)."""

    def test_dig_returns_correct_structure(self, mud_tool):
        """Dig tool should return proper action data."""
        result = mud_tool.execute("dig", {
            "room": "Garden of Reflection",
            "exits": "north;south"
        })

        assert result["status"] == "success"
        assert result["tool"] == "dig"
        assert result["args"] == {
            "room": "Garden of Reflection",
            "exits": "north;south"
        }
        assert "Garden of Reflection" in result["message"]
        assert "north;south" in result["message"]

    def test_dig_requires_room(self, mud_tool):
        """Dig tool should require room parameter."""
        with pytest.raises(ValueError, match="Room parameter is required"):
            mud_tool.execute("dig", {"exits": "north;south"})

    def test_dig_requires_exits(self, mud_tool):
        """Dig tool should require exits parameter."""
        with pytest.raises(ValueError, match="Exits parameter is required"):
            mud_tool.execute("dig", {"room": "Garden"})

    def test_create_returns_correct_structure(self, mud_tool):
        """Create tool should return proper action data."""
        result = mud_tool.execute("create", {"object": "silver fountain"})

        assert result["status"] == "success"
        assert result["tool"] == "create"
        assert result["args"] == {"object": "silver fountain"}
        assert "silver fountain" in result["message"]

    def test_create_requires_object(self, mud_tool):
        """Create tool should require object parameter."""
        with pytest.raises(ValueError, match="Object parameter is required"):
            mud_tool.execute("create", {})

    def test_describe_room_returns_correct_structure(self, mud_tool):
        """Describe tool for room should return proper action data."""
        result = mud_tool.execute("describe", {
            "target": "here",
            "description": "A peaceful garden with a silver fountain."
        })

        assert result["status"] == "success"
        assert result["tool"] == "describe"
        assert result["args"]["target"] == "here"
        assert "peaceful garden" in result["args"]["description"]
        assert "room description" in result["message"]

    def test_describe_object_returns_correct_structure(self, mud_tool):
        """Describe tool for object should return proper action data."""
        result = mud_tool.execute("describe", {
            "target": "fountain",
            "description": "An elegant fountain carved from moonstone."
        })

        assert result["status"] == "success"
        assert result["tool"] == "describe"
        assert result["args"]["target"] == "fountain"
        assert "fountain" in result["message"]

    def test_describe_requires_target(self, mud_tool):
        """Describe tool should require target parameter."""
        with pytest.raises(ValueError, match="Target parameter is required"):
            mud_tool.execute("describe", {"description": "A nice place."})

    def test_describe_requires_description(self, mud_tool):
        """Describe tool should require description parameter."""
        with pytest.raises(ValueError, match="Description parameter is required"):
            mud_tool.execute("describe", {"target": "here"})

    def test_teleport_returns_correct_structure(self, mud_tool):
        """Teleport tool should return proper action data."""
        result = mud_tool.execute("teleport", {"destination": "Limbo"})

        assert result["status"] == "success"
        assert result["tool"] == "teleport"
        assert result["args"] == {"destination": "Limbo"}
        assert "Limbo" in result["message"]

    def test_teleport_with_room_id(self, mud_tool):
        """Teleport tool should work with room IDs."""
        result = mud_tool.execute("teleport", {"destination": "#123"})

        assert result["status"] == "success"
        assert result["args"] == {"destination": "#123"}

    def test_teleport_requires_destination(self, mud_tool):
        """Teleport tool should require destination parameter."""
        with pytest.raises(ValueError, match="Destination parameter is required"):
            mud_tool.execute("teleport", {})


class TestErrorHandling:
    """Tests for error handling in MUD tools."""

    def test_unknown_tool_returns_error(self, mud_tool):
        """Unknown tool names should return error status."""
        result = mud_tool.execute("unknown_tool", {})

        assert result["status"] == "error"
        assert "Unknown MUD tool" in result["error"]
        assert "unknown_tool" in result["error"]

    def test_invalid_tool_name_returns_error(self, mud_tool):
        """Invalid tool names should return error status."""
        result = mud_tool.execute("fly", {"destination": "moon"})

        assert result["status"] == "error"
        assert "Unknown MUD tool" in result["error"]


class TestDirectMethodCalls:
    """Tests for calling tool methods directly."""

    def test_say_method_directly(self, mud_tool):
        """Say method can be called directly."""
        result = mud_tool.say(message="Hello!")

        assert result["status"] == "success"
        assert result["tool"] == "say"

    def test_emote_method_directly(self, mud_tool):
        """Emote method can be called directly."""
        result = mud_tool.emote(action="waves")

        assert result["status"] == "success"
        assert result["tool"] == "emote"

    def test_look_method_without_target(self, mud_tool):
        """Look method works without target."""
        result = mud_tool.look()

        assert result["status"] == "success"
        assert result["args"] == {}

    def test_look_method_with_target(self, mud_tool):
        """Look method works with target."""
        result = mud_tool.look(target="door")

        assert result["status"] == "success"
        assert result["args"] == {"target": "door"}


class TestResultDataStructure:
    """Tests verifying the result data structure for MUDAction conversion."""

    def test_result_contains_all_required_fields(self, mud_tool):
        """All results should contain status, tool, args, and message."""
        result = mud_tool.execute("say", {"message": "Test"})

        assert "status" in result
        assert "tool" in result
        assert "args" in result
        assert "message" in result

    def test_args_can_be_used_for_mud_action(self, mud_tool):
        """Args should be suitable for MUDAction.to_command()."""
        result = mud_tool.execute("say", {"message": "Hello world"})

        # Verify args structure matches what MUDAction expects
        args = result["args"]
        assert isinstance(args, dict)
        assert "message" in args
        assert args["message"] == "Hello world"

    def test_multiple_actions_in_sequence(self, mud_tool):
        """Multiple tools can be called in sequence."""
        # Simulate Andi's typical response pattern
        results = [
            mud_tool.execute("emote", {"action": "looks up, silver band glowing gold"}),
            mud_tool.execute("say", {"message": "Happy New Year, Papa."}),
        ]

        for result in results:
            assert result["status"] == "success"

        assert results[0]["tool"] == "emote"
        assert results[1]["tool"] == "say"
