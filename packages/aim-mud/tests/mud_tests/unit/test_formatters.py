# tests/mud_tests/unit/test_formatters.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for aim_mud_types.formatters module.

These tests verify the MUD event formatting functions produce correct output
for each event type and edge case.
"""

import pytest
from unittest.mock import MagicMock

from aim_mud_types import EventType, MUDEvent, ActorType
from aim_mud_types.formatters import (
    format_event,
    format_self_event,
    format_self_action_guidance,
    format_you_see_guidance,
    _format_emote_with_quotes,
    _format_code_event,
    _extract_object_from_content,
    _get_ground_items,
    _get_container_items,
    _get_current_inventory_summary,
)


# -----------------------------------------------------------------------------
# Fixtures for test data
# -----------------------------------------------------------------------------


@pytest.fixture
def speech_event():
    """Create a SPEECH event."""
    return MUDEvent(
        event_type=EventType.SPEECH,
        actor="Prax",
        actor_id="#prax_1",
        actor_type=ActorType.PLAYER,
        room_id="#123",
        content="Hello there!",
        metadata={},
    )


@pytest.fixture
def emote_event():
    """Create an EMOTE event without quotes."""
    return MUDEvent(
        event_type=EventType.EMOTE,
        actor="Andi",
        actor_id="#andi_1",
        actor_type=ActorType.AI,
        room_id="#123",
        content="waves enthusiastically.",
        metadata={},
    )


@pytest.fixture
def emote_event_with_quotes():
    """Create an EMOTE event with embedded quotes."""
    return MUDEvent(
        event_type=EventType.EMOTE,
        actor="Andi",
        actor_id="#andi_1",
        actor_type=ActorType.AI,
        room_id="#123",
        content='looks at you and smiles. "Hello!"',
        metadata={},
    )


@pytest.fixture
def movement_arrival_event():
    """Create a MOVEMENT arrival event."""
    return MUDEvent(
        event_type=EventType.MOVEMENT,
        actor="Prax",
        actor_id="#prax_1",
        actor_type=ActorType.PLAYER,
        room_id="#123",
        content="arrives from the north",
        metadata={"source_room_name": "The Hallway"},
    )


@pytest.fixture
def movement_departure_event():
    """Create a MOVEMENT departure event."""
    return MUDEvent(
        event_type=EventType.MOVEMENT,
        actor="Prax",
        actor_id="#prax_1",
        actor_type=ActorType.PLAYER,
        room_id="#123",
        content="leaves toward the south",
        metadata={"destination_room_name": "The Kitchen"},
    )


@pytest.fixture
def object_event():
    """Create an OBJECT interaction event."""
    return MUDEvent(
        event_type=EventType.OBJECT,
        actor="Andi",
        actor_id="#andi_1",
        actor_type=ActorType.AI,
        room_id="#123",
        content="picks up the apple",
        metadata={},
    )


@pytest.fixture
def ambient_event():
    """Create an AMBIENT event."""
    return MUDEvent(
        event_type=EventType.AMBIENT,
        actor="",
        actor_id="",
        actor_type=ActorType.SYSTEM,
        room_id="#123",
        content="A cool breeze drifts through the room.",
        metadata={},
    )


@pytest.fixture
def system_event():
    """Create a SYSTEM event."""
    return MUDEvent(
        event_type=EventType.SYSTEM,
        actor="",
        actor_id="",
        actor_type=ActorType.SYSTEM,
        room_id="#123",
        content="Server restart in 5 minutes.",
        metadata={},
    )


@pytest.fixture
def narrative_event():
    """Create a NARRATIVE event."""
    return MUDEvent(
        event_type=EventType.NARRATIVE,
        actor="",
        actor_id="",
        actor_type=ActorType.SYSTEM,
        room_id="#123",
        content="The sun sets over the horizon, painting the sky in shades of orange.",
        metadata={},
    )


@pytest.fixture
def terminal_event():
    """Create a TERMINAL event."""
    return MUDEvent(
        event_type=EventType.TERMINAL,
        actor="Andi",
        actor_id="#andi_1",
        actor_type=ActorType.AI,
        room_id="#123",
        content="[Terminal Alpha] web_search:\nResults for 'weather today'...",
        metadata={},
    )


@pytest.fixture
def code_action_event():
    """Create a CODE_ACTION event."""
    return MUDEvent(
        event_type=EventType.CODE_ACTION,
        actor="Andi",
        actor_id="#andi_1",
        actor_type=ActorType.AI,
        room_id="#123",
        content="total 48\ndrwxr-xr-x 2 user user 4096 Jan 1 00:00 .",
        metadata={"command_line": "ls -la"},
    )


@pytest.fixture
def code_file_event():
    """Create a CODE_FILE event."""
    return MUDEvent(
        event_type=EventType.CODE_FILE,
        actor="Andi",
        actor_id="#andi_1",
        actor_type=ActorType.AI,
        room_id="#123",
        content='print("Hello, world!")',
        metadata={"command_line": "cat script.py"},
    )


@pytest.fixture
def notification_event():
    """Create a NOTIFICATION event."""
    return MUDEvent(
        event_type=EventType.NOTIFICATION,
        actor="",
        actor_id="",
        actor_type=ActorType.SYSTEM,
        room_id="#123",
        content="*DING DONG* The doorbell rings!",
        metadata={},
    )


@pytest.fixture
def non_reactive_event():
    """Create a NON_REACTIVE event."""
    return MUDEvent(
        event_type=EventType.NON_REACTIVE,
        actor="Andi",
        actor_id="#andi_1",
        actor_type=ActorType.AI,
        room_id="#123",
        content="stretches and yawns contentedly.",
        metadata={},
    )


@pytest.fixture
def sleep_aware_event():
    """Create a SLEEP_AWARE event."""
    return MUDEvent(
        event_type=EventType.SLEEP_AWARE,
        actor="Andi",
        actor_id="#andi_1",
        actor_type=ActorType.AI,
        room_id="#123",
        content="falls asleep peacefully.",
        metadata={},
    )


# -----------------------------------------------------------------------------
# Tests for format_event
# -----------------------------------------------------------------------------


class TestFormatEvent:
    """Tests for the format_event function."""

    def test_speech_event(self, speech_event):
        """Test formatting a SPEECH event."""
        result = format_event(speech_event)
        assert result == 'Prax says, "Hello there!"'

    def test_emote_event_no_quotes(self, emote_event):
        """Test formatting an EMOTE event without quotes."""
        result = format_event(emote_event)
        assert result == "*Andi waves enthusiastically.*"

    def test_emote_event_with_quotes(self, emote_event_with_quotes):
        """Test formatting an EMOTE event with embedded quotes."""
        result = format_event(emote_event_with_quotes)
        assert result == '*Andi looks at you and smiles.* "Hello!"'

    def test_movement_arrival_with_source(self, movement_arrival_event):
        """Test formatting a MOVEMENT arrival event with source room."""
        result = format_event(movement_arrival_event)
        assert result == "*You see Prax arriving from The Hallway.*"

    def test_movement_arrival_without_source(self):
        """Test formatting a MOVEMENT arrival event without source room."""
        event = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Prax",
            actor_id="#prax_1",
            actor_type=ActorType.PLAYER,
            room_id="#123",
            content="enters the room",
            metadata={},
        )
        result = format_event(event)
        assert result == "*You see Prax has arrived.*"

    def test_movement_departure_with_destination(self, movement_departure_event):
        """Test formatting a MOVEMENT departure event with destination."""
        result = format_event(movement_departure_event)
        assert result == "*You see Prax leaving toward The Kitchen.*"

    def test_movement_departure_without_destination(self):
        """Test formatting a MOVEMENT departure event without destination."""
        event = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Prax",
            actor_id="#prax_1",
            actor_type=ActorType.PLAYER,
            room_id="#123",
            content="leaves the room",
            metadata={},
        )
        result = format_event(event)
        assert result == "*You see Prax leave.*"

    def test_object_event(self, object_event):
        """Test formatting an OBJECT event."""
        result = format_event(object_event)
        assert result == "*Andi picks up the apple*"

    def test_ambient_event(self, ambient_event):
        """Test formatting an AMBIENT event."""
        result = format_event(ambient_event)
        assert result == "A cool breeze drifts through the room."

    def test_system_event(self, system_event):
        """Test formatting a SYSTEM event."""
        result = format_event(system_event)
        assert result == "[System] Server restart in 5 minutes."

    def test_narrative_event(self, narrative_event):
        """Test formatting a NARRATIVE event."""
        result = format_event(narrative_event)
        assert result == "The sun sets over the horizon, painting the sky in shades of orange."

    def test_terminal_event(self, terminal_event):
        """Test formatting a TERMINAL event - content passes through directly."""
        result = format_event(terminal_event)
        assert result == "[Terminal Alpha] web_search:\nResults for 'weather today'..."

    def test_code_action_event(self, code_action_event):
        """Test formatting a CODE_ACTION event."""
        result = format_event(code_action_event)
        assert "[~~ You have executed `ls -la`. ~~]" in result
        assert "Command Output:" in result
        assert "total 48" in result
        assert "[~~ End of Command Output ~~]" in result

    def test_code_file_event(self, code_file_event):
        """Test formatting a CODE_FILE event."""
        result = format_event(code_file_event)
        assert "[~~ You have executed `cat script.py`. ~~]" in result
        assert "File Contents:" in result
        assert 'print("Hello, world!")' in result

    def test_notification_event(self, notification_event):
        """Test formatting a NOTIFICATION event."""
        result = format_event(notification_event)
        assert result == "*DING DONG* The doorbell rings!"

    def test_non_reactive_event(self, non_reactive_event):
        """Test formatting a NON_REACTIVE event."""
        result = format_event(non_reactive_event)
        assert result == "*Andi stretches and yawns contentedly.*"

    def test_sleep_aware_event(self, sleep_aware_event):
        """Test formatting a SLEEP_AWARE event."""
        result = format_event(sleep_aware_event)
        assert result == "*Andi falls asleep peacefully.*"

    def test_first_person_delegates_to_format_self_event(self, movement_arrival_event):
        """Test that first_person=True delegates to format_self_event."""
        # Create a movement event with destination metadata (required for self-event)
        event = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Andi",
            actor_id="#andi_1",
            actor_type=ActorType.AI,
            room_id="#123",
            content="arrives",
            metadata={"destination_room_name": "The Kitchen"},
        )
        result = format_event(event, first_person=True)
        assert result == "You successfully moved to The Kitchen."


# -----------------------------------------------------------------------------
# Tests for _format_emote_with_quotes
# -----------------------------------------------------------------------------


class TestFormatEmoteWithQuotes:
    """Tests for the _format_emote_with_quotes helper function."""

    def test_emote_without_quotes(self):
        """Test emote with no quotes wraps entire content."""
        result = _format_emote_with_quotes("Andi", "waves hello")
        assert result == "*Andi waves hello*"

    def test_emote_with_trailing_quotes(self):
        """Test emote with quotes at the end separates action from speech."""
        result = _format_emote_with_quotes("Andi", 'smiles warmly. "Hello there!"')
        assert result == '*Andi smiles warmly.* "Hello there!"'

    def test_emote_with_leading_quotes(self):
        """Test emote with quotes at the start just wraps actor."""
        result = _format_emote_with_quotes("Andi", '"Hello!" she says.')
        assert result == '*Andi* "Hello!" she says.'

    def test_emote_with_empty_action_before_quotes(self):
        """Test emote where quote appears immediately after actor."""
        result = _format_emote_with_quotes("Andi", '"Just the quote."')
        assert result == '*Andi* "Just the quote."'

    def test_emote_with_multiple_quotes(self):
        """Test emote with multiple quote pairs uses first quote as split point."""
        result = _format_emote_with_quotes("Andi", 'says "Hello" and then "Goodbye"')
        assert result == '*Andi says* "Hello" and then "Goodbye"'


# -----------------------------------------------------------------------------
# Tests for format_self_event
# -----------------------------------------------------------------------------


class TestFormatSelfEvent:
    """Tests for the format_self_event function."""

    def test_movement_self_event(self):
        """Test formatting a movement self-event."""
        event = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Andi",
            actor_id="#andi_1",
            actor_type=ActorType.AI,
            room_id="#123",
            content="moves",
            metadata={"destination_room_name": "The Kitchen"},
        )
        result = format_self_event(event)
        assert result == "You successfully moved to The Kitchen."

    def test_movement_self_event_no_destination(self):
        """Test formatting a movement self-event without destination metadata."""
        event = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Andi",
            actor_id="#andi_1",
            actor_type=ActorType.AI,
            room_id="#123",
            content="moves",
            metadata={},
        )
        result = format_self_event(event)
        assert result == "You successfully moved to somewhere."

    def test_object_pickup_self_event(self):
        """Test formatting an object pickup self-event."""
        event = MUDEvent(
            event_type=EventType.OBJECT,
            actor="Andi",
            actor_id="#andi_1",
            actor_type=ActorType.AI,
            room_id="#123",
            content="picks up the apple",
            target="apple",
            metadata={},
        )
        result = format_self_event(event)
        assert result == "You successfully picked up apple."

    def test_object_pickup_from_container(self):
        """Test formatting an object pickup from container."""
        event = MUDEvent(
            event_type=EventType.OBJECT,
            actor="Andi",
            actor_id="#andi_1",
            actor_type=ActorType.AI,
            room_id="#123",
            content="takes the apple from the basket",
            target="apple",
            metadata={"container_name": "the basket"},
        )
        result = format_self_event(event)
        assert result == "You successfully took apple from the basket."

    def test_object_drop_self_event(self):
        """Test formatting an object drop self-event."""
        event = MUDEvent(
            event_type=EventType.OBJECT,
            actor="Andi",
            actor_id="#andi_1",
            actor_type=ActorType.AI,
            room_id="#123",
            content="drops the apple",
            target="apple",
            metadata={},
        )
        result = format_self_event(event)
        assert result == "You successfully dropped apple."

    def test_object_give_self_event(self):
        """Test formatting an object give self-event."""
        event = MUDEvent(
            event_type=EventType.OBJECT,
            actor="Andi",
            actor_id="#andi_1",
            actor_type=ActorType.AI,
            room_id="#123",
            content="gives the apple to Prax",
            target="apple",
            metadata={"target_name": "Prax"},
        )
        result = format_self_event(event)
        assert result == "You successfully gave apple to Prax."

    def test_object_put_self_event(self):
        """Test formatting an object put self-event."""
        event = MUDEvent(
            event_type=EventType.OBJECT,
            actor="Andi",
            actor_id="#andi_1",
            actor_type=ActorType.AI,
            room_id="#123",
            content="puts the apple in the basket",
            target="apple",
            metadata={"container_name": "the basket"},
        )
        result = format_self_event(event)
        assert result == "You successfully put apple in the basket."

    def test_emote_self_event(self):
        """Test formatting an emote self-event."""
        event = MUDEvent(
            event_type=EventType.EMOTE,
            actor="Andi",
            actor_id="#andi_1",
            actor_type=ActorType.AI,
            room_id="#123",
            content="waves hello",
            metadata={},
        )
        result = format_self_event(event)
        assert result == "You successfully expressed: waves hello"


# -----------------------------------------------------------------------------
# Tests for _format_code_event
# -----------------------------------------------------------------------------


class TestFormatCodeEvent:
    """Tests for the _format_code_event helper function."""

    def test_code_action_with_command_line(self):
        """Test formatting a code action event with command_line metadata."""
        event = MUDEvent(
            event_type=EventType.CODE_ACTION,
            actor="Andi",
            actor_id="#andi_1",
            actor_type=ActorType.AI,
            room_id="#123",
            content="file1.py\nfile2.py",
            metadata={"command_line": "ls *.py"},
        )
        result = _format_code_event(event)
        assert "[~~ You have executed `ls *.py`. ~~]" in result
        assert "Command Output:" in result
        assert "file1.py\nfile2.py" in result

    def test_code_action_with_command_fallback(self):
        """Test formatting a code action event with command metadata fallback."""
        event = MUDEvent(
            event_type=EventType.CODE_ACTION,
            actor="Andi",
            actor_id="#andi_1",
            actor_type=ActorType.AI,
            room_id="#123",
            content="output",
            metadata={"command": "pwd"},
        )
        result = _format_code_event(event)
        assert "[~~ You have executed `pwd`. ~~]" in result

    def test_code_action_no_metadata(self):
        """Test formatting a code action event without metadata."""
        event = MUDEvent(
            event_type=EventType.CODE_ACTION,
            actor="Andi",
            actor_id="#andi_1",
            actor_type=ActorType.AI,
            room_id="#123",
            content="output",
            metadata={},
        )
        result = _format_code_event(event)
        assert "[~~ You have executed `command`. ~~]" in result

    def test_code_file_event(self):
        """Test formatting a code file event uses File Contents label."""
        event = MUDEvent(
            event_type=EventType.CODE_FILE,
            actor="Andi",
            actor_id="#andi_1",
            actor_type=ActorType.AI,
            room_id="#123",
            content="# Python code",
            metadata={"command_line": "cat file.py"},
        )
        result = _format_code_event(event)
        assert "File Contents:" in result
        assert "Command Output:" not in result

    def test_code_action_empty_content(self):
        """Test formatting a code action event with empty content."""
        event = MUDEvent(
            event_type=EventType.CODE_ACTION,
            actor="Andi",
            actor_id="#andi_1",
            actor_type=ActorType.AI,
            room_id="#123",
            content="",
            metadata={"command_line": "echo"},
        )
        result = _format_code_event(event)
        assert "(no output)" in result


# -----------------------------------------------------------------------------
# Tests for _extract_object_from_content
# -----------------------------------------------------------------------------


class TestExtractObjectFromContent:
    """Tests for the _extract_object_from_content helper function.

    Note: The function skips the first 2 words (actor name + verb) and returns
    the rest. For compound verbs like "picks up", "up" is included in the result.
    """

    def test_extract_from_full_sentence(self):
        """Test extracting object from full action sentence.

        The function skips first 2 words: 'Andi picks' -> 'up the red apple'
        """
        result = _extract_object_from_content("Andi picks up the red apple")
        assert result == "up the red apple"

    def test_extract_with_trailing_period(self):
        """Test extracting object removes trailing period."""
        result = _extract_object_from_content("Andi picks up the apple.")
        assert result == "up the apple"

    def test_extract_with_asterisks(self):
        """Test extracting object removes asterisks."""
        result = _extract_object_from_content("*Andi picks up the apple*")
        assert result == "up the apple"

    def test_extract_simple_verb(self):
        """Test extracting with a simple verb (not compound)."""
        result = _extract_object_from_content("Andi drops the apple")
        assert result == "the apple"

    def test_extract_short_content(self):
        """Test extracting from content with fewer than 3 words."""
        result = _extract_object_from_content("picks up")
        assert result == "something"


# -----------------------------------------------------------------------------
# Tests for world state helpers
# -----------------------------------------------------------------------------


class TestGetGroundItems:
    """Tests for the _get_ground_items helper function."""

    def test_no_world_state(self):
        """Test with None world_state."""
        result = _get_ground_items(None)
        assert result == []

    def test_empty_entities(self):
        """Test with empty entities list."""
        world_state = MagicMock()
        world_state.entities_present = []
        result = _get_ground_items(world_state)
        assert result == []

    def test_filters_players_and_npcs(self):
        """Test that players and NPCs are filtered out."""
        world_state = MagicMock()
        player = MagicMock()
        player.entity_type = "player"
        player.is_self = False
        player.name = "Prax"

        item = MagicMock()
        item.entity_type = "object"
        item.is_self = False
        item.contents = None
        item.name = "apple"

        world_state.entities_present = [player, item]
        result = _get_ground_items(world_state)
        assert result == ["apple"]

    def test_filters_self_items(self):
        """Test that self items are filtered out."""
        world_state = MagicMock()
        item = MagicMock()
        item.entity_type = "object"
        item.is_self = True
        item.contents = None
        item.name = "my item"

        world_state.entities_present = [item]
        result = _get_ground_items(world_state)
        assert result == []

    def test_filters_containers(self):
        """Test that containers (items with contents) are filtered out."""
        world_state = MagicMock()
        container = MagicMock()
        container.entity_type = "object"
        container.is_self = False
        container.contents = ["something"]
        container.name = "basket"

        item = MagicMock()
        item.entity_type = "object"
        item.is_self = False
        item.contents = None
        item.name = "apple"

        world_state.entities_present = [container, item]
        result = _get_ground_items(world_state)
        assert result == ["apple"]


class TestGetContainerItems:
    """Tests for the _get_container_items helper function."""

    def test_no_world_state(self):
        """Test with None world_state."""
        result = _get_container_items(None)
        assert result == {}

    def test_empty_entities(self):
        """Test with empty entities list."""
        world_state = MagicMock()
        world_state.entities_present = []
        result = _get_container_items(world_state)
        assert result == {}

    def test_returns_containers_with_contents(self):
        """Test that containers with contents are returned."""
        world_state = MagicMock()
        container = MagicMock()
        container.entity_type = "object"
        container.is_self = False
        container.contents = ["apple", "orange"]
        container.name = "basket"

        world_state.entities_present = [container]
        result = _get_container_items(world_state)
        assert result == {"basket": ["apple", "orange"]}

    def test_unnamed_container(self):
        """Test container without a name uses 'something'."""
        world_state = MagicMock()
        container = MagicMock()
        container.entity_type = "object"
        container.is_self = False
        container.contents = ["item"]
        container.name = None

        world_state.entities_present = [container]
        result = _get_container_items(world_state)
        assert result == {"something": ["item"]}


class TestGetCurrentInventorySummary:
    """Tests for the _get_current_inventory_summary helper function."""

    def test_no_world_state(self):
        """Test with None world_state."""
        result = _get_current_inventory_summary(None)
        assert result == "none"

    def test_no_inventory_attribute(self):
        """Test with world_state lacking inventory attribute."""
        world_state = MagicMock(spec=[])  # No attributes
        result = _get_current_inventory_summary(world_state)
        assert result == "none"

    def test_empty_inventory(self):
        """Test with empty inventory."""
        world_state = MagicMock()
        world_state.inventory = []
        result = _get_current_inventory_summary(world_state)
        assert result == "none"

    def test_single_item(self):
        """Test with single item in inventory."""
        world_state = MagicMock()
        item = MagicMock()
        item.name = "apple"
        world_state.inventory = [item]
        result = _get_current_inventory_summary(world_state)
        assert result == "apple"

    def test_multiple_items(self):
        """Test with multiple items in inventory."""
        world_state = MagicMock()
        item1 = MagicMock()
        item1.name = "apple"
        item2 = MagicMock()
        item2.name = "key"
        world_state.inventory = [item1, item2]
        result = _get_current_inventory_summary(world_state)
        assert result == "apple, key"


# -----------------------------------------------------------------------------
# Tests for format_you_see_guidance
# -----------------------------------------------------------------------------


class TestFormatYouSeeGuidance:
    """Tests for the format_you_see_guidance function."""

    def test_no_world_state(self):
        """Test with None world_state."""
        result = format_you_see_guidance(None)
        assert result == ""

    def test_no_room_state(self):
        """Test with world_state lacking room_state."""
        world_state = MagicMock()
        world_state.room_state = None
        result = format_you_see_guidance(world_state)
        assert result == ""

    def test_basic_room(self):
        """Test formatting with basic room info."""
        world_state = MagicMock()
        world_state.room_state = MagicMock()
        world_state.room_state.name = "The Kitchen"
        world_state.room_state.description = "A warm and cozy kitchen."
        world_state.entities_present = []
        world_state.inventory = []

        result = format_you_see_guidance(world_state)

        assert "[~~ You See ~~]" in result
        assert "You are in The Kitchen." in result
        assert "A warm and cozy kitchen." in result
        assert "[/~~ You See ~~]" in result

    def test_room_with_people(self):
        """Test formatting with people in the room."""
        world_state = MagicMock()
        world_state.room_state = MagicMock()
        world_state.room_state.name = "The Kitchen"
        world_state.room_state.description = ""

        person = MagicMock()
        person.entity_type = "player"
        person.name = "Prax"
        person.contents = ["sword"]

        world_state.entities_present = [person]
        world_state.inventory = []

        result = format_you_see_guidance(world_state)

        assert "People here:" in result
        assert "* Prax" in result
        assert "- Prax is carrying: [sword]" in result


# -----------------------------------------------------------------------------
# Tests for format_self_action_guidance
# -----------------------------------------------------------------------------


class TestFormatSelfActionGuidance:
    """Tests for the format_self_action_guidance function."""

    def test_empty_list(self):
        """Test with empty self_actions list."""
        result = format_self_action_guidance([])
        assert result == ""

    def test_single_movement_action(self):
        """Test formatting single movement action."""
        event = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Andi",
            actor_id="#andi_1",
            actor_type=ActorType.AI,
            room_id="#123",
            content="moves",
            metadata={
                "source_room_name": "The Hallway",
                "destination_room_name": "The Kitchen",
            },
        )

        result = format_self_action_guidance([event])

        assert "[== Your Turn ==]" in result
        assert "You started at The Hallway" in result
        assert "You decided to move to a new location." in result
        assert "Now you are at The Kitchen" in result

    def test_single_pickup_action(self):
        """Test formatting single pickup action."""
        event = MUDEvent(
            event_type=EventType.OBJECT,
            actor="Andi",
            actor_id="#andi_1",
            actor_type=ActorType.AI,
            room_id="#123",
            content="picks up the apple",
            target="apple",
            metadata={},
        )

        world_state = MagicMock()
        item = MagicMock()
        item.name = "apple"
        world_state.inventory = [item]

        result = format_self_action_guidance([event], world_state)

        assert "[== Your Turn ==]" in result
        assert "decided to pick up:" in result
        assert "apple" in result
        assert "You are now carrying: apple" in result

    def test_multiple_actions(self):
        """Test formatting multiple actions."""
        event1 = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Andi",
            actor_id="#andi_1",
            actor_type=ActorType.AI,
            room_id="#123",
            content="moves",
            metadata={
                "source_room_name": "Room A",
                "destination_room_name": "Room B",
            },
        )
        event2 = MUDEvent(
            event_type=EventType.OBJECT,
            actor="Andi",
            actor_id="#andi_1",
            actor_type=ActorType.AI,
            room_id="#124",
            content="picks up the key",
            target="key",
            metadata={},
        )

        result = format_self_action_guidance([event1, event2])

        assert "CONGRATULATIONS: YOU WERE SUCCESSFUL" in result
        assert "1. You successfully moved from Room A to Room B" in result
        assert "2. You successfully picked up key" in result


# -----------------------------------------------------------------------------
# Tests for backwards compatibility
# -----------------------------------------------------------------------------


class TestBackwardsCompatibility:
    """Tests to verify backwards compatibility of the re-export module."""

    def test_import_from_old_location(self):
        """Test that imports from the old location still work."""
        from andimud_worker.adapter.event import (
            format_event,
            format_self_event,
            format_self_action_guidance,
            format_you_see_guidance,
        )

        # Just verify they're the same functions
        from aim_mud_types.formatters import (
            format_event as new_format_event,
            format_self_event as new_format_self_event,
        )

        assert format_event is new_format_event
        assert format_self_event is new_format_self_event

    def test_import_from_aim_mud_types(self):
        """Test that imports from aim_mud_types work."""
        from aim_mud_types import (
            format_event,
            format_self_event,
            format_self_action_guidance,
            format_you_see_guidance,
        )

        # Verify they're callable
        assert callable(format_event)
        assert callable(format_self_event)
        assert callable(format_self_action_guidance)
        assert callable(format_you_see_guidance)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
