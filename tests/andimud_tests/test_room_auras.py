# tests/andimud_tests/test_room_auras.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for Room.get_room_auras() method.

These tests verify that room auras are correctly collected from objects
in the room, including objects inside containers (new functionality).
"""

import pytest
from unittest.mock import MagicMock, Mock
from evennia.objects.objects import DefaultExit


class TestRoomGetRoomAuras:
    """Tests for the Room.get_room_auras() method."""

    @pytest.fixture
    def mock_room(self):
        """Create a mock Room instance."""
        from andimud.typeclasses.rooms import Room

        room = MagicMock(spec=Room)
        room.contents = []

        # Bind the actual methods to the mock
        room.get_room_auras = lambda: Room.get_room_auras(room)
        room._collect_auras_from = lambda obj, auras, seen: Room._collect_auras_from(room, obj, auras, seen)

        return room

    @pytest.fixture
    def mock_object_with_aura(self):
        """Create a mock object that has get_room_auras method."""
        obj = MagicMock()
        obj.is_typeclass = MagicMock(return_value=False)
        obj.contents = []
        obj.get_room_auras = MagicMock(return_value=[
            {"name": "TEST_AURA", "source": "test_obj", "source_id": "123"}
        ])
        return obj

    @pytest.fixture
    def mock_object_without_aura(self):
        """Create a mock object without get_room_auras method."""
        obj = MagicMock()
        obj.is_typeclass = MagicMock(return_value=False)
        obj.contents = []
        # Remove the get_room_auras attribute
        delattr(obj, 'get_room_auras')
        return obj

    @pytest.fixture
    def mock_exit(self):
        """Create a mock exit object."""
        exit_obj = MagicMock(spec=DefaultExit)
        exit_obj.is_typeclass = MagicMock(return_value=True)
        return exit_obj

    def test_empty_room_returns_empty_list(self, mock_room):
        """Test that a room with no contents returns empty aura list."""
        mock_room.contents = []

        result = mock_room.get_room_auras()

        assert result == []

    def test_room_with_single_object_with_aura(self, mock_room, mock_object_with_aura):
        """Test that auras are collected from a single object."""
        mock_room.contents = [mock_object_with_aura]

        result = mock_room.get_room_auras()

        assert len(result) == 1
        assert result[0]["name"] == "TEST_AURA"
        assert result[0]["source"] == "test_obj"
        assert result[0]["source_id"] == "123"

    def test_room_with_object_without_aura_method(self, mock_room, mock_object_without_aura):
        """Test that objects without get_room_auras method are skipped."""
        mock_room.contents = [mock_object_without_aura]

        result = mock_room.get_room_auras()

        assert result == []

    def test_room_with_exit_object_skipped(self, mock_room, mock_exit):
        """Test that DefaultExit objects are skipped."""
        mock_exit.get_room_auras = MagicMock(return_value=[{"name": "EXIT_AURA"}])
        mock_room.contents = [mock_exit]

        result = mock_room.get_room_auras()

        assert result == []
        # Verify get_room_auras was never called on the exit
        mock_exit.get_room_auras.assert_not_called()

    def test_container_contents_are_collected(self, mock_room):
        """Test that auras from objects inside containers are collected.

        This is the NEW functionality - objects inside containers (like a terminal
        on a desk) should have their auras collected.
        """
        # Create a container (desk)
        container = MagicMock()
        container.is_typeclass = MagicMock(return_value=False)
        container.get_room_auras = MagicMock(return_value=[
            {"name": "DESK_AURA", "source": "desk", "source_id": "desk1"}
        ])

        # Create an object inside the container (terminal on desk)
        contained_obj = MagicMock()
        contained_obj.is_typeclass = MagicMock(return_value=False)
        contained_obj.get_room_auras = MagicMock(return_value=[
            {"name": "TERMINAL_AURA", "source": "terminal", "source_id": "term1"}
        ])

        container.contents = [contained_obj]
        mock_room.contents = [container]

        result = mock_room.get_room_auras()

        # Should have auras from both the container and its contents
        assert len(result) == 2
        aura_names = [aura["name"] for aura in result]
        assert "DESK_AURA" in aura_names
        assert "TERMINAL_AURA" in aura_names

    def test_exit_inside_container_skipped(self, mock_room, mock_exit):
        """Test that exits inside containers are skipped."""
        container = MagicMock()
        container.is_typeclass = MagicMock(return_value=False)
        container.contents = [mock_exit]

        mock_room.contents = [container]

        result = mock_room.get_room_auras()

        assert result == []

    def test_deduplication_same_name_source_sourceid(self, mock_room):
        """Test that duplicate auras are deduplicated by (name, source, source_id)."""
        obj1 = MagicMock()
        obj1.is_typeclass = MagicMock(return_value=False)
        obj1.contents = []
        obj1.get_room_auras = MagicMock(return_value=[
            {"name": "COMMON_AURA", "source": "obj", "source_id": "1"}
        ])

        obj2 = MagicMock()
        obj2.is_typeclass = MagicMock(return_value=False)
        obj2.contents = []
        obj2.get_room_auras = MagicMock(return_value=[
            {"name": "COMMON_AURA", "source": "obj", "source_id": "1"}
        ])

        mock_room.contents = [obj1, obj2]

        result = mock_room.get_room_auras()

        # Should only have one aura despite two sources
        assert len(result) == 1
        assert result[0]["name"] == "COMMON_AURA"

    def test_no_deduplication_different_source_id(self, mock_room):
        """Test that auras with same name/source but different source_id are not deduplicated."""
        obj1 = MagicMock()
        obj1.is_typeclass = MagicMock(return_value=False)
        obj1.contents = []
        obj1.get_room_auras = MagicMock(return_value=[
            {"name": "COMMON_AURA", "source": "obj", "source_id": "1"}
        ])

        obj2 = MagicMock()
        obj2.is_typeclass = MagicMock(return_value=False)
        obj2.contents = []
        obj2.get_room_auras = MagicMock(return_value=[
            {"name": "COMMON_AURA", "source": "obj", "source_id": "2"}
        ])

        mock_room.contents = [obj1, obj2]

        result = mock_room.get_room_auras()

        # Should have both auras since source_id differs
        assert len(result) == 2

    def test_string_aura_converted_to_dict(self, mock_room):
        """Test that string auras are converted to dict with 'name' key."""
        obj = MagicMock()
        obj.is_typeclass = MagicMock(return_value=False)
        obj.contents = []
        obj.get_room_auras = MagicMock(return_value=["STRING_AURA"])

        mock_room.contents = [obj]

        result = mock_room.get_room_auras()

        assert len(result) == 1
        assert result[0]["name"] == "STRING_AURA"
        assert result[0]["source"] == ""
        assert result[0]["source_id"] == ""

    def test_mixed_string_and_dict_auras(self, mock_room):
        """Test that mixed string and dict auras are handled correctly."""
        obj = MagicMock()
        obj.is_typeclass = MagicMock(return_value=False)
        obj.contents = []
        obj.get_room_auras = MagicMock(return_value=[
            "STRING_AURA",
            {"name": "DICT_AURA", "source": "obj", "source_id": "1"}
        ])

        mock_room.contents = [obj]

        result = mock_room.get_room_auras()

        assert len(result) == 2
        aura_names = [aura["name"] for aura in result]
        assert "STRING_AURA" in aura_names
        assert "DICT_AURA" in aura_names

    def test_non_dict_non_string_auras_skipped(self, mock_room):
        """Test that auras that are neither dict nor string are skipped."""
        obj = MagicMock()
        obj.is_typeclass = MagicMock(return_value=False)
        obj.contents = []
        obj.get_room_auras = MagicMock(return_value=[
            123,  # Number
            None,  # None
            ["list"],  # List
            {"name": "VALID_AURA"}  # Valid
        ])

        mock_room.contents = [obj]

        result = mock_room.get_room_auras()

        # Only the valid dict should be collected
        assert len(result) == 1
        assert result[0]["name"] == "VALID_AURA"

    def test_aura_without_name_skipped(self, mock_room):
        """Test that auras without a name field are skipped."""
        obj = MagicMock()
        obj.is_typeclass = MagicMock(return_value=False)
        obj.contents = []
        obj.get_room_auras = MagicMock(return_value=[
            {"source": "obj"},  # No name
            {"name": "VALID_AURA"}  # Valid
        ])

        mock_room.contents = [obj]

        result = mock_room.get_room_auras()

        assert len(result) == 1
        assert result[0]["name"] == "VALID_AURA"

    def test_aura_with_empty_name_skipped(self, mock_room):
        """Test that auras with empty name are skipped."""
        obj = MagicMock()
        obj.is_typeclass = MagicMock(return_value=False)
        obj.contents = []
        obj.get_room_auras = MagicMock(return_value=[
            {"name": ""},  # Empty name
            {"name": "   "},  # Whitespace only
            {"name": "VALID_AURA"}  # Valid
        ])

        mock_room.contents = [obj]

        result = mock_room.get_room_auras()

        assert len(result) == 1
        assert result[0]["name"] == "VALID_AURA"

    def test_exception_in_get_room_auras_handled(self, mock_room):
        """Test that exceptions in get_room_auras are caught and object is skipped."""
        obj_with_error = MagicMock()
        obj_with_error.is_typeclass = MagicMock(return_value=False)
        obj_with_error.contents = []
        obj_with_error.get_room_auras = MagicMock(side_effect=Exception("Test error"))

        obj_valid = MagicMock()
        obj_valid.is_typeclass = MagicMock(return_value=False)
        obj_valid.contents = []
        obj_valid.get_room_auras = MagicMock(return_value=[
            {"name": "VALID_AURA"}
        ])

        mock_room.contents = [obj_with_error, obj_valid]

        # Should not raise exception
        result = mock_room.get_room_auras()

        # Only the valid object's aura should be collected
        assert len(result) == 1
        assert result[0]["name"] == "VALID_AURA"

    def test_get_room_auras_returns_none_handled(self, mock_room):
        """Test that get_room_auras returning None is handled correctly."""
        obj = MagicMock()
        obj.is_typeclass = MagicMock(return_value=False)
        obj.contents = []
        obj.get_room_auras = MagicMock(return_value=None)

        mock_room.contents = [obj]

        result = mock_room.get_room_auras()

        assert result == []

    def test_get_room_auras_returns_empty_list(self, mock_room):
        """Test that get_room_auras returning empty list works correctly."""
        obj = MagicMock()
        obj.is_typeclass = MagicMock(return_value=False)
        obj.contents = []
        obj.get_room_auras = MagicMock(return_value=[])

        mock_room.contents = [obj]

        result = mock_room.get_room_auras()

        assert result == []

    def test_multiple_containers_with_contents(self, mock_room):
        """Test multiple containers each with their own contents."""
        # Container 1: Desk with terminal
        desk = MagicMock()
        desk.is_typeclass = MagicMock(return_value=False)
        desk.get_room_auras = MagicMock(return_value=[
            {"name": "DESK_AURA", "source": "desk", "source_id": "1"}
        ])

        terminal = MagicMock()
        terminal.is_typeclass = MagicMock(return_value=False)
        terminal.get_room_auras = MagicMock(return_value=[
            {"name": "TERMINAL_AURA", "source": "terminal", "source_id": "1"}
        ])

        desk.contents = [terminal]

        # Container 2: Shelf with book
        shelf = MagicMock()
        shelf.is_typeclass = MagicMock(return_value=False)
        shelf.get_room_auras = MagicMock(return_value=[
            {"name": "SHELF_AURA", "source": "shelf", "source_id": "1"}
        ])

        book = MagicMock()
        book.is_typeclass = MagicMock(return_value=False)
        book.get_room_auras = MagicMock(return_value=[
            {"name": "BOOK_AURA", "source": "book", "source_id": "1"}
        ])

        shelf.contents = [book]

        mock_room.contents = [desk, shelf]

        result = mock_room.get_room_auras()

        # Should have all 4 auras
        assert len(result) == 4
        aura_names = [aura["name"] for aura in result]
        assert "DESK_AURA" in aura_names
        assert "TERMINAL_AURA" in aura_names
        assert "SHELF_AURA" in aura_names
        assert "BOOK_AURA" in aura_names

    def test_nested_container_only_one_level_deep(self, mock_room):
        """Test that only one level of container nesting is processed.

        The implementation only checks obj.contents, not nested containers.
        This test documents that behavior.
        """
        # Top-level container
        container = MagicMock()
        container.is_typeclass = MagicMock(return_value=False)
        container.get_room_auras = MagicMock(return_value=[
            {"name": "CONTAINER_AURA", "source": "container", "source_id": "1"}
        ])

        # First-level nested object
        nested1 = MagicMock()
        nested1.is_typeclass = MagicMock(return_value=False)
        nested1.get_room_auras = MagicMock(return_value=[
            {"name": "NESTED1_AURA", "source": "nested1", "source_id": "1"}
        ])

        # Second-level nested object (should NOT be collected)
        nested2 = MagicMock()
        nested2.is_typeclass = MagicMock(return_value=False)
        nested2.get_room_auras = MagicMock(return_value=[
            {"name": "NESTED2_AURA", "source": "nested2", "source_id": "1"}
        ])

        nested1.contents = [nested2]
        container.contents = [nested1]
        mock_room.contents = [container]

        result = mock_room.get_room_auras()

        # Should only have container and first-level nested
        assert len(result) == 2
        aura_names = [aura["name"] for aura in result]
        assert "CONTAINER_AURA" in aura_names
        assert "NESTED1_AURA" in aura_names
        assert "NESTED2_AURA" not in aura_names

    def test_aura_fields_normalized_to_strings(self, mock_room):
        """Test that aura fields are normalized to strings."""
        obj = MagicMock()
        obj.is_typeclass = MagicMock(return_value=False)
        obj.contents = []
        obj.get_room_auras = MagicMock(return_value=[
            {"name": 123, "source": 456, "source_id": 789}  # Numbers
        ])

        mock_room.contents = [obj]

        result = mock_room.get_room_auras()

        assert len(result) == 1
        # All fields should be converted to strings
        assert result[0]["name"] == "123"
        assert result[0]["source"] == "456"
        assert result[0]["source_id"] == "789"

    def test_aura_fields_stripped_of_whitespace(self, mock_room):
        """Test that aura field values are stripped of whitespace."""
        obj = MagicMock()
        obj.is_typeclass = MagicMock(return_value=False)
        obj.contents = []
        obj.get_room_auras = MagicMock(return_value=[
            {"name": "  AURA_NAME  ", "source": " source ", "source_id": " 123 "}
        ])

        mock_room.contents = [obj]

        result = mock_room.get_room_auras()

        assert len(result) == 1
        assert result[0]["name"] == "AURA_NAME"
        assert result[0]["source"] == "source"
        assert result[0]["source_id"] == "123"

    def test_missing_source_fields_default_to_empty_string(self, mock_room):
        """Test that missing source/source_id fields default to empty string."""
        obj = MagicMock()
        obj.is_typeclass = MagicMock(return_value=False)
        obj.contents = []
        obj.get_room_auras = MagicMock(return_value=[
            {"name": "MINIMAL_AURA"}  # Only name provided
        ])

        mock_room.contents = [obj]

        result = mock_room.get_room_auras()

        assert len(result) == 1
        assert result[0]["name"] == "MINIMAL_AURA"
        assert result[0]["source"] == ""
        assert result[0]["source_id"] == ""

    def test_complex_scenario_mixed_objects(self, mock_room, mock_exit):
        """Test complex scenario with mix of objects, containers, and exits."""
        # Regular object with aura
        obj1 = MagicMock()
        obj1.is_typeclass = MagicMock(return_value=False)
        obj1.contents = []
        obj1.get_room_auras = MagicMock(return_value=[
            {"name": "OBJ1_AURA", "source": "obj1", "source_id": "1"}
        ])

        # Object without get_room_auras
        obj2 = MagicMock()
        obj2.is_typeclass = MagicMock(return_value=False)
        obj2.contents = []
        delattr(obj2, 'get_room_auras')

        # Container with two objects (one with aura, one without)
        container = MagicMock()
        container.is_typeclass = MagicMock(return_value=False)
        container.get_room_auras = MagicMock(return_value=[
            {"name": "CONTAINER_AURA", "source": "container", "source_id": "1"}
        ])

        contained1 = MagicMock()
        contained1.is_typeclass = MagicMock(return_value=False)
        contained1.get_room_auras = MagicMock(return_value=[
            {"name": "CONTAINED1_AURA", "source": "contained1", "source_id": "1"}
        ])

        contained2 = MagicMock()
        contained2.is_typeclass = MagicMock(return_value=False)
        delattr(contained2, 'get_room_auras')

        container.contents = [contained1, contained2]

        # Exit (should be skipped)

        mock_room.contents = [obj1, obj2, container, mock_exit]

        result = mock_room.get_room_auras()

        # Should have: obj1, container, contained1 (3 auras)
        assert len(result) == 3
        aura_names = [aura["name"] for aura in result]
        assert "OBJ1_AURA" in aura_names
        assert "CONTAINER_AURA" in aura_names
        assert "CONTAINED1_AURA" in aura_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
