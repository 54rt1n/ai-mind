# tests/andimud_tests/test_new_characters.py
"""Tests for Nova, Corroded, Lin Yu, and Tiberius character classes.

These tests verify that new AI character classes are properly configured
with correct agent IDs, permissions, and behavior.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Try to import character classes - skip all tests if not available
EVENNIA_AVAILABLE = False
NovaCharacter = None
CorrodedCharacter = None
LinYuCharacter = None
TiberiusCharacter = None
ActorType = None

try:
    from andimud.typeclasses.ai_characters import (
        NovaCharacter,
        CorrodedCharacter,
        LinYuCharacter,
        TiberiusCharacter,
    )
    from aim_mud_types import ActorType
    EVENNIA_AVAILABLE = True
except Exception:
    # Skip all tests if Evennia is not available or Django is not configured
    pass

if not EVENNIA_AVAILABLE:
    pytestmark = pytest.mark.skip(
        reason="Evennia character classes not available or Django not configured"
    )


@pytest.fixture
def mock_character_base():
    """Create a mock for character testing."""
    char = MagicMock()
    char.db = MagicMock()
    char.locks = MagicMock()
    char.permissions = MagicMock()
    return char


class TestNovaCharacter:
    """Tests for Nova's character class."""

    def test_at_object_creation_sets_agent_id(self, mock_character_base):
        """Test that at_object_creation sets the correct agent_id."""
        nova = NovaCharacter()
        nova.db = mock_character_base.db
        nova.locks = mock_character_base.locks
        nova.permissions = mock_character_base.permissions

        # Mock the parent class's at_object_creation
        with patch.object(NovaCharacter.__bases__[0], 'at_object_creation'):
            nova.at_object_creation()

        # Verify agent_id is set
        assert nova.db.agent_id == "nova"

    def test_at_object_creation_sets_permissions(self, mock_character_base):
        """Test that at_object_creation sets correct permissions."""
        nova = NovaCharacter()
        nova.db = mock_character_base.db
        nova.locks = mock_character_base.locks
        nova.permissions = mock_character_base.permissions

        with patch.object(NovaCharacter.__bases__[0], 'at_object_creation'):
            nova.at_object_creation()

        # Verify permissions
        nova.locks.add.assert_called_with("control:perm(Developer)")
        assert nova.permissions.add.call_count == 2
        nova.permissions.add.assert_any_call("Developer")
        nova.permissions.add.assert_any_call("Builder")


class TestCorrodedCharacter:
    """Tests for Corroded's character class."""

    def test_at_object_creation_sets_agent_id(self, mock_character_base):
        """Test that at_object_creation sets the correct agent_id."""
        corroded = CorrodedCharacter()
        corroded.db = mock_character_base.db
        corroded.locks = mock_character_base.locks
        corroded.permissions = mock_character_base.permissions

        with patch.object(CorrodedCharacter.__bases__[0], 'at_object_creation'):
            corroded.at_object_creation()

        assert corroded.db.agent_id == "corroded"

    def test_at_object_creation_sets_permissions(self, mock_character_base):
        """Test that at_object_creation sets correct permissions."""
        corroded = CorrodedCharacter()
        corroded.db = mock_character_base.db
        corroded.locks = mock_character_base.locks
        corroded.permissions = mock_character_base.permissions

        with patch.object(CorrodedCharacter.__bases__[0], 'at_object_creation'):
            corroded.at_object_creation()

        corroded.locks.add.assert_called_with("control:perm(Developer)")
        assert corroded.permissions.add.call_count == 2
        corroded.permissions.add.assert_any_call("Developer")
        corroded.permissions.add.assert_any_call("Builder")


class TestLinYuCharacter:
    """Tests for Lin Yu's character class."""

    def test_at_object_creation_sets_agent_id(self, mock_character_base):
        """Test that at_object_creation sets the correct agent_id."""
        linyu = LinYuCharacter()
        linyu.db = mock_character_base.db
        linyu.locks = mock_character_base.locks
        linyu.permissions = mock_character_base.permissions

        with patch.object(LinYuCharacter.__bases__[0], 'at_object_creation'):
            linyu.at_object_creation()

        assert linyu.db.agent_id == "linyu"

    def test_at_object_creation_sets_permissions(self, mock_character_base):
        """Test that at_object_creation sets correct permissions."""
        linyu = LinYuCharacter()
        linyu.db = mock_character_base.db
        linyu.locks = mock_character_base.locks
        linyu.permissions = mock_character_base.permissions

        with patch.object(LinYuCharacter.__bases__[0], 'at_object_creation'):
            linyu.at_object_creation()

        linyu.locks.add.assert_called_with("control:perm(Developer)")
        assert linyu.permissions.add.call_count == 2
        linyu.permissions.add.assert_any_call("Developer")
        linyu.permissions.add.assert_any_call("Builder")


class TestTiberiusCharacter:
    """Tests for Tiberius's character class."""

    def test_at_object_creation_sets_agent_id(self, mock_character_base):
        """Test that at_object_creation sets the correct agent_id."""
        tiberius = TiberiusCharacter()
        tiberius.db = mock_character_base.db
        tiberius.locks = mock_character_base.locks
        tiberius.permissions = mock_character_base.permissions

        with patch.object(TiberiusCharacter.__bases__[0], 'at_object_creation'):
            tiberius.at_object_creation()

        assert tiberius.db.agent_id == "tiberius"

    def test_at_object_creation_sets_permissions(self, mock_character_base):
        """Test that at_object_creation sets correct permissions."""
        tiberius = TiberiusCharacter()
        tiberius.db = mock_character_base.db
        tiberius.locks = mock_character_base.locks
        tiberius.permissions = mock_character_base.permissions

        with patch.object(TiberiusCharacter.__bases__[0], 'at_object_creation'):
            tiberius.at_object_creation()

        tiberius.locks.add.assert_called_with("control:perm(Developer)")
        assert tiberius.permissions.add.call_count == 2
        tiberius.permissions.add.assert_any_call("Developer")
        tiberius.permissions.add.assert_any_call("Builder")
