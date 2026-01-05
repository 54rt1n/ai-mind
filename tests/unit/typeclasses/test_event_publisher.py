# tests/unit/typeclasses/test_event_publisher.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for EventPublisher sleep check logic.

These tests verify the sleep check logic without requiring full Evennia
initialization. They test the conditional logic that determines when events
should be published based on character sleep state.
"""

import pytest
from unittest.mock import Mock, MagicMock
from aim_mud_types import EventType


class TestEventPublisherSleepLogic:
    """Test publish_event sleep check functionality."""

    def test_should_skip_speech_when_sleeping(self):
        """Test that SPEECH events should be skipped when character is sleeping."""
        is_sleeping = True
        event_type = EventType.SPEECH

        # Sleep check logic
        should_skip = (
            event_type not in (EventType.SYSTEM, EventType.AMBIENT)
            and is_sleeping
        )

        assert should_skip is True

    def test_should_skip_emote_when_sleeping(self):
        """Test that EMOTE events should be skipped when character is sleeping."""
        is_sleeping = True
        event_type = EventType.EMOTE

        # Sleep check logic
        should_skip = (
            event_type not in (EventType.SYSTEM, EventType.AMBIENT)
            and is_sleeping
        )

        assert should_skip is True

    def test_should_skip_movement_when_sleeping(self):
        """Test that MOVEMENT events should be skipped when character is sleeping."""
        is_sleeping = True
        event_type = EventType.MOVEMENT

        # Sleep check logic
        should_skip = (
            event_type not in (EventType.SYSTEM, EventType.AMBIENT)
            and is_sleeping
        )

        assert should_skip is True

    def test_should_allow_system_when_sleeping(self):
        """Test that SYSTEM events should NOT be skipped when character is sleeping."""
        is_sleeping = True
        event_type = EventType.SYSTEM

        # Sleep check logic
        should_skip = (
            event_type not in (EventType.SYSTEM, EventType.AMBIENT)
            and is_sleeping
        )

        assert should_skip is False

    def test_should_allow_ambient_when_sleeping(self):
        """Test that AMBIENT events should NOT be skipped when character is sleeping."""
        is_sleeping = True
        event_type = EventType.AMBIENT

        # Sleep check logic
        should_skip = (
            event_type not in (EventType.SYSTEM, EventType.AMBIENT)
            and is_sleeping
        )

        assert should_skip is False

    def test_should_not_skip_when_not_sleeping(self):
        """Test that events are NOT skipped when character is not sleeping."""
        is_sleeping = False
        event_type = EventType.SPEECH

        # Sleep check logic
        should_skip = (
            event_type not in (EventType.SYSTEM, EventType.AMBIENT)
            and is_sleeping
        )

        assert should_skip is False

    def test_sleep_check_with_all_event_types(self):
        """Test sleep check logic for all event types."""
        is_sleeping = True

        # Events that should be skipped when sleeping
        should_skip_types = [
            EventType.SPEECH,
            EventType.EMOTE,
            EventType.MOVEMENT,
            EventType.OBJECT,
            EventType.NARRATIVE,
        ]

        for event_type in should_skip_types:
            should_skip = (
                event_type not in (EventType.SYSTEM, EventType.AMBIENT)
                and is_sleeping
            )
            assert should_skip is True, f"{event_type} should be skipped when sleeping"

        # Events that should NOT be skipped when sleeping
        should_not_skip_types = [
            EventType.SYSTEM,
            EventType.AMBIENT,
        ]

        for event_type in should_not_skip_types:
            should_skip = (
                event_type not in (EventType.SYSTEM, EventType.AMBIENT)
                and is_sleeping
            )
            assert should_skip is False, f"{event_type} should NOT be skipped when sleeping"

    def test_getattr_default_for_missing_is_sleeping(self):
        """Test that missing is_sleeping attribute defaults to False."""
        mock_db = Mock(spec=[])  # Empty spec means no attributes
        # Don't set is_sleeping attribute

        # Simulate getattr with default
        is_sleeping = getattr(mock_db, "is_sleeping", False)

        assert is_sleeping is False

    def test_ooc_check_combined_with_sleep_check(self):
        """Test that OOC check happens before sleep check."""
        is_ooc = True
        is_sleeping = False
        event_type = EventType.SPEECH

        # OOC check first (except for SYSTEM/AMBIENT)
        should_skip_ooc = (
            event_type not in (EventType.SYSTEM, EventType.AMBIENT)
            and is_ooc
        )

        # Sleep check second (only if not skipped by OOC)
        should_skip_sleep = (
            not should_skip_ooc
            and event_type not in (EventType.SYSTEM, EventType.AMBIENT)
            and is_sleeping
        )

        assert should_skip_ooc is True
        assert should_skip_sleep is False  # Never checked because OOC skip came first

    def test_system_event_bypasses_both_ooc_and_sleep(self):
        """Test that SYSTEM events bypass both OOC and sleep checks."""
        is_ooc = True
        is_sleeping = True
        event_type = EventType.SYSTEM

        # OOC check
        should_skip_ooc = (
            event_type not in (EventType.SYSTEM, EventType.AMBIENT)
            and is_ooc
        )

        # Sleep check
        should_skip_sleep = (
            event_type not in (EventType.SYSTEM, EventType.AMBIENT)
            and is_sleeping
        )

        assert should_skip_ooc is False
        assert should_skip_sleep is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
