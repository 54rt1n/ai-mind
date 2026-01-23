# packages/aim-mud/tests/mud_tests/unit/test_coordination.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for coordination types and enums.

Tests the TurnReason enum's is_immediate_command() method which identifies
immediate commands (FLUSH, CLEAR, NEW) that should execute with EXECUTE status.
"""

import pytest

from aim_mud_types.models.coordination import TurnReason, TurnRequestStatus


class TestTurnReasonIsImmediateCommand:
    """Tests for TurnReason.is_immediate_command() method."""

    def test_flush_is_immediate_command(self):
        """Test that FLUSH is identified as an immediate command."""
        assert TurnReason.FLUSH.is_immediate_command() is True

    def test_clear_is_immediate_command(self):
        """Test that CLEAR is identified as an immediate command."""
        assert TurnReason.CLEAR.is_immediate_command() is True

    def test_new_is_immediate_command(self):
        """Test that NEW is identified as an immediate command."""
        assert TurnReason.NEW.is_immediate_command() is True

    def test_events_is_not_immediate_command(self):
        """Test that EVENTS is not an immediate command."""
        assert TurnReason.EVENTS.is_immediate_command() is False

    def test_idle_is_not_immediate_command(self):
        """Test that IDLE is not an immediate command."""
        assert TurnReason.IDLE.is_immediate_command() is False

    def test_dream_is_not_immediate_command(self):
        """Test that DREAM is not an immediate command."""
        assert TurnReason.DREAM.is_immediate_command() is False

    def test_agent_is_not_immediate_command(self):
        """Test that AGENT is not an immediate command."""
        assert TurnReason.AGENT.is_immediate_command() is False

    def test_choose_is_not_immediate_command(self):
        """Test that CHOOSE is not an immediate command."""
        assert TurnReason.CHOOSE.is_immediate_command() is False

    def test_retry_is_not_immediate_command(self):
        """Test that RETRY is not an immediate command."""
        assert TurnReason.RETRY.is_immediate_command() is False

    def test_think_is_not_immediate_command(self):
        """Test that THINK is not an immediate command."""
        assert TurnReason.THINK.is_immediate_command() is False

    def test_all_reasons_handled(self):
        """Test that all TurnReason values are covered by the method."""
        # Verify that all enum values return either True or False
        for reason in TurnReason:
            result = reason.is_immediate_command()
            assert isinstance(result, bool), f"{reason} should return bool"

    def test_immediate_commands_set(self):
        """Test that exactly three reasons are immediate commands."""
        immediate_commands = [r for r in TurnReason if r.is_immediate_command()]
        assert len(immediate_commands) == 3
        assert set(immediate_commands) == {
            TurnReason.FLUSH,
            TurnReason.CLEAR,
            TurnReason.NEW,
        }


class TestTurnReasonThink:
    """Tests for TurnReason.THINK enum value."""

    def test_think_exists(self):
        """Test THINK is a valid TurnReason."""
        assert TurnReason.THINK == "think"

    def test_think_value(self):
        """Test THINK enum value is correct."""
        assert TurnReason.THINK.value == "think"

    def test_think_in_enum(self):
        """Test THINK is in the TurnReason enum."""
        assert TurnReason.THINK in TurnReason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
