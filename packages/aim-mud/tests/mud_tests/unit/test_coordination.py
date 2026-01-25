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


class TestTurnRequestStatusPending:
    """Tests for TurnRequestStatus.PENDING enum value."""

    def test_pending_exists(self):
        """Test PENDING is a valid TurnRequestStatus."""
        assert TurnRequestStatus.PENDING == "pending"

    def test_pending_value(self):
        """Test PENDING enum value is correct."""
        assert TurnRequestStatus.PENDING.value == "pending"

    def test_pending_in_enum(self):
        """Test PENDING is in the TurnRequestStatus enum."""
        assert TurnRequestStatus.PENDING in TurnRequestStatus


class TestTurnRequestIsAvailableForAssignment:
    """Tests for MUDTurnRequest.is_available_for_assignment() with PENDING status."""

    def test_pending_is_not_available(self):
        """Agent waiting for action echo (PENDING) should not be available."""
        from aim_mud_types.models.coordination import MUDTurnRequest

        request = MUDTurnRequest(
            turn_id="test-turn",
            status=TurnRequestStatus.PENDING,
            sequence_id=1
        )

        assert request.is_available_for_assignment() is False

    def test_in_progress_is_not_available(self):
        """Agent in progress should not be available."""
        from aim_mud_types.models.coordination import MUDTurnRequest

        request = MUDTurnRequest(
            turn_id="test-turn",
            status=TurnRequestStatus.IN_PROGRESS,
            sequence_id=1
        )

        assert request.is_available_for_assignment() is False

    def test_done_is_available(self):
        """Agent that completed turn should be available."""
        from aim_mud_types.models.coordination import MUDTurnRequest

        request = MUDTurnRequest(
            turn_id="test-turn",
            status=TurnRequestStatus.DONE,
            sequence_id=1
        )

        assert request.is_available_for_assignment() is True

    def test_ready_is_available(self):
        """Agent in ready state should be available."""
        from aim_mud_types.models.coordination import MUDTurnRequest

        request = MUDTurnRequest(
            turn_id="test-turn",
            status=TurnRequestStatus.READY,
            sequence_id=1
        )

        assert request.is_available_for_assignment() is True

    def test_busy_statuses_are_not_available(self):
        """All busy statuses should mark agent as unavailable."""
        from aim_mud_types.models.coordination import MUDTurnRequest

        busy_statuses = [
            TurnRequestStatus.ASSIGNED,
            TurnRequestStatus.IN_PROGRESS,
            TurnRequestStatus.PENDING,
            TurnRequestStatus.ABORT_REQUESTED,
            TurnRequestStatus.EXECUTING,
            TurnRequestStatus.EXECUTE,
        ]

        for status in busy_statuses:
            request = MUDTurnRequest(
                turn_id="test-turn",
                status=status,
                sequence_id=1
            )
            assert request.is_available_for_assignment() is False, (
                f"Status {status} should mark agent as unavailable"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
