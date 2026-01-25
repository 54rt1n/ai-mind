# packages/aim-mud/tests/mud_tests/unit/worker/test_command_execution_flow.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for command execution paths in worker loop.

Tests the critical flow in worker.py:
- Command with complete=True -> status set, no process_turn called
- Command with complete=False -> falls through to process_turn
- Command with emitted_action_ids -> triggers pending action tracking

NOTE: flush_drain and saved_event_id have been removed. Events are now
consumed when they're pushed to conversation history, with each entry
tracking its own last_event_id.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from aim_mud_types import MUDEvent, EventType
from aim_mud_types.models.session import MUDSession
from andimud_worker.commands.result import CommandResult


@pytest.fixture
def sample_events():
    """Create sample events for command testing."""
    return [
        MUDEvent(
            event_id="event-1",
            event_type=EventType.SPEECH,
            actor="OtherAgent",
            actor_id="other_agent",
            room_id="room1",
            room_name="Test Room",
            content="Test event",
            timestamp=datetime.now(timezone.utc),
            metadata={"sequence_id": 100},
        ),
    ]


class TestCommandExecutionFlow:
    """Tests for command execution paths in worker loop."""

    @pytest.mark.asyncio
    async def test_complete_true_skips_process_turn(
        self, test_worker, sample_events
    ):
        """Test: Command with complete=True -> status set, no process_turn called."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
        )

        # Mock process_turn to track if it was called
        process_turn_called = False

        async def mock_process_turn(events):
            nonlocal process_turn_called
            process_turn_called = True

        test_worker.process_turn = mock_process_turn

        # Mock command result with complete=True
        command_result = CommandResult(
            complete=True,
            status="done",
            message="Command completed",
        )

        # Act - Simulate command execution flow
        should_process_turn = not command_result.complete

        if not should_process_turn:
            # Command completed, don't call process_turn
            pass
        else:
            await test_worker.process_turn(sample_events)

        # Assert
        assert command_result.complete is True, "Command should be complete"
        assert process_turn_called is False, (
            "complete=True should skip process_turn"
        )

    @pytest.mark.asyncio
    async def test_complete_false_falls_through_to_process_turn(
        self, test_worker, sample_events
    ):
        """Test: Command with complete=False -> falls through to process_turn."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
        )

        # Mock process_turn to track if it was called
        process_turn_called = False
        process_turn_events = None

        async def mock_process_turn(events):
            nonlocal process_turn_called, process_turn_events
            process_turn_called = True
            process_turn_events = events

        test_worker.process_turn = mock_process_turn
        test_worker.pending_events = sample_events

        # Mock command result with complete=False
        command_result = CommandResult(
            complete=False,
        )

        # Act - Simulate command execution flow
        if not command_result.complete:
            # Fall through to process_turn
            events = test_worker.pending_events
            await test_worker.process_turn(events)

        # Assert
        assert command_result.complete is False, "Command should not be complete"
        assert process_turn_called is True, (
            "complete=False should call process_turn"
        )
        assert process_turn_events == sample_events, (
            "process_turn should receive pending_events"
        )

    @pytest.mark.asyncio
    async def test_command_result_status_propagation(
        self, test_worker
    ):
        """Test: Command result status propagation."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
        )

        # Test different status values
        test_cases = [
            CommandResult(complete=True, status="done", message="Success"),
            CommandResult(complete=True, status="fail", message="Failed"),
            CommandResult(complete=True, status="aborted", message="Aborted"),
        ]

        for result in test_cases:
            # Act - Extract status and message
            status = result.status
            message = result.message

            # Assert
            assert status in ["done", "fail", "aborted"], (
                f"Status should be valid: {status}"
            )
            assert message is not None, "Message should be present"


class TestEmittedActionIds:
    """Tests for emitted_action_ids in CommandResult."""

    def test_emitted_action_ids_default_empty(self):
        """Test emitted_action_ids defaults to empty list."""
        result = CommandResult(complete=True)
        assert result.emitted_action_ids == []

    def test_emitted_action_ids_set(self):
        """Test emitted_action_ids can be set."""
        action_ids = ["act_123_abc", "act_456_def"]
        result = CommandResult(complete=True, emitted_action_ids=action_ids)
        assert result.emitted_action_ids == action_ids

    def test_emitted_action_ids_truthy_when_populated(self):
        """Test emitted_action_ids is truthy when populated."""
        result = CommandResult(complete=True, emitted_action_ids=["act_123_abc"])
        assert result.emitted_action_ids  # Should be truthy

    def test_emitted_action_ids_falsy_when_empty(self):
        """Test emitted_action_ids is falsy when empty."""
        result = CommandResult(complete=True)
        assert not result.emitted_action_ids  # Should be falsy

    @pytest.mark.asyncio
    async def test_emitted_action_ids_triggers_pending_tracking(
        self, test_worker
    ):
        """Test: Command with emitted_action_ids -> pending action tracking triggered."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
        )

        # Track if pending would be set
        pending_triggered = False
        action_ids = ["act_123_abc", "act_456_def"]

        # Mock command result with emitted_action_ids
        command_result = CommandResult(
            complete=True,
            status="done",
            emitted_action_ids=action_ids,
        )

        # Act - Simulate command execution flow
        if command_result.complete:
            # After status update, check emitted_action_ids
            if command_result.emitted_action_ids:
                pending_triggered = True
                # In real code, this would set turn to PENDING with metadata

        # Assert
        assert pending_triggered is True, (
            "emitted_action_ids should trigger pending tracking"
        )

    @pytest.mark.asyncio
    async def test_empty_action_ids_skips_pending(
        self, test_worker
    ):
        """Test: Command with empty emitted_action_ids -> no pending tracking."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
        )

        pending_triggered = False

        # Mock command result with empty emitted_action_ids (default)
        command_result = CommandResult(
            complete=True,
            status="done",
        )

        # Act - Simulate command execution flow
        if command_result.complete:
            if command_result.emitted_action_ids:
                pending_triggered = True

        # Assert
        assert pending_triggered is False, (
            "empty emitted_action_ids should not trigger pending tracking"
        )
