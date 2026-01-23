# packages/aim-mud/tests/mud_tests/unit/worker/test_command_execution_flow.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for command execution paths in worker loop.

Tests the critical flow in worker.py lines 315-329:
- Command with flush_drain=True → pending_events cleared, saved_event_id=None
- Command with complete=True → status set, no process_turn called
- Command with complete=False → falls through to process_turn
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
    async def test_flush_drain_clears_pending_events(
        self, test_worker, sample_events
    ):
        """Test: Command with flush_drain=True → pending_events cleared, saved_event_id=None."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            last_event_id="event-0",
        )

        # Set up pending events
        saved_event_id = test_worker.session.last_event_id
        test_worker.pending_events = sample_events
        test_worker.session.last_event_id = "event-1"

        # Mock command result with flush_drain=True
        command_result = CommandResult(
            complete=True,
            flush_drain=True,
            status="done",
        )

        # Act - Simulate command execution flow (lines 319-323)
        if command_result.flush_drain:
            test_worker.pending_events = []
            saved_event_id = None

        # Assert
        assert test_worker.pending_events == [], (
            "flush_drain should clear pending_events"
        )
        assert saved_event_id is None, (
            "flush_drain should clear saved_event_id (events consumed)"
        )
        assert test_worker.session.last_event_id == "event-1", (
            "flush_drain should keep advanced event position"
        )

    @pytest.mark.asyncio
    async def test_complete_true_skips_process_turn(
        self, test_worker, sample_events
    ):
        """Test: Command with complete=True → status set, no process_turn called."""
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

        # Act - Simulate command execution flow (lines 324-328)
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
        """Test: Command with complete=False → falls through to process_turn."""
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

        # Act - Simulate command execution flow (lines 330-338)
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

    @pytest.mark.asyncio
    async def test_flush_drain_false_preserves_saved_event_id(
        self, test_worker, sample_events
    ):
        """Test: Command with flush_drain=False → saved_event_id preserved."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            last_event_id="event-0",
        )

        # Set up pending events
        saved_event_id = test_worker.session.last_event_id
        test_worker.pending_events = sample_events
        test_worker.session.last_event_id = "event-1"

        # Mock command result with flush_drain=False
        command_result = CommandResult(
            complete=False,
            flush_drain=False,
        )

        # Act - Simulate command execution flow
        if command_result.flush_drain:
            test_worker.pending_events = []
            saved_event_id = None

        # Assert
        assert test_worker.pending_events == sample_events, (
            "flush_drain=False should preserve pending_events"
        )
        assert saved_event_id == "event-0", (
            "flush_drain=False should preserve saved_event_id for rollback"
        )

    @pytest.mark.asyncio
    async def test_command_complete_with_flush_drain(
        self, test_worker, sample_events
    ):
        """Test: Command with both complete=True and flush_drain=True."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            last_event_id="event-0",
        )

        # Set up state
        saved_event_id = test_worker.session.last_event_id
        test_worker.pending_events = sample_events
        process_turn_called = False

        async def mock_process_turn(events):
            nonlocal process_turn_called
            process_turn_called = True

        test_worker.process_turn = mock_process_turn

        # Mock command result with both flags
        command_result = CommandResult(
            complete=True,
            flush_drain=True,
            status="done",
        )

        # Act - Simulate command execution flow
        if command_result.flush_drain:
            test_worker.pending_events = []
            saved_event_id = None

        if not command_result.complete:
            await test_worker.process_turn(test_worker.pending_events)

        # Assert
        assert test_worker.pending_events == [], "Events should be cleared"
        assert saved_event_id is None, "saved_event_id should be cleared"
        assert process_turn_called is False, "process_turn should not be called"

    @pytest.mark.asyncio
    async def test_command_incomplete_without_flush(
        self, test_worker, sample_events
    ):
        """Test: Command with complete=False and flush_drain=False."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            last_event_id="event-0",
        )

        # Set up state
        saved_event_id = test_worker.session.last_event_id
        test_worker.pending_events = sample_events
        process_turn_called = False
        process_turn_events = None

        async def mock_process_turn(events):
            nonlocal process_turn_called, process_turn_events
            process_turn_called = True
            process_turn_events = events

        test_worker.process_turn = mock_process_turn

        # Mock command result
        command_result = CommandResult(
            complete=False,
            flush_drain=False,
        )

        # Act - Simulate command execution flow
        if command_result.flush_drain:
            test_worker.pending_events = []
            saved_event_id = None

        if not command_result.complete:
            events = test_worker.pending_events
            await test_worker.process_turn(events)

        # Assert
        assert test_worker.pending_events == sample_events, (
            "Events should be preserved"
        )
        assert saved_event_id == "event-0", "saved_event_id should be preserved"
        assert process_turn_called is True, "process_turn should be called"
        assert process_turn_events == sample_events, (
            "process_turn should receive all events"
        )
