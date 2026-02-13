# tests/mud_tests/unit/worker/test_sleep_metadata.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for @sleep command metadata handling.

Verifies that agent=True metadata flows correctly from @sleep command
through to SleepCommand.execute() to trigger CVM flush.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aim_mud_types import MUDTurnRequest, TurnReason, TurnRequestStatus
from andimud_worker.commands.sleep import SleepCommand


class TestSleepMetadataFlow:
    """Test that agent=True metadata flows through to SleepCommand."""

    @pytest.fixture
    def mock_worker(self):
        """Create a mock worker with required attributes."""
        worker = MagicMock()
        worker.persona = MagicMock()
        worker.session = MagicMock()
        worker.model = MagicMock()
        worker.model.max_tokens = 4096
        worker.cvm = MagicMock()

        # Mock async methods
        worker._setup_turn_context = AsyncMock()
        worker._call_llm = AsyncMock(return_value="settles into a comfortable position and drifts off to sleep.")
        worker._emit_actions = AsyncMock(return_value=(["action_1"], False))
        worker._save_agent_profile = AsyncMock()
        worker._update_conversation_report = AsyncMock()

        # Mock response strategy
        worker._response_strategy = MagicMock()
        worker._response_strategy.build_turns = AsyncMock(return_value=[])

        # Mock conversation manager
        worker.conversation_manager = MagicMock()
        worker.conversation_manager.retag_unsaved_entries = AsyncMock(return_value=5)
        worker.conversation_manager.set_conversation_id = MagicMock()
        worker.conversation_manager.flush_to_cvm = AsyncMock(return_value=10)

        return worker

    @pytest.mark.asyncio
    async def test_sleep_with_agent_true_flushes_to_cvm(self, mock_worker):
        """When agent=True metadata is passed, SleepCommand should flush to CVM."""
        # Create turn_request with agent=True metadata
        turn_request = MUDTurnRequest(
            turn_id="sleep_test_123",
            status=TurnRequestStatus.ASSIGNED,
            reason=TurnReason.SLEEP,
            sequence_id=1,
            metadata={"agent": True},
        )

        # Execute command with turn_request passed directly (as worker now does)
        cmd = SleepCommand()
        result = await cmd.execute(
            mock_worker,
            turn_request=turn_request,
            events=[],
        )

        # Assert CVM flush was called (only happens when agent=True)
        mock_worker.conversation_manager.flush_to_cvm.assert_called_once_with(mock_worker.cvm)
        mock_worker.conversation_manager.retag_unsaved_entries.assert_called_once()
        mock_worker.conversation_manager.set_conversation_id.assert_called_once()
        mock_worker._save_agent_profile.assert_called_once()
        mock_worker._update_conversation_report.assert_called_once()

        # Result should indicate agent mode was active
        assert result.complete is True
        assert "agent: new conv + flush" in result.message

    @pytest.mark.asyncio
    async def test_sleep_without_agent_flag_does_not_flush(self, mock_worker):
        """When agent metadata is not set, SleepCommand should NOT flush to CVM."""
        # Create turn_request without agent metadata
        turn_request = MUDTurnRequest(
            turn_id="sleep_test_456",
            status=TurnRequestStatus.ASSIGNED,
            reason=TurnReason.SLEEP,
            sequence_id=1,
            metadata={},  # No agent flag
        )

        # Execute command
        cmd = SleepCommand()
        result = await cmd.execute(
            mock_worker,
            turn_request=turn_request,
            events=[],
        )

        # Assert CVM flush was NOT called
        mock_worker.conversation_manager.flush_to_cvm.assert_not_called()
        mock_worker.conversation_manager.retag_unsaved_entries.assert_not_called()

        # Result should NOT indicate agent mode
        assert result.complete is True
        assert "agent: new conv + flush" not in result.message

    @pytest.mark.asyncio
    async def test_sleep_with_agent_false_does_not_flush(self, mock_worker):
        """When agent=False, SleepCommand should NOT flush to CVM."""
        # Create turn_request with agent=False
        turn_request = MUDTurnRequest(
            turn_id="sleep_test_789",
            status=TurnRequestStatus.ASSIGNED,
            reason=TurnReason.SLEEP,
            sequence_id=1,
            metadata={"agent": False},
        )

        # Execute command
        cmd = SleepCommand()
        result = await cmd.execute(
            mock_worker,
            turn_request=turn_request,
            events=[],
        )

        # Assert CVM flush was NOT called
        mock_worker.conversation_manager.flush_to_cvm.assert_not_called()

        # Result should NOT indicate agent mode
        assert result.complete is True
        assert "agent: new conv + flush" not in result.message

    @pytest.mark.asyncio
    async def test_sleep_with_agent_string_true_flushes(self, mock_worker):
        """When agent='true' (string), SleepCommand should still flush."""
        # Some code paths may pass agent as string
        turn_request = MUDTurnRequest(
            turn_id="sleep_test_str",
            status=TurnRequestStatus.ASSIGNED,
            reason=TurnReason.SLEEP,
            sequence_id=1,
            metadata={"agent": "true"},  # String instead of bool
        )

        # Execute command
        cmd = SleepCommand()
        result = await cmd.execute(
            mock_worker,
            turn_request=turn_request,
            events=[],
        )

        # Assert CVM flush was called
        mock_worker.conversation_manager.flush_to_cvm.assert_called_once()
        assert "agent: new conv + flush" in result.message

    @pytest.mark.asyncio
    async def test_backward_compatibility_with_kwargs_only(self, mock_worker):
        """When turn_request is not passed directly, fallback should work."""
        # This tests backward compatibility with old dispatch pattern
        # (though the new code always passes turn_request directly)

        cmd = SleepCommand()

        # Simulate old pattern: pass kwargs that can be validated into MUDTurnRequest
        result = await cmd.execute(
            mock_worker,
            turn_id="sleep_compat_test",
            status=TurnRequestStatus.ASSIGNED.value,
            reason=TurnReason.SLEEP.value,
            sequence_id=1,
            metadata={"agent": True},  # This was the bug - metadata lost in reconstruction
            heartbeat_at=1234567890,
            assigned_at=1234567890,
            events=[],
        )

        # With the fallback, this should still work
        assert result.complete is True
