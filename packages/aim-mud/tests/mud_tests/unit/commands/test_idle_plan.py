# packages/aim-mud/tests/mud_tests/unit/commands/test_idle_plan.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for IdleCommand plan integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from andimud_worker.commands.idle import IdleCommand
from aim_mud_types import TurnRequestStatus


@pytest.fixture
def mock_worker():
    """Create a mock worker with PlannerMixin methods."""
    worker = MagicMock()
    worker.get_active_plan = MagicMock(return_value=None)
    worker.check_auto_dream_triggers = AsyncMock(return_value=None)
    worker.set_active_plan = MagicMock()
    worker.chat_config = MagicMock()
    worker.chat_config.tools_path = "config/tools"
    worker._tool_helper = MagicMock()
    worker._decision_strategy = MagicMock()
    worker._decision_strategy.get_plan_guidance = MagicMock(return_value="Test guidance")
    return worker


@pytest.fixture
def command():
    """Create an IdleCommand instance."""
    return IdleCommand()


@pytest.fixture
def sample_plan():
    """Create a sample plan for testing."""
    from aim_mud_types.plan import AgentPlan, PlanTask, PlanStatus, TaskStatus

    return AgentPlan(
        plan_id="test-plan-123",
        agent_id="andi",
        objective="Test objective",
        summary="Test plan",
        status=PlanStatus.ACTIVE,
        tasks=[
            PlanTask(
                id=0,
                description="First task",
                summary="Task 1",
                context="Context 1",
                status=TaskStatus.IN_PROGRESS,
            ),
        ],
        current_task_id=0,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


class TestIdlePlanPriority:
    """Tests for plan priority in idle command."""

    @pytest.mark.asyncio
    async def test_no_plan_no_dream(self, command, mock_worker):
        """Test idle with no plan returns idle message."""
        result = await command.execute(mock_worker, turn_id="test-turn")

        assert result.complete is False
        assert result.status == TurnRequestStatus.DONE
        assert "Idle turn ready" in result.message
        mock_worker.get_active_plan.assert_called_once()

    @pytest.mark.asyncio
    async def test_plan_detected(self, command, mock_worker, sample_plan):
        """Test that active plan is detected and message includes task."""
        mock_worker.get_active_plan.return_value = sample_plan

        result = await command.execute(mock_worker, turn_id="test-turn")

        assert result.complete is False
        assert "Plan active" in result.message or "Task 1" in result.message

    @pytest.mark.asyncio
    async def test_plan_tools_added(self, command, mock_worker, sample_plan):
        """Test that plan context is set when plan is active."""
        mock_worker.get_active_plan.return_value = sample_plan

        result = await command.execute(mock_worker, turn_id="test-turn")

        # IdleCommand doesn't directly add tools, just detects plan
        assert result.complete is False
        assert "Plan active" in result.message

    @pytest.mark.asyncio
    async def test_sets_context_on_decision_strategy(self, command, mock_worker, sample_plan):
        """Test that plan guidance is available when plan is active."""
        mock_worker.get_active_plan.return_value = sample_plan
        mock_worker._decision_strategy.get_plan_guidance = MagicMock(return_value="Test guidance")

        result = await command.execute(mock_worker, turn_id="test-turn")

        # Guidance is retrieved from strategy
        assert result.plan_guidance == "Test guidance"

    @pytest.mark.asyncio
    async def test_handles_no_tool_helper(self, command, mock_worker, sample_plan):
        """Test graceful handling when no tool helper."""
        mock_worker.get_active_plan.return_value = sample_plan
        mock_worker._tool_helper = None

        # Should not raise
        result = await command.execute(mock_worker, turn_id="test-turn")
        assert result.complete is False

    @pytest.mark.asyncio
    async def test_plan_guidance_returned(self, command, mock_worker, sample_plan):
        """Test that plan guidance is returned in result."""
        mock_worker.get_active_plan.return_value = sample_plan
        mock_worker._decision_strategy.get_plan_guidance.return_value = "Execute task 1"

        result = await command.execute(mock_worker, turn_id="test-turn")

        assert result.plan_guidance == "Execute task 1"

    @pytest.mark.asyncio
    async def test_handles_no_decision_strategy(self, command, mock_worker, sample_plan):
        """Test graceful handling when no decision strategy."""
        mock_worker.get_active_plan.return_value = sample_plan
        mock_worker._decision_strategy = None

        # Should not raise
        result = await command.execute(mock_worker, turn_id="test-turn")
        assert result.complete is False
        assert result.plan_guidance == ""

    @pytest.mark.asyncio
    async def test_message_includes_current_task(self, command, mock_worker, sample_plan):
        """Test that message includes current task summary."""
        mock_worker.get_active_plan.return_value = sample_plan

        result = await command.execute(mock_worker, turn_id="test-turn")

        assert "Task 1" in result.message

    @pytest.mark.asyncio
    async def test_handles_completed_plan_all_tasks_done(
        self, command, mock_worker, sample_plan
    ):
        """Test message when all tasks complete (current_task_id out of range)."""
        sample_plan.current_task_id = 5  # Out of range
        mock_worker.get_active_plan.return_value = sample_plan

        result = await command.execute(mock_worker, turn_id="test-turn")

        # Should fall back to generic message
        assert result.message == "Plan active"


class TestIdleDreamFallback:
    """Tests for IdleCommand behavior - no dream logic in command itself."""

    @pytest.mark.asyncio
    async def test_no_plan_returns_idle_ready(self, command, mock_worker):
        """Test that IdleCommand returns idle ready when no plan."""
        result = await command.execute(mock_worker, turn_id="test-turn")

        assert result.complete is False
        assert result.flush_drain is True
        assert "Idle turn ready" in result.message

    @pytest.mark.asyncio
    async def test_flush_drain_always_true(self, command, mock_worker):
        """Test that flush_drain is always True for idle."""
        result = await command.execute(mock_worker, turn_id="test-turn")

        assert result.flush_drain is True


class TestIdleCommandResult:
    """Tests for CommandResult structure."""

    @pytest.mark.asyncio
    async def test_result_flush_drain_true(self, command, mock_worker):
        """Test that flush_drain is always True for idle."""
        result = await command.execute(mock_worker, turn_id="test-turn")

        assert result.flush_drain is True

    @pytest.mark.asyncio
    async def test_result_complete_false(self, command, mock_worker, sample_plan):
        """Test that complete is always False (falls through to process_turn)."""
        # Without plan
        result = await command.execute(mock_worker, turn_id="test-turn")
        assert result.complete is False

        # With plan
        mock_worker.get_active_plan.return_value = sample_plan
        result = await command.execute(mock_worker, turn_id="test-turn")
        assert result.complete is False

    @pytest.mark.asyncio
    async def test_result_status_done(self, command, mock_worker):
        """Test that status is DONE."""
        result = await command.execute(mock_worker, turn_id="test-turn")

        assert result.status == TurnRequestStatus.DONE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
