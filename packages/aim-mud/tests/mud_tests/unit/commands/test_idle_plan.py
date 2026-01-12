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
    worker.check_active_plan = AsyncMock(return_value=None)
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
        """Test idle with no plan and no dream triggers."""
        result = await command.execute(mock_worker, turn_id="test-turn")

        assert result.complete is False
        assert result.status == TurnRequestStatus.DONE
        assert "Idle turn ready" in result.message
        mock_worker.check_active_plan.assert_called_once()
        mock_worker.check_auto_dream_triggers.assert_called_once()

    @pytest.mark.asyncio
    async def test_plan_takes_priority(self, command, mock_worker, sample_plan):
        """Test that active plan takes priority over dream."""
        mock_worker.check_active_plan.return_value = sample_plan

        result = await command.execute(mock_worker, turn_id="test-turn")

        assert result.complete is False
        assert "Plan active" in result.message or "Task 1" in result.message
        mock_worker.set_active_plan.assert_called_once_with(sample_plan)
        # Dream triggers should NOT be checked when plan is active
        mock_worker.check_auto_dream_triggers.assert_not_called()

    @pytest.mark.asyncio
    async def test_plan_tools_added(self, command, mock_worker, sample_plan):
        """Test that plan tools are added when plan is active."""
        mock_worker.check_active_plan.return_value = sample_plan

        await command.execute(mock_worker, turn_id="test-turn")

        mock_worker._tool_helper.add_plan_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_sets_context_on_decision_strategy(self, command, mock_worker, sample_plan):
        """Test that set_context is called on decision strategy when plan is active."""
        mock_worker.check_active_plan.return_value = sample_plan
        mock_worker._decision_strategy.set_context = MagicMock()
        mock_worker.redis = MagicMock()
        mock_worker.config.agent_id = "test-agent-id"

        await command.execute(mock_worker, turn_id="test-turn")

        mock_worker._decision_strategy.set_context.assert_called_once_with(
            mock_worker.redis, "test-agent-id"
        )

    @pytest.mark.asyncio
    async def test_handles_no_tool_helper(self, command, mock_worker, sample_plan):
        """Test graceful handling when no tool helper."""
        mock_worker.check_active_plan.return_value = sample_plan
        mock_worker._tool_helper = None

        # Should not raise
        result = await command.execute(mock_worker, turn_id="test-turn")
        assert result.complete is False

    @pytest.mark.asyncio
    async def test_plan_guidance_returned(self, command, mock_worker, sample_plan):
        """Test that plan guidance is returned in result."""
        mock_worker.check_active_plan.return_value = sample_plan
        mock_worker._decision_strategy.get_plan_guidance.return_value = "Execute task 1"

        result = await command.execute(mock_worker, turn_id="test-turn")

        assert result.plan_guidance == "Execute task 1"

    @pytest.mark.asyncio
    async def test_handles_no_decision_strategy(self, command, mock_worker, sample_plan):
        """Test graceful handling when no decision strategy."""
        mock_worker.check_active_plan.return_value = sample_plan
        mock_worker._decision_strategy = None

        # Should not raise
        result = await command.execute(mock_worker, turn_id="test-turn")
        assert result.complete is False
        assert result.plan_guidance == ""

    @pytest.mark.asyncio
    async def test_message_includes_current_task(self, command, mock_worker, sample_plan):
        """Test that message includes current task summary."""
        mock_worker.check_active_plan.return_value = sample_plan

        result = await command.execute(mock_worker, turn_id="test-turn")

        assert "Task 1" in result.message

    @pytest.mark.asyncio
    async def test_handles_completed_plan_all_tasks_done(
        self, command, mock_worker, sample_plan
    ):
        """Test message when all tasks complete (current_task_id out of range)."""
        sample_plan.current_task_id = 5  # Out of range
        mock_worker.check_active_plan.return_value = sample_plan

        result = await command.execute(mock_worker, turn_id="test-turn")

        # Should fall back to generic message
        assert result.message == "Plan active"


class TestIdleDreamFallback:
    """Tests for dream fallback when no plan."""

    @pytest.mark.asyncio
    async def test_dream_triggered_when_no_plan(self, command, mock_worker):
        """Test that dream triggers are checked when no plan."""
        mock_dream_request = MagicMock()
        mock_dream_request.scenario = "analysis_dialogue"
        mock_dream_request.query = None
        mock_dream_request.guidance = None
        mock_worker.check_auto_dream_triggers.return_value = mock_dream_request

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.pipeline_id = "dream-123"
        mock_result.duration_seconds = 5.0
        mock_worker.process_dream_turn = AsyncMock(return_value=mock_result)
        mock_worker._update_conversation_report = AsyncMock()

        result = await command.execute(mock_worker, turn_id="test-turn")

        mock_worker.process_dream_turn.assert_called_once()
        assert result.complete is False

    @pytest.mark.asyncio
    async def test_dream_failure_logged(self, command, mock_worker):
        """Test that dream failure is handled gracefully."""
        mock_dream_request = MagicMock()
        mock_dream_request.scenario = "analysis_dialogue"
        mock_dream_request.query = None
        mock_dream_request.guidance = None
        mock_worker.check_auto_dream_triggers.return_value = mock_dream_request

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Test error"
        mock_worker.process_dream_turn = AsyncMock(return_value=mock_result)

        # Should not raise
        result = await command.execute(mock_worker, turn_id="test-turn")

        assert result.complete is False
        # _update_conversation_report should NOT be called on failure
        mock_worker._update_conversation_report.assert_not_called()


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
        mock_worker.check_active_plan.return_value = sample_plan
        result = await command.execute(mock_worker, turn_id="test-turn")
        assert result.complete is False

    @pytest.mark.asyncio
    async def test_result_status_done(self, command, mock_worker):
        """Test that status is DONE."""
        result = await command.execute(mock_worker, turn_id="test-turn")

        assert result.status == TurnRequestStatus.DONE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
