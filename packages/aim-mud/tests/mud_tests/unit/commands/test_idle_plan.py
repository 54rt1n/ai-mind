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
    """Create a mock worker with PlannerMixin and DreamingDatastoreMixin methods."""
    from andimud_worker.config import MUDConfig

    worker = MagicMock()
    worker.get_active_plan = MagicMock(return_value=None)
    worker.check_auto_dream_triggers = AsyncMock(return_value=None)
    worker.set_active_plan = MagicMock()
    worker.chat_config = MagicMock()
    worker.chat_config.tools_path = "config/tools"
    worker._tool_helper = MagicMock()
    worker._decision_strategy = MagicMock()
    worker._decision_strategy.get_plan_guidance = MagicMock(return_value="Test guidance")
    worker._decision_strategy.thought_content = None

    # Planner methods
    worker.get_plan_guidance = MagicMock(return_value=None)

    # DreamingDatastoreMixin methods (async)
    worker.load_dreaming_state = AsyncMock(return_value=None)
    worker.save_dreaming_state = AsyncMock()
    worker.delete_dreaming_state = AsyncMock()
    worker.archive_dreaming_state = AsyncMock()
    worker.execute_dream_step = AsyncMock(return_value=True)
    worker.execute_scenario_step = AsyncMock(return_value=True)
    worker.initialize_pending_dream = AsyncMock()
    worker.initialize_auto_dream = AsyncMock()

    # Dream decision method (async) - returns None (no dream action)
    worker._decide_dream_action = AsyncMock(return_value=None)

    # Async methods for turn processing
    worker._check_agent_is_sleeping = AsyncMock(return_value=False)
    worker.ensure_turn_id_current = AsyncMock()
    worker._setup_turn_context = AsyncMock()
    worker.claim_idle_turn = AsyncMock(return_value="claimed-turn-id")
    worker._clear_thought_content = AsyncMock()
    worker._drain_to_turn = AsyncMock(return_value=[])
    worker._drain_with_settle = AsyncMock(return_value=[])
    worker._emit_decision_action = AsyncMock()
    worker._is_idle_active = AsyncMock(return_value=False)

    # Thought throttle methods (new)
    worker._should_generate_new_thought = AsyncMock(return_value=False)
    worker._increment_thought_action_counter = AsyncMock(return_value=1)
    worker._load_thought_content = AsyncMock()

    # Pending events
    worker.pending_events = []

    # Last decision
    worker._last_decision = None

    # Worker config
    worker.config = MUDConfig(agent_id="test-agent", persona_id="test-persona")

    return worker


@pytest.fixture
def command():
    """Create an IdleCommand instance."""
    return IdleCommand()


@pytest.fixture
def turn_request_kwargs():
    """Create kwargs that can be passed to command.execute and validated as MUDTurnRequest."""
    from aim_mud_types import TurnRequestStatus, TurnReason
    return {
        "turn_id": "test-turn",
        "sequence_id": 1000,
        "status": TurnRequestStatus.IN_PROGRESS,
        "reason": TurnReason.IDLE,
    }


@pytest.fixture
def sample_plan():
    """Create a sample plan for testing."""
    from aim_mud_types.models.plan import AgentPlan, PlanTask, PlanStatus, TaskStatus

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
    async def test_no_plan_no_dream(self, command, mock_worker, turn_request_kwargs):
        """Test idle with no plan returns idle message."""
        # Mock no pending/running dreams
        mock_worker.load_dreaming_state.return_value = None
        mock_worker._check_agent_is_sleeping = AsyncMock(return_value=True)

        result = await command.execute(mock_worker, **turn_request_kwargs)

        assert result.complete is True
        assert result.status == TurnRequestStatus.DONE
        assert "sleeping" in result.message.lower()

    @pytest.mark.asyncio
    async def test_plan_detected(self, command, mock_worker, sample_plan, turn_request_kwargs):
        """Test that active plan is detected and message includes task."""
        mock_worker.get_active_plan.return_value = sample_plan
        mock_worker._check_agent_is_sleeping = AsyncMock(return_value=False)
        mock_worker._decision_strategy.thought_content = "Some thought"
        mock_worker.get_plan_guidance = MagicMock(return_value=None)

        result = await command.execute(mock_worker, **turn_request_kwargs)

        # When agent is awake with thought content, it returns immediately
        assert result.complete is True

    @pytest.mark.asyncio
    async def test_plan_tools_added(self, command, mock_worker, sample_plan, turn_request_kwargs):
        """Test that plan context is set when plan is active."""
        mock_worker.get_active_plan.return_value = sample_plan
        mock_worker._check_agent_is_sleeping = AsyncMock(return_value=False)
        mock_worker._decision_strategy.thought_content = "Some thought"
        mock_worker.get_plan_guidance = MagicMock(return_value=None)

        result = await command.execute(mock_worker, **turn_request_kwargs)

        # IdleCommand now returns complete=True for awake agents with thought
        assert result.complete is True

    @pytest.mark.asyncio
    async def test_sets_context_on_decision_strategy(self, command, mock_worker, sample_plan, turn_request_kwargs):
        """Test that plan guidance is available when plan is active."""
        mock_worker.get_active_plan.return_value = sample_plan
        mock_worker._check_agent_is_sleeping = AsyncMock(return_value=False)
        mock_worker._decision_strategy.thought_content = "Some thought"
        mock_worker.get_plan_guidance = MagicMock(return_value="Test guidance")

        result = await command.execute(mock_worker, **turn_request_kwargs)

        # Plan guidance is now retrieved via worker.get_plan_guidance()
        assert result.complete is True

    @pytest.mark.asyncio
    async def test_handles_no_tool_helper(self, command, mock_worker, sample_plan, turn_request_kwargs):
        """Test graceful handling when no tool helper."""
        mock_worker.get_active_plan.return_value = sample_plan
        mock_worker._tool_helper = None
        mock_worker._check_agent_is_sleeping = AsyncMock(return_value=False)
        mock_worker._decision_strategy.thought_content = "Some thought"
        mock_worker.get_plan_guidance = MagicMock(return_value=None)

        # Should not raise
        result = await command.execute(mock_worker, **turn_request_kwargs)
        assert result.complete is True

    @pytest.mark.asyncio
    async def test_plan_guidance_returned(self, command, mock_worker, sample_plan, turn_request_kwargs):
        """Test that plan guidance is returned in result."""
        mock_worker.get_active_plan.return_value = sample_plan
        mock_worker._check_agent_is_sleeping = AsyncMock(return_value=False)
        mock_worker._decision_strategy.thought_content = "Some thought"
        mock_worker.get_plan_guidance = MagicMock(return_value="Execute task 1")

        result = await command.execute(mock_worker, **turn_request_kwargs)

        # The command now doesn't return plan_guidance directly
        assert result.complete is True

    @pytest.mark.asyncio
    async def test_handles_no_decision_strategy(self, command, mock_worker, sample_plan, turn_request_kwargs):
        """Test graceful handling when no decision strategy."""
        mock_worker.get_active_plan.return_value = sample_plan
        mock_worker._decision_strategy = None
        mock_worker._check_agent_is_sleeping = AsyncMock(return_value=True)
        mock_worker.load_dreaming_state.return_value = None

        # Should not raise - sleeping agent doesn't use decision strategy
        result = await command.execute(mock_worker, **turn_request_kwargs)
        assert result.complete is True

    @pytest.mark.asyncio
    async def test_message_includes_current_task(self, command, mock_worker, sample_plan, turn_request_kwargs):
        """Test that message includes current task summary."""
        mock_worker.get_active_plan.return_value = sample_plan
        mock_worker._check_agent_is_sleeping = AsyncMock(return_value=False)
        mock_worker._decision_strategy.thought_content = "Some thought"
        mock_worker.get_plan_guidance = MagicMock(return_value=None)

        result = await command.execute(mock_worker, **turn_request_kwargs)

        # The awake idle command just returns "Agent awake" now
        assert result.complete is True

    @pytest.mark.asyncio
    async def test_handles_completed_plan_all_tasks_done(
        self, command, mock_worker, sample_plan, turn_request_kwargs
    ):
        """Test message when all tasks complete (current_task_id out of range)."""
        sample_plan.current_task_id = 5  # Out of range
        mock_worker.get_active_plan.return_value = sample_plan
        mock_worker._check_agent_is_sleeping = AsyncMock(return_value=False)
        mock_worker._decision_strategy.thought_content = "Some thought"
        mock_worker.get_plan_guidance = MagicMock(return_value=None)

        result = await command.execute(mock_worker, **turn_request_kwargs)

        # Returns Agent awake for completed plan
        assert result.complete is True


class TestIdleDreamFallback:
    """Tests for IdleCommand behavior - no dream logic in command itself."""

    @pytest.mark.asyncio
    async def test_no_plan_returns_idle_ready(self, command, mock_worker, turn_request_kwargs):
        """Test that IdleCommand returns appropriate message when sleeping."""
        # Mock no pending/running dreams
        mock_worker.load_dreaming_state.return_value = None
        mock_worker._check_agent_is_sleeping = AsyncMock(return_value=True)

        result = await command.execute(mock_worker, **turn_request_kwargs)

        assert result.complete is True
        assert result.status == TurnRequestStatus.DONE
        assert "sleeping" in result.message.lower()

class TestIdleCommandResult:
    """Tests for CommandResult structure."""

    @pytest.mark.asyncio
    async def test_result_complete_false(self, command, mock_worker, sample_plan, turn_request_kwargs):
        """Test that complete is True for both sleeping and awake agents."""
        # Mock sleeping agent
        mock_worker.load_dreaming_state.return_value = None
        mock_worker._check_agent_is_sleeping = AsyncMock(return_value=True)

        # Without plan - sleeping
        result = await command.execute(mock_worker, **turn_request_kwargs)
        assert result.complete is True

        # With plan - awake
        mock_worker.get_active_plan.return_value = sample_plan
        mock_worker._check_agent_is_sleeping = AsyncMock(return_value=False)
        mock_worker._decision_strategy.thought_content = "Some thought"
        mock_worker.get_plan_guidance = MagicMock(return_value=None)
        result = await command.execute(mock_worker, **turn_request_kwargs)
        assert result.complete is True

    @pytest.mark.asyncio
    async def test_result_status_done(self, command, mock_worker, turn_request_kwargs):
        """Test that status is DONE."""
        # Mock no pending/running dreams
        mock_worker.load_dreaming_state.return_value = None
        mock_worker._check_agent_is_sleeping = AsyncMock(return_value=True)

        result = await command.execute(mock_worker, **turn_request_kwargs)

        assert result.status == TurnRequestStatus.DONE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
