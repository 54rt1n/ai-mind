# packages/aim-mud/tests/mud_tests/unit/worker/test_planner_mixin.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for PlannerMixin."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone


class MockWorker:
    """Mock worker for testing PlannerMixin."""

    def __init__(self):
        self.redis = AsyncMock()
        self.config = MagicMock()
        self.config.agent_id = "andi"
        self.chat_config = MagicMock()
        self.chat_config.tools_path = "config/tools"
        self.config.decision_tool_file = "config/tools/mud_phase1.yaml"
        self._decision_strategy = MagicMock()
        self._active_plan = None


# Import and apply mixin to mock worker
from andimud_worker.mixins.planner import PlannerMixin


class WorkerWithPlannerMixin(MockWorker, PlannerMixin):
    """Test class combining mock worker with mixin."""
    pass


@pytest.fixture
def worker():
    """Create a test worker with mixin."""
    return WorkerWithPlannerMixin()


@pytest.fixture
def sample_plan():
    """Create a sample plan for testing."""
    from aim_mud_types.plan import AgentPlan, PlanTask, PlanStatus, TaskStatus

    return AgentPlan(
        plan_id="test-plan-123",
        agent_id="andi",
        objective="Test objective",
        summary="Test plan summary",
        status=PlanStatus.ACTIVE,
        tasks=[
            PlanTask(
                id=0,
                description="First task",
                summary="Task 1",
                context="Why task 1",
                status=TaskStatus.IN_PROGRESS,
            ),
        ],
        current_task_id=0,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


class TestCheckActivePlan:
    """Tests for check_active_plan method."""

    @pytest.mark.asyncio
    async def test_planner_disabled(self, worker):
        """Test returns None when planner is disabled."""
        with patch("aim_mud_types.client.RedisMUDClient") as MockClient:
            mock_client = MagicMock()
            mock_client.is_planner_enabled = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await worker.check_active_plan()

        assert result is None

    @pytest.mark.asyncio
    async def test_no_plan(self, worker):
        """Test returns None when no plan exists."""
        with patch("aim_mud_types.client.RedisMUDClient") as MockClient:
            mock_client = MagicMock()
            mock_client.is_planner_enabled = AsyncMock(return_value=True)
            mock_client.get_plan = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await worker.check_active_plan()

        assert result is None

    @pytest.mark.asyncio
    async def test_plan_not_active(self, worker, sample_plan):
        """Test returns None when plan is not ACTIVE."""
        from aim_mud_types.plan import PlanStatus

        sample_plan.status = PlanStatus.BLOCKED

        with patch("aim_mud_types.client.RedisMUDClient") as MockClient:
            mock_client = MagicMock()
            mock_client.is_planner_enabled = AsyncMock(return_value=True)
            mock_client.get_plan = AsyncMock(return_value=sample_plan)
            MockClient.return_value = mock_client

            result = await worker.check_active_plan()

        assert result is None

    @pytest.mark.asyncio
    async def test_plan_completed(self, worker, sample_plan):
        """Test returns None when plan is COMPLETED."""
        from aim_mud_types.plan import PlanStatus

        sample_plan.status = PlanStatus.COMPLETED

        with patch("aim_mud_types.client.RedisMUDClient") as MockClient:
            mock_client = MagicMock()
            mock_client.is_planner_enabled = AsyncMock(return_value=True)
            mock_client.get_plan = AsyncMock(return_value=sample_plan)
            MockClient.return_value = mock_client

            result = await worker.check_active_plan()

        assert result is None

    @pytest.mark.asyncio
    async def test_plan_paused(self, worker, sample_plan):
        """Test returns None when plan is PAUSED."""
        from aim_mud_types.plan import PlanStatus

        sample_plan.status = PlanStatus.PAUSED

        with patch("aim_mud_types.client.RedisMUDClient") as MockClient:
            mock_client = MagicMock()
            mock_client.is_planner_enabled = AsyncMock(return_value=True)
            mock_client.get_plan = AsyncMock(return_value=sample_plan)
            MockClient.return_value = mock_client

            result = await worker.check_active_plan()

        assert result is None

    @pytest.mark.asyncio
    async def test_active_plan_returned(self, worker, sample_plan):
        """Test returns plan when planner enabled and plan is ACTIVE."""
        with patch("aim_mud_types.client.RedisMUDClient") as MockClient:
            mock_client = MagicMock()
            mock_client.is_planner_enabled = AsyncMock(return_value=True)
            mock_client.get_plan = AsyncMock(return_value=sample_plan)
            MockClient.return_value = mock_client

            result = await worker.check_active_plan()

        assert result is sample_plan


class TestSetActivePlan:
    """Tests for set_active_plan method."""

    def test_sets_plan_on_worker(self, worker, sample_plan):
        """Test plan is set on worker."""
        worker.set_active_plan(sample_plan)

        assert worker._active_plan is sample_plan

    def test_sets_plan_on_strategy(self, worker, sample_plan):
        """Test plan is set on decision strategy."""
        worker.set_active_plan(sample_plan)

        assert worker._decision_strategy._active_plan is sample_plan

    def test_handles_no_strategy(self, worker, sample_plan):
        """Test handles case when no decision strategy."""
        worker._decision_strategy = None

        worker.set_active_plan(sample_plan)  # Should not raise

        assert worker._active_plan is sample_plan


class TestClearActivePlan:
    """Tests for clear_active_plan method."""

    def test_clears_plan_on_worker(self, worker, sample_plan):
        """Test plan is cleared on worker."""
        worker._active_plan = sample_plan

        worker.clear_active_plan()

        assert worker._active_plan is None

    def test_clears_plan_on_strategy(self, worker, sample_plan):
        """Test plan is cleared on decision strategy."""
        worker._active_plan = sample_plan
        worker._decision_strategy._active_plan = sample_plan

        worker.clear_active_plan()

        assert worker._decision_strategy._active_plan is None

    def test_handles_no_strategy(self, worker, sample_plan):
        """Test handles case when no decision strategy."""
        worker._active_plan = sample_plan
        worker._decision_strategy = None

        worker.clear_active_plan()  # Should not raise

        assert worker._active_plan is None


class TestGetActivePlan:
    """Tests for get_active_plan method."""

    def test_returns_active_plan(self, worker, sample_plan):
        """Test returns active plan when set."""
        worker._active_plan = sample_plan

        result = worker.get_active_plan()

        assert result is sample_plan

    def test_returns_none_when_no_plan(self, worker):
        """Test returns None when no plan set."""
        result = worker.get_active_plan()

        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
