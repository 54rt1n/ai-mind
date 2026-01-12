# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for PlanExecutionTool."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from aim.tool.impl.plan import PlanExecutionTool


@pytest.fixture
def tool():
    """Create a PlanExecutionTool for testing."""
    return PlanExecutionTool()


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    return AsyncMock()


@pytest.fixture
def sample_plan():
    """Create a sample AgentPlan for testing."""
    from aim_mud_types import AgentPlan, PlanTask, PlanStatus, TaskStatus

    return AgentPlan(
        plan_id="test-plan-123",
        agent_id="andi",
        objective="Test objective",
        summary="Test plan summary",
        status=PlanStatus.ACTIVE,
        tasks=[
            PlanTask(
                id=0,
                description="First task description",
                summary="Task 1",
                context="Why task 1",
                status=TaskStatus.IN_PROGRESS,
            ),
            PlanTask(
                id=1,
                description="Second task description",
                summary="Task 2",
                context="Why task 2",
                status=TaskStatus.PENDING,
            ),
            PlanTask(
                id=2,
                description="Third task description",
                summary="Task 3",
                context="Why task 3",
                status=TaskStatus.PENDING,
            ),
        ],
        current_task_id=0,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


class TestPlanExecutionTool:
    """Tests for PlanExecutionTool initialization and context."""

    def test_execute_raises_error(self, tool):
        """Test that sync execute raises RuntimeError."""
        with pytest.raises(RuntimeError, match="requires async execution"):
            tool.execute("plan_update", {"status": "completed", "resolution": "Done"})

    @pytest.mark.asyncio
    async def test_no_context_error(self, tool):
        """Test error when context not set."""
        result = await tool.execute_async(
            "plan_update",
            {"status": "completed", "resolution": "Done"},
        )
        assert "error" in result
        assert "Context not set" in result["error"]

    def test_set_context(self, tool, mock_redis):
        """Test setting context."""
        tool.set_context(mock_redis, "andi")
        assert tool._redis_client is mock_redis
        assert tool._agent_id == "andi"


class TestPlanUpdate:
    """Tests for _plan_update method."""

    @pytest.mark.asyncio
    async def test_no_active_plan(self, tool, mock_redis):
        """Test error when no active plan."""
        tool.set_context(mock_redis, "andi")

        with patch("aim_mud_types.RedisMUDClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get_plan = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await tool.execute_async(
                "plan_update",
                {"status": "completed", "resolution": "Done"},
            )

        assert "error" in result
        assert "No active plan" in result["error"]

    @pytest.mark.asyncio
    async def test_complete_task_advances_to_next(self, tool, mock_redis, sample_plan):
        """Test completing a task advances to the next."""
        tool.set_context(mock_redis, "andi")

        with patch("aim_mud_types.RedisMUDClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get_plan = AsyncMock(return_value=sample_plan)
            mock_client.update_plan_fields = AsyncMock(return_value=True)
            MockClient.return_value = mock_client

            result = await tool.execute_async(
                "plan_update",
                {"status": "completed", "resolution": "Finished task 1"},
            )

        assert result["status"] == "ok"
        assert result["plan_status"] == "active"
        assert result["new_task_id"] == 1
        assert result["next_task"] == "Task 2"

    @pytest.mark.asyncio
    async def test_complete_last_task_completes_plan(
        self, tool, mock_redis, sample_plan
    ):
        """Test completing last task completes the plan."""
        tool.set_context(mock_redis, "andi")

        # Set up plan with only last task remaining
        from aim_mud_types import TaskStatus

        sample_plan.current_task_id = 2
        sample_plan.tasks[0].status = TaskStatus.COMPLETED
        sample_plan.tasks[1].status = TaskStatus.COMPLETED
        sample_plan.tasks[2].status = TaskStatus.IN_PROGRESS

        with patch("aim_mud_types.RedisMUDClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get_plan = AsyncMock(return_value=sample_plan)
            mock_client.update_plan_fields = AsyncMock(return_value=True)
            MockClient.return_value = mock_client

            result = await tool.execute_async(
                "plan_update",
                {"status": "completed", "resolution": "All done"},
            )

        assert result["status"] == "ok"
        assert result["plan_status"] == "completed"
        assert result["next_task"] is None

    @pytest.mark.asyncio
    async def test_blocked_task_blocks_plan(self, tool, mock_redis, sample_plan):
        """Test blocking a task blocks the plan."""
        tool.set_context(mock_redis, "andi")

        with patch("aim_mud_types.RedisMUDClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get_plan = AsyncMock(return_value=sample_plan)
            mock_client.update_plan_fields = AsyncMock(return_value=True)
            MockClient.return_value = mock_client

            result = await tool.execute_async(
                "plan_update",
                {"status": "blocked", "resolution": "Cannot proceed - missing dependency"},
            )

        assert result["status"] == "ok"
        assert result["plan_status"] == "blocked"

    @pytest.mark.asyncio
    async def test_skip_task_advances_to_next(self, tool, mock_redis, sample_plan):
        """Test skipping a task advances to the next."""
        tool.set_context(mock_redis, "andi")

        with patch("aim_mud_types.RedisMUDClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get_plan = AsyncMock(return_value=sample_plan)
            mock_client.update_plan_fields = AsyncMock(return_value=True)
            MockClient.return_value = mock_client

            result = await tool.execute_async(
                "plan_update",
                {"status": "skipped", "resolution": "Not needed"},
            )

        assert result["status"] == "ok"
        assert result["plan_status"] == "active"
        assert result["new_task_id"] == 1
        assert result["next_task"] == "Task 2"

    @pytest.mark.asyncio
    async def test_skip_last_task_completes_plan(self, tool, mock_redis, sample_plan):
        """Test skipping last task completes the plan."""
        tool.set_context(mock_redis, "andi")

        from aim_mud_types import TaskStatus

        # Set up plan with only last task remaining
        sample_plan.current_task_id = 2
        sample_plan.tasks[0].status = TaskStatus.COMPLETED
        sample_plan.tasks[1].status = TaskStatus.COMPLETED
        sample_plan.tasks[2].status = TaskStatus.IN_PROGRESS

        with patch("aim_mud_types.RedisMUDClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get_plan = AsyncMock(return_value=sample_plan)
            mock_client.update_plan_fields = AsyncMock(return_value=True)
            MockClient.return_value = mock_client

            result = await tool.execute_async(
                "plan_update",
                {"status": "skipped", "resolution": "Not applicable"},
            )

        assert result["status"] == "ok"
        assert result["plan_status"] == "completed"
        assert result["next_task"] is None

    @pytest.mark.asyncio
    async def test_unknown_function(self, tool, mock_redis):
        """Test handling of unknown function name."""
        tool.set_context(mock_redis, "andi")

        result = await tool.execute_async("unknown_function", {})

        assert "error" in result
        assert "Unknown function" in result["error"]

    @pytest.mark.asyncio
    async def test_plan_beyond_task_count(self, tool, mock_redis, sample_plan):
        """Test error when current_task_id is beyond task count."""
        tool.set_context(mock_redis, "andi")

        sample_plan.current_task_id = 10  # Beyond task count

        with patch("aim_mud_types.RedisMUDClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get_plan = AsyncMock(return_value=sample_plan)
            MockClient.return_value = mock_client

            result = await tool.execute_async(
                "plan_update",
                {"status": "completed", "resolution": "Done"},
            )

        assert "error" in result
        assert "No current task" in result["error"]


class TestTaskResolution:
    """Tests for task resolution recording."""

    @pytest.mark.asyncio
    async def test_resolution_recorded_on_complete(self, tool, mock_redis, sample_plan):
        """Test that resolution is recorded when task is completed."""
        tool.set_context(mock_redis, "andi")

        with patch("aim_mud_types.RedisMUDClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get_plan = AsyncMock(return_value=sample_plan)
            mock_client.update_plan_fields = AsyncMock(return_value=True)
            MockClient.return_value = mock_client

            await tool.execute_async(
                "plan_update",
                {"status": "completed", "resolution": "Task completed successfully"},
            )

        # Verify the task resolution was set
        assert sample_plan.tasks[0].resolution == "Task completed successfully"

    @pytest.mark.asyncio
    async def test_resolution_recorded_on_blocked(self, tool, mock_redis, sample_plan):
        """Test that resolution is recorded when task is blocked."""
        tool.set_context(mock_redis, "andi")

        with patch("aim_mud_types.RedisMUDClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get_plan = AsyncMock(return_value=sample_plan)
            mock_client.update_plan_fields = AsyncMock(return_value=True)
            MockClient.return_value = mock_client

            await tool.execute_async(
                "plan_update",
                {"status": "blocked", "resolution": "Missing API key"},
            )

        assert sample_plan.tasks[0].resolution == "Missing API key"


class TestFindNextPendingTask:
    """Tests for _find_next_pending_task helper."""

    def test_finds_next_pending(self, tool, sample_plan):
        """Test finding next pending task."""
        result = tool._find_next_pending_task(sample_plan)
        assert result == 1  # Task at index 1 is pending

    def test_skips_non_pending_tasks(self, tool, sample_plan):
        """Test that non-pending tasks are skipped."""
        from aim_mud_types import TaskStatus

        # Mark task 1 as completed
        sample_plan.tasks[1].status = TaskStatus.COMPLETED

        result = tool._find_next_pending_task(sample_plan)
        assert result == 2  # Should skip to task at index 2

    def test_returns_none_when_no_pending(self, tool, sample_plan):
        """Test returns None when no pending tasks remain."""
        from aim_mud_types import TaskStatus

        # Mark all tasks as completed
        for task in sample_plan.tasks:
            task.status = TaskStatus.COMPLETED

        result = tool._find_next_pending_task(sample_plan)
        assert result is None
