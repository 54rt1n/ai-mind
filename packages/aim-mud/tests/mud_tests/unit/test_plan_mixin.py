# packages/aim-mud/tests/mud_tests/unit/test_plan_mixin.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for plan mixin."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from aim_mud_types import (
    AgentPlan,
    PlanTask,
    PlanStatus,
    TaskStatus,
    RedisKeys,
)
from aim_mud_types.client import RedisMUDClient


@pytest.fixture
def client():
    """Create RedisMUDClient with mocked Redis."""
    mock_redis = AsyncMock()
    return RedisMUDClient(mock_redis)


@pytest.fixture
def sample_plan():
    """Create a sample plan for testing."""
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
        ],
        current_task_id=0,
        created_at=datetime(2026, 1, 12, 12, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 12, 12, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_blocked_plan():
    """Create a blocked plan for testing."""
    return AgentPlan(
        plan_id="blocked-plan",
        agent_id="andi",
        objective="Blocked objective",
        summary="A blocked plan",
        status=PlanStatus.BLOCKED,
        tasks=[
            PlanTask(
                id=0,
                description="Blocked task",
                summary="Task blocked",
                context="Context",
                status=TaskStatus.BLOCKED,
                resolution="Cannot proceed due to X",
            ),
        ],
        current_task_id=0,
    )


@pytest.fixture
def sample_completed_plan():
    """Create a completed plan for testing."""
    return AgentPlan(
        plan_id="completed-plan",
        agent_id="andi",
        objective="Completed objective",
        summary="A completed plan",
        status=PlanStatus.COMPLETED,
        tasks=[
            PlanTask(
                id=0,
                description="First task",
                summary="Task 1",
                context="Context 1",
                status=TaskStatus.COMPLETED,
                resolution="Done successfully",
            ),
            PlanTask(
                id=1,
                description="Second task",
                summary="Task 2",
                context="Context 2",
                status=TaskStatus.COMPLETED,
                resolution="Also done",
            ),
        ],
        current_task_id=1,
    )


class TestPlanMixinGetPlan:
    """Tests for get_plan method."""

    @pytest.mark.asyncio
    async def test_get_plan_success(self, client, sample_plan):
        """Should fetch and deserialize plan."""
        client.redis.hgetall.return_value = {
            b"plan_id": b"test-plan-123",
            b"agent_id": b"andi",
            b"objective": b"Test objective",
            b"summary": b"Test plan summary",
            b"status": b"active",
            b"tasks": b'[{"id": 0, "description": "First task description", "summary": "Task 1", "context": "Why task 1", "status": "in_progress"}, {"id": 1, "description": "Second task description", "summary": "Task 2", "context": "Why task 2", "status": "pending"}]',
            b"current_task_id": b"0",
            b"created_at": b"2026-01-12T12:00:00+00:00",
            b"updated_at": b"2026-01-12T12:00:00+00:00",
        }

        result = await client.get_plan("andi")

        assert result is not None
        assert result.plan_id == "test-plan-123"
        assert result.agent_id == "andi"
        assert result.status == PlanStatus.ACTIVE
        assert len(result.tasks) == 2
        assert result.tasks[0].status == TaskStatus.IN_PROGRESS
        assert result.tasks[1].status == TaskStatus.PENDING
        client.redis.hgetall.assert_called_once_with(
            RedisKeys.agent_plan("andi")
        )

    @pytest.mark.asyncio
    async def test_get_plan_not_found(self, client):
        """Should return None when plan doesn't exist."""
        client.redis.hgetall.return_value = {}

        result = await client.get_plan("andi")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_blocked_plan(self, client, sample_blocked_plan):
        """Should fetch blocked plan with resolution."""
        client.redis.hgetall.return_value = {
            b"plan_id": b"blocked-plan",
            b"agent_id": b"andi",
            b"objective": b"Blocked objective",
            b"summary": b"A blocked plan",
            b"status": b"blocked",
            b"tasks": b'[{"id": 0, "description": "Blocked task", "summary": "Task blocked", "context": "Context", "status": "blocked", "resolution": "Cannot proceed due to X"}]',
            b"current_task_id": b"0",
            b"created_at": b"2026-01-12T12:00:00+00:00",
            b"updated_at": b"2026-01-12T12:00:00+00:00",
        }

        result = await client.get_plan("andi")

        assert result is not None
        assert result.status == PlanStatus.BLOCKED
        assert result.tasks[0].status == TaskStatus.BLOCKED
        assert result.tasks[0].resolution == "Cannot proceed due to X"


class TestPlanMixinCreatePlan:
    """Tests for create_plan method."""

    @pytest.mark.asyncio
    async def test_create_plan(self, client, sample_plan):
        """Should create plan in Redis."""
        client.redis.eval.return_value = 1

        result = await client.create_plan(sample_plan)

        assert result is True
        # Verify eval was called with the correct key
        call_args = client.redis.eval.call_args
        assert RedisKeys.agent_plan("andi") in call_args[0]

    @pytest.mark.asyncio
    async def test_create_plan_overwrites(self, client, sample_plan):
        """Should overwrite existing plan (exists_ok=True)."""
        client.redis.eval.return_value = 1

        result = await client.create_plan(sample_plan)

        assert result is True


class TestPlanMixinUpdatePlanFields:
    """Tests for update_plan_fields method."""

    @pytest.mark.asyncio
    async def test_update_plan_fields(self, client):
        """Should update specific fields."""
        client.redis.eval.return_value = 1

        result = await client.update_plan_fields(
            "andi",
            status=PlanStatus.COMPLETED,
            current_task_id=2
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_update_plan_status_only(self, client):
        """Should update just status."""
        client.redis.eval.return_value = 1

        result = await client.update_plan_fields(
            "andi",
            status=PlanStatus.PAUSED
        )

        assert result is True


class TestPlanMixinDeletePlan:
    """Tests for delete_plan method."""

    @pytest.mark.asyncio
    async def test_delete_plan_exists(self, client):
        """Should delete existing plan."""
        client.redis.delete.return_value = 1

        result = await client.delete_plan("andi")

        assert result is True
        client.redis.delete.assert_called_once_with(
            RedisKeys.agent_plan("andi")
        )

    @pytest.mark.asyncio
    async def test_delete_plan_not_exists(self, client):
        """Should return False when plan doesn't exist."""
        client.redis.delete.return_value = 0

        result = await client.delete_plan("andi")

        assert result is False


class TestPlanMixinPlannerEnabled:
    """Tests for planner enabled flag operations."""

    @pytest.mark.asyncio
    async def test_is_planner_enabled_true(self, client):
        """Should return True when planner is enabled."""
        client.redis.get.return_value = b"true"

        result = await client.is_planner_enabled("andi")

        assert result is True
        client.redis.get.assert_called_once_with(
            RedisKeys.agent_planner_enabled("andi")
        )

    @pytest.mark.asyncio
    async def test_is_planner_enabled_false(self, client):
        """Should return False when planner is disabled."""
        client.redis.get.return_value = b"false"

        result = await client.is_planner_enabled("andi")

        assert result is False

    @pytest.mark.asyncio
    async def test_is_planner_enabled_not_set(self, client):
        """Should return False when flag not set."""
        client.redis.get.return_value = None

        result = await client.is_planner_enabled("andi")

        assert result is False

    @pytest.mark.asyncio
    async def test_is_planner_enabled_string_value(self, client):
        """Should handle string return value (not bytes)."""
        client.redis.get.return_value = "true"

        result = await client.is_planner_enabled("andi")

        assert result is True

    @pytest.mark.asyncio
    async def test_is_planner_enabled_case_insensitive(self, client):
        """Should handle TRUE, True, true."""
        client.redis.get.return_value = b"TRUE"

        result = await client.is_planner_enabled("andi")

        assert result is True

    @pytest.mark.asyncio
    async def test_set_planner_enabled_true(self, client):
        """Should set planner enabled to true."""
        await client.set_planner_enabled("andi", True)

        client.redis.set.assert_called_once_with(
            RedisKeys.agent_planner_enabled("andi"),
            "true"
        )

    @pytest.mark.asyncio
    async def test_set_planner_enabled_false(self, client):
        """Should set planner enabled to false."""
        await client.set_planner_enabled("andi", False)

        client.redis.set.assert_called_once_with(
            RedisKeys.agent_planner_enabled("andi"),
            "false"
        )


class TestRedisKeys:
    """Tests for plan-related Redis keys."""

    def test_agent_plan_key(self):
        """Should generate correct plan key."""
        key = RedisKeys.agent_plan("andi")
        assert key == "agent:andi:plan"

    def test_agent_planner_enabled_key(self):
        """Should generate correct planner enabled key."""
        key = RedisKeys.agent_planner_enabled("andi")
        assert key == "agent:andi:planner:enabled"

    def test_keys_different_agents(self):
        """Should generate different keys for different agents."""
        assert RedisKeys.agent_plan("andi") != RedisKeys.agent_plan("nova")
        assert RedisKeys.agent_planner_enabled("andi") != RedisKeys.agent_planner_enabled("nova")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
