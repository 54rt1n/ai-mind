# packages/aim-mud/tests/mud_tests/unit/test_plan.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for plan models."""

import json
import pytest
from datetime import datetime, timezone

from aim_mud_types.plan import (
    AgentPlan,
    PlanTask,
    TaskStatus,
    PlanStatus,
)


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_task_status_values(self):
        """All expected task statuses exist."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.BLOCKED.value == "blocked"
        assert TaskStatus.SKIPPED.value == "skipped"


class TestPlanStatus:
    """Tests for PlanStatus enum."""

    def test_plan_status_values(self):
        """All expected plan statuses exist."""
        assert PlanStatus.PENDING.value == "pending"
        assert PlanStatus.ACTIVE.value == "active"
        assert PlanStatus.PAUSED.value == "paused"
        assert PlanStatus.COMPLETED.value == "completed"
        assert PlanStatus.BLOCKED.value == "blocked"
        assert PlanStatus.ABANDONED.value == "abandoned"


class TestPlanTask:
    """Tests for PlanTask model."""

    def test_create_task(self):
        """Test creating a basic task."""
        task = PlanTask(
            id=0,
            description="Test description",
            summary="Test summary",
            context="Test context",
        )
        assert task.id == 0
        assert task.description == "Test description"
        assert task.summary == "Test summary"
        assert task.context == "Test context"
        assert task.status == TaskStatus.PENDING
        assert task.resolution is None

    def test_task_with_status(self):
        """Test task with explicit status."""
        task = PlanTask(
            id=1,
            description="desc",
            summary="sum",
            context="ctx",
            status=TaskStatus.COMPLETED,
            resolution="Done",
        )
        assert task.status == TaskStatus.COMPLETED
        assert task.resolution == "Done"

    def test_task_serialization(self):
        """Test task serializes to dict correctly."""
        task = PlanTask(
            id=2,
            description="d",
            summary="s",
            context="c",
            status=TaskStatus.IN_PROGRESS,
        )
        data = task.model_dump()
        assert data["id"] == 2
        assert data["description"] == "d"
        assert data["status"] == TaskStatus.IN_PROGRESS

    def test_task_blocked_with_resolution(self):
        """Test blocked task with resolution."""
        task = PlanTask(
            id=0,
            description="Blocked task",
            summary="Blocked",
            context="Testing blocked state",
            status=TaskStatus.BLOCKED,
            resolution="Cannot proceed due to missing dependency",
        )
        assert task.status == TaskStatus.BLOCKED
        assert "missing dependency" in task.resolution


class TestAgentPlan:
    """Tests for AgentPlan model."""

    def test_create_plan(self):
        """Test creating a basic plan."""
        tasks = [
            PlanTask(id=0, description="d", summary="s", context="c"),
        ]
        plan = AgentPlan(
            plan_id="test-123",
            agent_id="andi",
            objective="Test obj",
            summary="Test sum",
            tasks=tasks,
        )
        assert plan.plan_id == "test-123"
        assert plan.agent_id == "andi"
        assert plan.status == PlanStatus.PENDING
        assert plan.current_task_id == 0
        assert len(plan.tasks) == 1

    def test_parse_tasks_from_json(self):
        """Test parsing tasks from JSON string (Redis format)."""
        tasks_json = json.dumps([
            {"id": 0, "description": "d1", "summary": "s1", "context": "c1"},
            {"id": 1, "description": "d2", "summary": "s2", "context": "c2"},
        ])
        plan = AgentPlan(
            plan_id="test",
            agent_id="andi",
            objective="obj",
            summary="sum",
            tasks=tasks_json,
        )
        assert len(plan.tasks) == 2
        assert plan.tasks[0].id == 0
        assert plan.tasks[1].id == 1
        assert plan.tasks[0].description == "d1"
        assert plan.tasks[1].description == "d2"

    def test_parse_tasks_from_list(self):
        """Test parsing tasks from Python list."""
        tasks = [
            {"id": 0, "description": "d1", "summary": "s1", "context": "c1"},
            {"id": 1, "description": "d2", "summary": "s2", "context": "c2"},
        ]
        plan = AgentPlan(
            plan_id="test",
            agent_id="andi",
            objective="obj",
            summary="sum",
            tasks=tasks,
        )
        assert len(plan.tasks) == 2
        assert isinstance(plan.tasks[0], PlanTask)
        assert isinstance(plan.tasks[1], PlanTask)

    def test_parse_tasks_invalid_json(self):
        """Test parsing invalid JSON raises error."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            AgentPlan(
                plan_id="test",
                agent_id="andi",
                objective="obj",
                summary="sum",
                tasks="not valid json",
            )

    def test_timestamps_default(self):
        """Test timestamps are set by default."""
        plan = AgentPlan(
            plan_id="test",
            agent_id="andi",
            objective="obj",
            summary="sum",
            tasks=[],
        )
        assert isinstance(plan.created_at, datetime)
        assert isinstance(plan.updated_at, datetime)
        assert plan.created_at.tzinfo == timezone.utc
        assert plan.updated_at.tzinfo == timezone.utc

    def test_plan_with_multiple_tasks(self):
        """Test plan with multiple tasks in various states."""
        tasks = [
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
                status=TaskStatus.IN_PROGRESS,
            ),
            PlanTask(
                id=2,
                description="Third task",
                summary="Task 3",
                context="Context 3",
                status=TaskStatus.PENDING,
            ),
        ]
        plan = AgentPlan(
            plan_id="multi-task",
            agent_id="andi",
            objective="Multi-step objective",
            summary="A plan with multiple tasks",
            status=PlanStatus.ACTIVE,
            tasks=tasks,
            current_task_id=1,
        )
        assert len(plan.tasks) == 3
        assert plan.current_task_id == 1
        assert plan.status == PlanStatus.ACTIVE
        assert plan.tasks[0].status == TaskStatus.COMPLETED
        assert plan.tasks[1].status == TaskStatus.IN_PROGRESS
        assert plan.tasks[2].status == TaskStatus.PENDING

    def test_plan_serialization_round_trip(self):
        """Test plan serializes and deserializes correctly."""
        original = AgentPlan(
            plan_id="round-trip",
            agent_id="andi",
            objective="Test objective",
            summary="Test summary",
            status=PlanStatus.ACTIVE,
            tasks=[
                PlanTask(id=0, description="d", summary="s", context="c"),
            ],
            current_task_id=0,
        )

        # Serialize to dict
        data = original.model_dump()

        # Convert tasks to JSON as Redis would store it
        data["tasks"] = json.dumps([t if isinstance(t, dict) else t.model_dump() for t in data["tasks"]])

        # Deserialize back
        restored = AgentPlan.model_validate(data)

        assert restored.plan_id == original.plan_id
        assert restored.agent_id == original.agent_id
        assert restored.status == original.status
        assert len(restored.tasks) == 1
        assert restored.tasks[0].description == "d"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
