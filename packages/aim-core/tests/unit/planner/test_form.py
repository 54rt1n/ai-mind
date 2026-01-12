# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for PlanFormBuilder state machine."""

import pytest

from aim.planner.form import PlanFormBuilder, FormState, DraftTask
from aim.planner.constants import FORM_BUILDER_MAX_TASKS


class TestDraftTask:
    """Tests for DraftTask model."""

    def test_create_task(self):
        """Test creating a basic draft task."""
        task = DraftTask(
            id=0,
            description="Test description",
            summary="Test summary",
            context="Test context",
        )
        assert task.id == 0
        assert task.description == "Test description"
        assert task.verified is False

    def test_task_with_verified_true(self):
        """Test creating a verified task."""
        task = DraftTask(
            id=1,
            description="Verified task",
            summary="Verified",
            context="Context",
            verified=True,
        )
        assert task.verified is True


class TestPlanFormBuilder:
    """Tests for PlanFormBuilder state machine."""

    @pytest.fixture
    def builder(self):
        """Create a form builder for testing."""
        return PlanFormBuilder(
            objective="Test objective",
            deliberation_context="Test deliberation output",
        )

    def test_initial_state(self, builder):
        """Test initial state is ADD_TASK."""
        assert builder.state == FormState.ADD_TASK
        assert builder.tasks == []
        assert builder.summary is None
        assert builder.objective == "Test objective"
        assert builder.deliberation_context == "Test deliberation output"

    def test_add_task(self, builder):
        """Test adding a task."""
        result = builder.add_task(
            description="Do something",
            summary="Task 1",
            context="Because reasons",
        )
        assert result["status"] == "added"
        assert result["task_id"] == 0
        assert result["summary"] == "Task 1"
        assert len(builder.tasks) == 1
        assert builder.tasks[0].description == "Do something"

    def test_add_multiple_tasks(self, builder):
        """Test adding multiple tasks with correct IDs."""
        builder.add_task("desc1", "sum1", "ctx1")
        builder.add_task("desc2", "sum2", "ctx2")
        builder.add_task("desc3", "sum3", "ctx3")

        assert len(builder.tasks) == 3
        assert builder.tasks[0].id == 0
        assert builder.tasks[1].id == 1
        assert builder.tasks[2].id == 2

    def test_add_task_max_limit(self, builder):
        """Test adding tasks up to max limit."""
        for i in range(FORM_BUILDER_MAX_TASKS):
            builder.add_task(f"desc {i}", f"sum {i}", f"ctx {i}")

        result = builder.add_task("one more", "sum", "ctx")
        assert result["status"] == "error"
        assert "Maximum" in result["message"]
        assert len(builder.tasks) == FORM_BUILDER_MAX_TASKS

    def test_done_adding_no_tasks(self, builder):
        """Test done_adding with no tasks fails."""
        result = builder.done_adding()
        assert result["status"] == "error"
        assert "No tasks" in result["message"]
        assert builder.state == FormState.ADD_TASK

    def test_done_adding_transitions_to_verify(self, builder):
        """Test done_adding transitions to verify state."""
        builder.add_task("desc", "sum", "ctx")
        result = builder.done_adding()

        assert result["status"] == "ok"
        assert result["next"] == "verify"
        assert result["task_count"] == 1
        assert builder.state == FormState.VERIFY_TASK
        assert builder.current_verify_index == 0

    def test_verify_task_approved(self, builder):
        """Test approving a task."""
        builder.add_task("desc", "sum", "ctx")
        builder.done_adding()

        result = builder.verify_task(approved=True)
        assert result["status"] == "ok"
        assert result["next"] == "confirm"
        assert builder.tasks[0].verified is True
        assert builder.state == FormState.CONFIRM

    def test_verify_task_multiple_tasks(self, builder):
        """Test verifying multiple tasks."""
        builder.add_task("desc1", "sum1", "ctx1")
        builder.add_task("desc2", "sum2", "ctx2")
        builder.done_adding()

        # Verify first task
        result = builder.verify_task(approved=True)
        assert result["next"] == "verify"
        assert result["task_index"] == 1
        assert builder.state == FormState.VERIFY_TASK
        assert builder.current_verify_index == 1

        # Verify second task
        result = builder.verify_task(approved=True)
        assert result["next"] == "confirm"
        assert builder.state == FormState.CONFIRM

    def test_verify_task_not_approved(self, builder):
        """Test rejecting a task."""
        builder.add_task("desc", "sum", "ctx")
        builder.done_adding()

        result = builder.verify_task(approved=False, revision="better desc")
        assert result["status"] == "revision_needed"
        assert result["task_id"] == 0
        assert builder.state == FormState.ADD_TASK
        assert builder.tasks[0].description == "better desc"
        assert builder.tasks[0].verified is False

    def test_verify_task_not_approved_no_revision(self, builder):
        """Test rejecting a task without providing revision."""
        builder.add_task("original desc", "sum", "ctx")
        builder.done_adding()

        result = builder.verify_task(approved=False)
        assert result["status"] == "revision_needed"
        assert builder.state == FormState.ADD_TASK
        # Original description should be unchanged
        assert builder.tasks[0].description == "original desc"

    def test_edit_task(self, builder):
        """Test editing a task field."""
        builder.add_task("desc", "sum", "ctx")
        builder.add_task("desc2", "sum2", "ctx2")
        builder.done_adding()
        builder.verify_task(approved=True)
        builder.verify_task(approved=True)

        result = builder.edit_task(task_id=0, field="summary", new_value="new sum")
        assert result["status"] == "ok"
        assert result["next"] == "verify"
        assert builder.tasks[0].summary == "new sum"
        assert builder.tasks[0].verified is False
        assert builder.state == FormState.VERIFY_TASK
        assert builder.current_verify_index == 0

    def test_edit_task_description(self, builder):
        """Test editing task description."""
        builder.add_task("desc", "sum", "ctx")
        result = builder.edit_task(task_id=0, field="description", new_value="new desc")
        assert result["status"] == "ok"
        assert builder.tasks[0].description == "new desc"

    def test_edit_task_context(self, builder):
        """Test editing task context."""
        builder.add_task("desc", "sum", "ctx")
        result = builder.edit_task(task_id=0, field="context", new_value="new ctx")
        assert result["status"] == "ok"
        assert builder.tasks[0].context == "new ctx"

    def test_edit_task_invalid_id(self, builder):
        """Test editing non-existent task."""
        result = builder.edit_task(task_id=99, field="summary", new_value="x")
        assert result["status"] == "error"
        assert "No task 99" in result["message"]

    def test_edit_task_invalid_field(self, builder):
        """Test editing with invalid field."""
        builder.add_task("desc", "sum", "ctx")
        result = builder.edit_task(task_id=0, field="invalid", new_value="x")
        assert result["status"] == "error"
        assert "Unknown field" in result["message"]

    def test_set_summary(self, builder):
        """Test setting plan summary."""
        result = builder.set_summary("My plan summary")
        assert result["status"] == "ok"
        assert builder.summary == "My plan summary"

    def test_confirm_no_summary(self, builder):
        """Test confirm fails without summary."""
        builder.add_task("desc", "sum", "ctx")
        builder.done_adding()
        builder.verify_task(approved=True)

        result = builder.confirm()
        assert result["status"] == "error"
        assert "summary" in result["message"]

    def test_confirm_unverified_tasks(self, builder):
        """Test confirm fails with unverified tasks."""
        builder.add_task("desc", "sum", "ctx")
        builder.done_adding()
        builder.set_summary("Summary")
        # Skip verification by forcing state
        builder.state = FormState.CONFIRM

        result = builder.confirm()
        assert result["status"] == "error"
        assert "not verified" in result["message"]
        assert "0" in result["message"]

    def test_confirm_success(self, builder):
        """Test successful confirmation."""
        builder.add_task("desc", "sum", "ctx")
        builder.done_adding()
        builder.verify_task(approved=True)
        builder.set_summary("Summary")

        result = builder.confirm()
        assert result["status"] == "complete"
        assert result["task_count"] == 1
        assert builder.state == FormState.COMPLETE

    def test_to_plan_incomplete(self, builder):
        """Test to_plan fails when not complete."""
        builder.add_task("desc", "sum", "ctx")

        with pytest.raises(ValueError, match="incomplete"):
            builder.to_plan("andi")

    def test_to_plan_success(self, builder):
        """Test converting to AgentPlan."""
        builder.add_task("desc1", "sum1", "ctx1")
        builder.add_task("desc2", "sum2", "ctx2")
        builder.done_adding()
        builder.verify_task(approved=True)
        builder.verify_task(approved=True)
        builder.set_summary("Plan summary")
        builder.confirm()

        plan = builder.to_plan("andi")

        assert plan.agent_id == "andi"
        assert plan.objective == "Test objective"
        assert plan.summary == "Plan summary"
        assert len(plan.tasks) == 2
        # First task should be IN_PROGRESS
        assert plan.tasks[0].status.value == "in_progress"
        # Second task should be PENDING
        assert plan.tasks[1].status.value == "pending"
        # Plan should be ACTIVE
        assert plan.status.value == "active"
        assert plan.current_task_id == 0
        # plan_id should be a UUID string
        assert len(plan.plan_id) == 36  # UUID format

    def test_get_prompt_add_task(self, builder):
        """Test getting add task prompt."""
        prompt = builder.get_prompt()
        assert "plan_add_task" in prompt
        assert "plan_done_adding" in prompt
        assert "Objective: Test objective" in prompt
        assert "(none yet)" in prompt

    def test_get_prompt_add_task_with_tasks(self, builder):
        """Test add task prompt shows existing tasks."""
        builder.add_task("desc", "First task", "ctx")
        prompt = builder.get_prompt()
        assert "1. First task" in prompt
        assert "(none yet)" not in prompt

    def test_get_prompt_verify_task(self, builder):
        """Test getting verify task prompt."""
        builder.add_task("desc", "sum", "ctx")
        builder.done_adding()

        prompt = builder.get_prompt()
        assert "Verify task 1 of 1" in prompt
        assert "plan_verify_task" in prompt
        assert "**sum**" in prompt

    def test_get_prompt_confirm(self, builder):
        """Test getting confirm prompt."""
        builder.add_task("desc", "sum", "ctx")
        builder.done_adding()
        builder.verify_task(approved=True)

        prompt = builder.get_prompt()
        assert "plan_confirm" in prompt
        assert "plan_set_summary" in prompt
        assert "plan_edit_task" in prompt
        assert "(needs summary" in prompt

    def test_get_prompt_confirm_with_summary(self, builder):
        """Test confirm prompt shows summary when set."""
        builder.add_task("desc", "sum", "ctx")
        builder.done_adding()
        builder.verify_task(approved=True)
        builder.set_summary("My summary")

        prompt = builder.get_prompt()
        assert "Summary: My summary" in prompt
        assert "(needs summary" not in prompt

    def test_get_prompt_complete_state(self, builder):
        """Test get_prompt returns empty for COMPLETE state."""
        builder.add_task("desc", "sum", "ctx")
        builder.done_adding()
        builder.verify_task(approved=True)
        builder.set_summary("Summary")
        builder.confirm()

        prompt = builder.get_prompt()
        assert prompt == ""

    def test_full_workflow(self, builder):
        """Test complete form building workflow."""
        # Add tasks
        builder.add_task("First thing", "Task 1", "Sets foundation")
        builder.add_task("Second thing", "Task 2", "Builds on first")
        builder.add_task("Third thing", "Task 3", "Completes work")

        # Done adding
        result = builder.done_adding()
        assert result["task_count"] == 3

        # Verify all tasks
        builder.verify_task(approved=True)
        builder.verify_task(approved=True)
        builder.verify_task(approved=True)

        # Set summary and confirm
        builder.set_summary("Three step plan")
        result = builder.confirm()

        assert result["status"] == "complete"

        # Convert to plan
        plan = builder.to_plan("andi")
        assert len(plan.tasks) == 3
        assert plan.current_task_id == 0
        assert all(
            plan.tasks[i].id == i for i in range(3)
        ), "Task IDs should match positions"

    def test_workflow_with_revision(self, builder):
        """Test workflow where a task is revised during verification."""
        builder.add_task("First task", "Task 1", "Context 1")
        builder.add_task("Second task", "Task 2", "Context 2")
        builder.done_adding()

        # Approve first task
        builder.verify_task(approved=True)

        # Reject second task
        builder.verify_task(approved=False, revision="Revised second task")

        # Now we're back in ADD_TASK state
        assert builder.state == FormState.ADD_TASK
        assert builder.tasks[1].description == "Revised second task"

        # Continue to done_adding again
        builder.done_adding()

        # The verify index should reset to 0
        assert builder.current_verify_index == 0

        # Re-verify both tasks
        builder.verify_task(approved=True)
        builder.verify_task(approved=True)

        # Now we should be in CONFIRM
        assert builder.state == FormState.CONFIRM

    def test_workflow_with_edit_after_verification(self, builder):
        """Test workflow where a task is edited after all verification."""
        builder.add_task("First task", "Task 1", "Context 1")
        builder.add_task("Second task", "Task 2", "Context 2")
        builder.done_adding()
        builder.verify_task(approved=True)
        builder.verify_task(approved=True)

        # In CONFIRM state, edit task 0
        builder.edit_task(task_id=0, field="summary", new_value="Revised Task 1")

        # Should be back in VERIFY_TASK at task 0
        assert builder.state == FormState.VERIFY_TASK
        assert builder.current_verify_index == 0
        assert builder.tasks[0].verified is False
        # Task 1 should still be verified
        assert builder.tasks[1].verified is True

        # Re-verify task 0
        builder.verify_task(approved=True)

        # Should be in CONFIRM again
        assert builder.state == FormState.CONFIRM
