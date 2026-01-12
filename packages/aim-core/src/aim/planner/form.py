# aim/planner/form.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Form builder state machine for Stage 2 of plan creation."""

from enum import Enum
from typing import Optional, TYPE_CHECKING
import uuid

from pydantic import BaseModel

from .constants import FORM_BUILDER_MAX_TASKS

if TYPE_CHECKING:
    from aim_mud_types.plan import AgentPlan


class FormState(str, Enum):
    """States in the form building state machine."""

    ADD_TASK = "add_task"
    VERIFY_TASK = "verify_task"
    CONFIRM = "confirm"
    COMPLETE = "complete"


class DraftTask(BaseModel):
    """In-memory task during form building.

    Attributes:
        id: Zero-indexed, auto-assigned as len(tasks) when added.
        description: Detailed explanation of what to do.
        summary: One-line summary shown during execution.
        context: Why this task matters.
        verified: Whether the task has been verified.
    """

    id: int
    description: str
    summary: str
    context: str
    verified: bool = False


class PlanFormBuilder:
    """State machine for building a plan through tool calls.

    Thread-safe: No. Single-threaded use only.

    The form builder guides an LLM through creating a plan:
    1. ADD_TASK state: LLM adds tasks one by one
    2. VERIFY_TASK state: LLM reviews each task
    3. CONFIRM state: LLM sets summary and confirms
    4. COMPLETE state: Plan ready for conversion

    Attributes:
        objective: The plan objective (high-level goal).
        summary: One-sentence plan summary (set during CONFIRM).
        deliberation_context: Output from Stage 1 deliberation.
        tasks: List of draft tasks being built.
        state: Current state in the state machine.
        current_verify_index: Index of task being verified.
    """

    def __init__(self, objective: str, deliberation_context: str):
        """Initialize form builder.

        Args:
            objective: The plan objective (high-level goal).
            deliberation_context: Output from Stage 1 deliberation.
        """
        self.objective = objective
        self.summary: Optional[str] = None
        self.deliberation_context = deliberation_context
        self.tasks: list[DraftTask] = []
        self.state = FormState.ADD_TASK
        self.current_verify_index = 0

    def get_prompt(self) -> str:
        """Get the current prompt based on state.

        Returns:
            User prompt string for current state.
        """
        if self.state == FormState.ADD_TASK:
            return self._add_task_prompt()
        elif self.state == FormState.VERIFY_TASK:
            return self._verify_task_prompt()
        elif self.state == FormState.CONFIRM:
            return self._confirm_prompt()
        return ""

    def _add_task_prompt(self) -> str:
        """Generate prompt for ADD_TASK state."""
        existing = (
            "\n".join(f"  {i+1}. {t.summary}" for i, t in enumerate(self.tasks))
            or "  (none yet)"
        )

        return f"""Based on the deliberation, add tasks to the plan.

Objective: {self.objective}

Current tasks:
{existing}

Use the **plan_add_task** tool to add a task, or **plan_done_adding** when finished.

Each task needs:
- description: Detailed explanation of what to do
- summary: One-line summary (shown during execution)
- context: Why this task matters"""

    def _verify_task_prompt(self) -> str:
        """Generate prompt for VERIFY_TASK state."""
        task = self.tasks[self.current_verify_index]
        return f"""Verify task {self.current_verify_index + 1} of {len(self.tasks)}:

**{task.summary}**
Description: {task.description}
Context: {task.context}

Use **plan_verify_task** with approved=true to confirm, or approved=false with a revision."""

    def _confirm_prompt(self) -> str:
        """Generate prompt for CONFIRM state."""
        task_list = "\n".join(
            f"  {i+1}. {t.summary}" for i, t in enumerate(self.tasks)
        )
        return f"""Plan ready for confirmation:

Objective: {self.objective}
Summary: {self.summary or "(needs summary - use plan_set_summary first)"}

Tasks:
{task_list}

Use **plan_set_summary** if not set, then **plan_confirm** to save and activate.
Or use **plan_edit_task** to revise a task."""

    # Tool handlers - all return dict with status and relevant info

    def add_task(self, description: str, summary: str, context: str) -> dict:
        """Handle plan_add_task tool call.

        Args:
            description: Detailed task description.
            summary: One-line summary.
            context: Why this task matters.

        Returns:
            {"status": "added", "task_id": int, "summary": str} or error dict.
        """
        if len(self.tasks) >= FORM_BUILDER_MAX_TASKS:
            return {
                "status": "error",
                "message": f"Maximum {FORM_BUILDER_MAX_TASKS} tasks allowed",
            }

        task = DraftTask(
            id=len(self.tasks),
            description=description,
            summary=summary,
            context=context,
        )
        self.tasks.append(task)
        return {"status": "added", "task_id": task.id, "summary": summary}

    def done_adding(self) -> dict:
        """Handle plan_done_adding tool call.

        Returns:
            {"status": "ok", "next": "verify", "task_count": int} or error.
        """
        if not self.tasks:
            return {"status": "error", "message": "No tasks added yet"}
        self.state = FormState.VERIFY_TASK
        self.current_verify_index = 0
        return {"status": "ok", "next": "verify", "task_count": len(self.tasks)}

    def verify_task(self, approved: bool, revision: Optional[str] = None) -> dict:
        """Handle plan_verify_task tool call.

        Args:
            approved: True to approve current task.
            revision: If not approved, the revised description.

        Returns:
            {"status": "ok", "next": "verify"|"confirm"} or revision_needed.
        """
        task = self.tasks[self.current_verify_index]

        if approved:
            task.verified = True

            # Find next unverified task, skipping already-verified ones
            next_index = self._find_next_unverified_task(self.current_verify_index + 1)

            if next_index is None:
                # All tasks verified
                self.state = FormState.CONFIRM
                return {"status": "ok", "next": "confirm"}

            self.current_verify_index = next_index
            return {
                "status": "ok",
                "next": "verify",
                "task_index": self.current_verify_index,
            }
        else:
            # Not approved - back to add state to revise
            if revision:
                task.description = revision
            self.state = FormState.ADD_TASK
            return {"status": "revision_needed", "task_id": task.id}

    def _find_next_unverified_task(self, start_index: int) -> Optional[int]:
        """Find the next unverified task starting from start_index.

        Args:
            start_index: Index to start searching from.

        Returns:
            Index of next unverified task, or None if all verified.
        """
        for i in range(start_index, len(self.tasks)):
            if not self.tasks[i].verified:
                return i
        return None

    def edit_task(self, task_id: int, field: str, new_value: str) -> dict:
        """Handle plan_edit_task tool call.

        Args:
            task_id: Task ID to edit.
            field: Field name (description, summary, context).
            new_value: New value for the field.

        Returns:
            {"status": "ok", "next": "verify"} or error.
        """
        if task_id >= len(self.tasks):
            return {"status": "error", "message": f"No task {task_id}"}

        task = self.tasks[task_id]
        if field == "description":
            task.description = new_value
        elif field == "summary":
            task.summary = new_value
        elif field == "context":
            task.context = new_value
        else:
            return {"status": "error", "message": f"Unknown field: {field}"}

        task.verified = False
        self.current_verify_index = task_id
        self.state = FormState.VERIFY_TASK
        return {"status": "ok", "next": "verify"}

    def set_summary(self, summary: str) -> dict:
        """Handle plan_set_summary tool call.

        Args:
            summary: One-sentence plan summary.

        Returns:
            {"status": "ok"}
        """
        self.summary = summary
        return {"status": "ok"}

    def confirm(self) -> dict:
        """Handle plan_confirm tool call.

        Returns:
            {"status": "complete", "task_count": int} or error.
        """
        if not self.summary:
            return {
                "status": "error",
                "message": "Plan needs a summary first (use plan_set_summary)",
            }
        if not all(t.verified for t in self.tasks):
            unverified = [t.id for t in self.tasks if not t.verified]
            return {"status": "error", "message": f"Tasks not verified: {unverified}"}

        self.state = FormState.COMPLETE
        return {"status": "complete", "task_count": len(self.tasks)}

    def to_plan(self, agent_id: str) -> "AgentPlan":
        """Convert draft to final AgentPlan for Redis storage.

        Args:
            agent_id: Agent identifier for the plan.

        Returns:
            AgentPlan ready for Redis storage.

        Raises:
            ValueError: If state is not COMPLETE.
        """
        if self.state != FormState.COMPLETE:
            raise ValueError(f"Cannot convert incomplete form (state={self.state})")

        from aim_mud_types.plan import AgentPlan, PlanTask, PlanStatus, TaskStatus

        tasks = [
            PlanTask(
                id=t.id,
                description=t.description,
                summary=t.summary,
                context=t.context,
                status=TaskStatus.IN_PROGRESS if t.id == 0 else TaskStatus.PENDING,
            )
            for t in self.tasks
        ]

        return AgentPlan(
            plan_id=str(uuid.uuid4()),
            agent_id=agent_id,
            objective=self.objective,
            summary=self.summary,
            status=PlanStatus.ACTIVE,
            tasks=tasks,
            current_task_id=0,
        )
