# aim-mud-types/plan.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Plan data models for agent plan-following."""

import json
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, field_serializer

from ..helper import _utc_now, _datetime_to_unix, _unix_to_datetime


class PlanStatus(str, Enum):
    """Overall plan status values."""
    PENDING = "pending"       # Created but not started
    ACTIVE = "active"         # Currently executing
    PAUSED = "paused"         # Suspended by user
    COMPLETED = "completed"   # All tasks done
    BLOCKED = "blocked"       # Current task is blocked
    ABANDONED = "abandoned"   # Cancelled


class TaskStatus(str, Enum):
    """Individual task status values."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class PlanTask(BaseModel):
    """A single task within a plan.

    Attributes:
        id: Zero-indexed integer, auto-assigned during form building.
            Matches position in the tasks list. Not a UUID.
        description: Detailed explanation of what to do.
        summary: One-line summary for guidance injection.
        context: Why this task matters in the plan.
        status: Current task status.
        resolution: What happened when completed/blocked/skipped.
    """
    id: int
    description: str
    summary: str
    context: str
    status: TaskStatus = TaskStatus.PENDING
    resolution: Optional[str] = None


class AgentPlan(BaseModel):
    """Agent's execution plan stored in Redis.

    Stored in `agent:{id}:plan` Redis hash.

    Attributes:
        plan_id: UUID identifying this plan.
        agent_id: Agent this plan belongs to.
        objective: High-level goal being pursued.
        summary: One-sentence summary for injection.
        status: Current plan status.
        tasks: Ordered list of steps (JSON serialized in Redis).
        current_task_id: Index into tasks list.
        created_at: When plan was created.
        updated_at: Last modification time.
    """
    plan_id: str
    agent_id: str
    objective: str
    summary: str
    status: PlanStatus = PlanStatus.PENDING
    tasks: list[PlanTask]
    current_task_id: int = 0
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    @field_validator("tasks", mode="before")
    @classmethod
    def parse_tasks(cls, v):
        """Parse tasks from JSON string (Redis) or list (Python).

        When tasks are retrieved from Redis, they come as a JSON string.
        This validator converts them back to PlanTask objects.

        Args:
            v: Either a JSON string (from Redis) or list (from Python code)

        Returns:
            List of PlanTask objects

        Raises:
            ValueError: If v is a string but not valid JSON
        """
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                return [PlanTask.model_validate(t) for t in parsed]
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in tasks field: {e}")
        return v

    # Validators to parse Unix timestamps from Redis
    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def parse_required_datetime(cls, v):
        return _unix_to_datetime(v) or _utc_now()

    # Serializers to output Unix timestamps
    @field_serializer("created_at", "updated_at")
    def serialize_required_datetime(self, dt: datetime) -> int:
        return _datetime_to_unix(dt)
