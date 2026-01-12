# aim/tool/impl/plan.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Plan execution tool - used by agents to update task status.

NOTE: This tool requires async execution. The base ToolImplementation only
defines sync execute(). This tool implements execute() to raise an error
directing callers to use execute_async().

The worker must handle this specially:
1. Check if tool has execute_async() method
2. If so, await it instead of calling execute()
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from .base import ToolImplementation

logger = logging.getLogger(__name__)


class PlanExecutionTool(ToolImplementation):
    """Tool for updating plan execution status.

    Requires external injection of redis_client and agent_id via set_context().
    This is done by the worker before making tool available.
    """

    def __init__(self) -> None:
        """Initialize tool. Context must be set via set_context() before use."""
        self._redis_client = None
        self._agent_id: Optional[str] = None

    def set_context(self, redis_client, agent_id: str) -> None:
        """Inject execution context.

        Args:
            redis_client: Async Redis client.
            agent_id: Current agent ID.
        """
        self._redis_client = redis_client
        self._agent_id = agent_id

    def execute(self, function_name: str, parameters: dict[str, Any]) -> dict[str, Any]:
        """Synchronous execute - not supported, use execute_async.

        Raises:
            RuntimeError: Always, use execute_async instead.
        """
        raise RuntimeError(
            f"PlanExecutionTool.{function_name} requires async execution. "
            "Use execute_async() instead."
        )

    async def execute_async(
        self,
        function_name: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute plan_update asynchronously.

        Args:
            function_name: Must be "plan_update".
            parameters: {"status": str, "resolution": str}

        Returns:
            Result dictionary with status and new task info.
        """
        if function_name == "plan_update":
            return await self._plan_update(
                parameters["status"],
                parameters["resolution"],
            )
        return {"error": f"Unknown function: {function_name}"}

    async def _plan_update(self, status: str, resolution: str) -> dict[str, Any]:
        """Update current task status and advance to next if completed.

        Args:
            status: New status (completed, blocked, skipped).
            resolution: Description of what happened.

        Returns:
            {"status": "ok", "plan_status": str, "new_task_id": int, "next_task": str|None}
            or {"error": str} on failure.
        """
        if self._redis_client is None or self._agent_id is None:
            return {"error": "Context not set - call set_context() first"}

        from aim_mud_types import RedisMUDClient, TaskStatus, PlanStatus

        client = RedisMUDClient(self._redis_client)
        plan = await client.get_plan(self._agent_id)

        if plan is None:
            return {"error": "No active plan"}

        if plan.current_task_id >= len(plan.tasks):
            return {"error": "No current task (plan may be complete)"}

        # Update current task
        task = plan.tasks[plan.current_task_id]
        task.status = TaskStatus(status)
        task.resolution = resolution

        # Handle state transitions
        next_task_summary: Optional[str] = None

        if status == "completed":
            # Find next pending task
            next_id = self._find_next_pending_task(plan)

            if next_id is not None:
                plan.current_task_id = next_id
                plan.tasks[next_id].status = TaskStatus.IN_PROGRESS
                next_task_summary = plan.tasks[next_id].summary
            else:
                plan.status = PlanStatus.COMPLETED

        elif status == "blocked":
            plan.status = PlanStatus.BLOCKED

        elif status == "skipped":
            # Find next pending task
            next_id = self._find_next_pending_task(plan)

            if next_id is not None:
                plan.current_task_id = next_id
                plan.tasks[next_id].status = TaskStatus.IN_PROGRESS
                next_task_summary = plan.tasks[next_id].summary
            else:
                plan.status = PlanStatus.COMPLETED

        plan.updated_at = datetime.now(timezone.utc)

        # Serialize tasks for Redis
        tasks_json = json.dumps([t.model_dump() for t in plan.tasks])

        await client.update_plan_fields(
            self._agent_id,
            tasks=tasks_json,
            current_task_id=plan.current_task_id,
            status=plan.status.value,
            updated_at=plan.updated_at.isoformat(),
        )

        return {
            "status": "ok",
            "plan_status": plan.status.value,
            "new_task_id": plan.current_task_id,
            "next_task": next_task_summary,
        }

    def _find_next_pending_task(self, plan) -> Optional[int]:
        """Find the next pending task after current_task_id.

        Args:
            plan: The AgentPlan to search.

        Returns:
            Index of next pending task, or None if none found.
        """
        from aim_mud_types import TaskStatus

        for i in range(plan.current_task_id + 1, len(plan.tasks)):
            if plan.tasks[i].status == TaskStatus.PENDING:
                return i
        return None
