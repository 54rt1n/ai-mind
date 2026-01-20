# aim/dreamer/server/scheduler.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""DAG resolution and Redis queue operations."""

from datetime import datetime, timezone
from typing import Optional
import redis.asyncio as redis
import json

from aim.dreamer.core.models import StepJob, Scenario, StepStatus, StepDefinition

from .state import StateStore


class Scheduler:
    """Manages step queue and DAG dependency resolution."""

    def __init__(self, redis_client: redis.Redis, state_store: StateStore):
        """Initialize scheduler with Redis client and state store."""
        self.redis = redis_client
        self.state_store = state_store
        self.queue_key = "dreamer:queue:steps"
        self.delayed_key = "dreamer:queue:steps:delayed"

    async def enqueue_step(
        self,
        pipeline_id: str,
        step_id: str,
        priority: int = 0,
        delay: Optional[int] = None,
    ) -> None:
        """Add a step job to the queue.

        Args:
            pipeline_id: Pipeline identifier
            step_id: Step identifier
            priority: Priority (lower = higher priority, default 0)
            delay: Optional delay in seconds before job becomes available
        """
        # Create StepJob
        job = StepJob(
            pipeline_id=pipeline_id,
            step_id=step_id,
            attempt=1,
            max_attempts=3,
            enqueued_at=datetime.now(timezone.utc),
            priority=priority,
        )

        # Serialize to JSON
        job_json = job.model_dump_json()

        if delay and delay > 0:
            # Add to delayed queue with score = now + delay
            score = datetime.now(timezone.utc).timestamp() + delay
            await self.redis.zadd(self.delayed_key, {job_json: score})
        else:
            # Add to main queue
            # LPUSH for priority 0 (most common case)
            # For priority queues, we'd need a more sophisticated approach
            # For now, we'll use simple FIFO with LPUSH
            await self.redis.lpush(self.queue_key, job_json)

    async def pop_step_job(self, timeout: int = 0) -> Optional[StepJob]:
        """Pop the next step job from the queue (blocking).

        Args:
            timeout: Timeout in seconds (0 = block indefinitely)

        Returns:
            StepJob if available, None if timeout and queue empty
        """
        # BRPOP returns tuple: (key, value) or None on timeout
        result = await self.redis.brpop(self.queue_key, timeout=timeout)

        if result is None:
            return None

        # result is tuple: (key_name, job_json)
        _, job_json = result

        # Deserialize from JSON
        if isinstance(job_json, bytes):
            job_json = job_json.decode('utf-8')

        return StepJob.model_validate_json(job_json)

    async def requeue_step(self, job: StepJob, delay: int = 0) -> None:
        """Requeue a failed step with optional delay.

        Args:
            job: StepJob to requeue
            delay: Delay in seconds before job becomes available again
        """
        # Update enqueued_at to current time
        job = job.model_copy(update={'enqueued_at': datetime.now(timezone.utc)})

        # Serialize to JSON
        job_json = job.model_dump_json()

        if delay > 0:
            # Add to delayed queue with score = now + delay
            score = datetime.now(timezone.utc).timestamp() + delay
            await self.redis.zadd(self.delayed_key, {job_json: score})
        else:
            # Add back to main queue
            await self.redis.lpush(self.queue_key, job_json)

    async def process_delayed_jobs(self) -> int:
        """Move due delayed jobs to the main queue. Returns count moved.

        Returns:
            Number of jobs moved from delayed to main queue
        """
        # Get current timestamp
        now = datetime.now(timezone.utc).timestamp()

        # Get all jobs with score <= now (due jobs)
        # ZRANGEBYSCORE returns list of members with scores in range
        due_jobs = await self.redis.zrangebyscore(
            self.delayed_key,
            min=0,
            max=now,
        )

        if not due_jobs:
            return 0

        # Move each job to main queue
        pipe = self.redis.pipeline()
        for job_json in due_jobs:
            # Add to main queue
            pipe.lpush(self.queue_key, job_json)
            # Remove from delayed queue
            pipe.zrem(self.delayed_key, job_json)

        await pipe.execute()

        return len(due_jobs)

    async def clear_pipeline_jobs(self, pipeline_id: str) -> int:
        """Remove all queued jobs for a specific pipeline.

        Args:
            pipeline_id: Pipeline identifier to clear jobs for

        Returns:
            Number of jobs removed
        """
        removed = 0

        # Get all jobs from main queue
        all_jobs = await self.redis.lrange(self.queue_key, 0, -1)

        # Find and remove jobs for this pipeline
        for job_json in all_jobs:
            if isinstance(job_json, bytes):
                job_json_str = job_json.decode('utf-8')
            else:
                job_json_str = job_json

            try:
                job_data = json.loads(job_json_str)
                if job_data.get('pipeline_id') == pipeline_id:
                    await self.redis.lrem(self.queue_key, 1, job_json)
                    removed += 1
            except json.JSONDecodeError:
                continue

        # Also clear from delayed queue
        delayed_jobs = await self.redis.zrange(self.delayed_key, 0, -1)
        for job_json in delayed_jobs:
            if isinstance(job_json, bytes):
                job_json_str = job_json.decode('utf-8')
            else:
                job_json_str = job_json

            try:
                job_data = json.loads(job_json_str)
                if job_data.get('pipeline_id') == pipeline_id:
                    await self.redis.zrem(self.delayed_key, job_json)
                    removed += 1
            except json.JSONDecodeError:
                continue

        return removed

    async def all_deps_complete(
        self, pipeline_id: str, step: StepDefinition
    ) -> bool:
        """Check if all dependencies for a step are complete.

        Args:
            pipeline_id: Pipeline identifier
            step: StepDefinition to check dependencies for

        Returns:
            True if all dependencies are complete, False otherwise
        """
        # If no dependencies, return True
        if not step.depends_on:
            return True

        # Check each dependency
        for dep_id in step.depends_on:
            status = await self.state_store.get_step_status(pipeline_id, dep_id)
            if status != StepStatus.COMPLETE:
                return False

        return True

    async def mark_complete(self, pipeline_id: str, step_id: str) -> None:
        """Mark a step as complete.

        Args:
            pipeline_id: Pipeline identifier
            step_id: Step identifier
        """
        await self.state_store.set_step_status(
            pipeline_id, step_id, StepStatus.COMPLETE
        )

    async def mark_failed(
        self, pipeline_id: str, step_id: str, error: str
    ) -> None:
        """Mark a step as failed and store the error message.

        Args:
            pipeline_id: Pipeline identifier
            step_id: Step identifier
            error: Error message
        """
        await self.state_store.set_step_status(
            pipeline_id, step_id, StepStatus.FAILED
        )
        await self.state_store.set_step_error(pipeline_id, step_id, error)

    async def check_pipeline_complete(
        self, pipeline_id: str, scenario: Scenario
    ) -> bool:
        """Check if all steps in the pipeline are complete.

        Args:
            pipeline_id: Pipeline identifier
            scenario: Scenario definition

        Returns:
            True if all steps are complete, False otherwise
        """
        # Check each step in the scenario
        for step_id in scenario.steps.keys():
            status = await self.state_store.get_step_status(pipeline_id, step_id)
            if status != StepStatus.COMPLETE:
                return False

        return True
