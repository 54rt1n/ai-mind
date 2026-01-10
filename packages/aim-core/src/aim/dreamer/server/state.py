# aim/dreamer/server/state.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Redis state persistence for pipeline execution."""

from typing import Optional
import redis.asyncio as redis

from ..core.models import PipelineState, StepStatus, Scenario
from ..core.dialogue.models import DialogueState


class StateStore:
    """Redis-backed state management for pipelines."""

    def __init__(self, redis_client: redis.Redis, key_prefix: str = "dreamer"):
        """Initialize state store with Redis client."""
        self.redis = redis_client
        self.key_prefix = key_prefix

    def _state_key(self, pipeline_id: str) -> str:
        """Generate Redis key for pipeline state."""
        return f"{self.key_prefix}:pipeline:{pipeline_id}:state"

    def _dag_key(self, pipeline_id: str) -> str:
        """Generate Redis key for DAG status hash."""
        return f"{self.key_prefix}:pipeline:{pipeline_id}:dag"

    def _lock_key(self, pipeline_id: str, step_id: str) -> str:
        """Generate Redis key for step lock."""
        return f"{self.key_prefix}:pipeline:{pipeline_id}:lock:{step_id}"

    def _errors_key(self, pipeline_id: str) -> str:
        """Generate Redis key for step errors hash."""
        return f"{self.key_prefix}:pipeline:{pipeline_id}:errors"

    async def save_state(self, state: PipelineState) -> None:
        """Persist pipeline state to Redis."""
        key = self._state_key(state.pipeline_id)
        # Serialize to JSON using Pydantic's model_dump_json
        state_json = state.model_dump_json()
        await self.redis.set(key, state_json)

    async def load_state(self, pipeline_id: str) -> Optional[PipelineState]:
        """Load pipeline state from Redis."""
        key = self._state_key(pipeline_id)
        state_json = await self.redis.get(key)

        if state_json is None:
            return None

        # Deserialize from JSON using Pydantic's model_validate_json
        return PipelineState.model_validate_json(state_json)

    async def get_state_type(self, pipeline_id: str) -> Optional[str]:
        """Check what type of state is stored for a pipeline.

        Returns:
            'dialogue' if DialogueState, 'pipeline' if PipelineState, None if not found
        """
        import json
        key = self._state_key(pipeline_id)
        state_json = await self.redis.get(key)

        if state_json is None:
            return None

        # Parse just enough to check the type
        data = json.loads(state_json)
        if 'strategy_name' in data:
            return 'dialogue'
        return 'pipeline'

    async def delete_state(self, pipeline_id: str) -> None:
        """Delete pipeline state from Redis."""
        state_key = self._state_key(pipeline_id)
        dag_key = self._dag_key(pipeline_id)
        errors_key = self._errors_key(pipeline_id)

        # Delete state, DAG, and errors
        await self.redis.delete(state_key, dag_key, errors_key)

    # === Dialogue State Methods ===

    def _dialogue_state_key(self, pipeline_id: str) -> str:
        """Generate Redis key for dialogue state (same as pipeline state)."""
        # Use same key pattern as pipeline state - they're mutually exclusive
        return f"{self.key_prefix}:pipeline:{pipeline_id}:state"

    async def save_dialogue_state(self, state: DialogueState) -> None:
        """Persist dialogue state to Redis."""
        key = self._dialogue_state_key(state.pipeline_id)
        await self.redis.set(key, state.model_dump_json())

    async def load_dialogue_state(self, pipeline_id: str) -> Optional[DialogueState]:
        """Load dialogue state from Redis."""
        key = self._dialogue_state_key(pipeline_id)
        state_json = await self.redis.get(key)

        if state_json is None:
            return None

        return DialogueState.model_validate_json(state_json)

    async def delete_dialogue_state(self, pipeline_id: str) -> None:
        """Delete dialogue state from Redis."""
        state_key = self._dialogue_state_key(pipeline_id)
        dag_key = self._dag_key(pipeline_id)
        errors_key = self._errors_key(pipeline_id)

        await self.redis.delete(state_key, dag_key, errors_key)

    async def init_dag(self, pipeline_id: str, scenario: Scenario) -> None:
        """Initialize DAG status for all steps as pending."""
        dag_key = self._dag_key(pipeline_id)

        # Build mapping of step_id -> "pending"
        mapping = {step_id: StepStatus.PENDING.value for step_id in scenario.steps.keys()}

        # Store as Redis HASH
        if mapping:
            await self.redis.hset(dag_key, mapping=mapping)

    async def get_step_status(self, pipeline_id: str, step_id: str) -> StepStatus:
        """Get the current status of a step."""
        dag_key = self._dag_key(pipeline_id)
        status_str = await self.redis.hget(dag_key, step_id)

        if status_str is None:
            # Default to PENDING if not found
            return StepStatus.PENDING

        # Decode bytes to string if necessary
        if isinstance(status_str, bytes):
            status_str = status_str.decode('utf-8')

        return StepStatus(status_str)

    async def set_step_status(
        self, pipeline_id: str, step_id: str, status: StepStatus
    ) -> None:
        """Update the status of a step."""
        dag_key = self._dag_key(pipeline_id)
        await self.redis.hset(dag_key, step_id, status.value)

    async def acquire_lock(
        self, pipeline_id: str, step_id: str, ttl: int = 300
    ) -> bool:
        """Acquire distributed lock for step execution."""
        lock_key = self._lock_key(pipeline_id, step_id)

        # SET with NX (only if not exists) and EX (expiry in seconds)
        # Returns True if lock acquired, False if already held
        result = await self.redis.set(
            lock_key,
            "locked",
            nx=True,  # Only set if key doesn't exist
            ex=ttl    # Expire after ttl seconds
        )

        return result is not None

    async def release_lock(self, pipeline_id: str, step_id: str) -> None:
        """Release distributed lock for step execution."""
        lock_key = self._lock_key(pipeline_id, step_id)
        await self.redis.delete(lock_key)

    async def set_step_error(
        self, pipeline_id: str, step_id: str, error: str
    ) -> None:
        """Store an error message for a failed step."""
        errors_key = self._errors_key(pipeline_id)
        await self.redis.hset(errors_key, step_id, error)

    async def get_step_errors(self, pipeline_id: str) -> dict[str, str]:
        """Get all step errors for a pipeline.

        Returns:
            Dict mapping step_id -> error message
        """
        errors_key = self._errors_key(pipeline_id)
        errors = await self.redis.hgetall(errors_key)

        # Decode bytes to strings
        return {
            (k.decode('utf-8') if isinstance(k, bytes) else k):
            (v.decode('utf-8') if isinstance(v, bytes) else v)
            for k, v in errors.items()
        }
