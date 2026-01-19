# andimud_worker/mixins/dreaming_datastore.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Dreaming datastore mixin for MUD worker.

Adds step-by-step dream state persistence to MUDAgentWorker:
- load_dreaming_state(): Load active dream state from Redis
- save_dreaming_state(): Save dream state to Redis
- delete_dreaming_state(): Delete dream state from Redis
- archive_dreaming_state(): Archive completed dream to history
- update_dreaming_heartbeat(): Update heartbeat timestamp
"""

from typing import TYPE_CHECKING, Optional
from datetime import datetime, timezone
import json
import logging

if TYPE_CHECKING:
    from aim_mud_types.coordination import DreamingState
    from ..worker import MUDAgentWorker

logger = logging.getLogger(__name__)


class DreamingDatastoreMixin:
    """Mixin adding dreaming state persistence to MUDAgentWorker.

    Manages serialization/deserialization of DreamingState to/from Redis.

    Expected attributes from MUDAgentWorker:
    - self.redis: Async Redis client
    - self.config: MUDConfig (has agent_id)
    """

    async def load_dreaming_state(
        self: "MUDAgentWorker",
        agent_id: str,
    ) -> Optional["DreamingState"]:
        """Load active dream state from Redis.

        Args:
            agent_id: Agent identifier.

        Returns:
            DreamingState if exists, None otherwise.
        """
        from aim_mud_types.coordination import DreamingState, DreamStatus
        from aim_mud_types.redis_keys import RedisKeys

        key = RedisKeys.agent_dreaming_state(agent_id)
        data = await self.redis.hgetall(key)

        if not data:
            return None

        # Deserialize from Redis hash (all values are strings)
        try:
            # Parse datetime fields
            created_at = datetime.fromisoformat(data["created_at"])
            updated_at = datetime.fromisoformat(data["updated_at"])
            completed_at = (
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            )
            next_retry_at = (
                datetime.fromisoformat(data["next_retry_at"])
                if data.get("next_retry_at")
                else None
            )
            heartbeat_at = (
                datetime.fromisoformat(data["heartbeat_at"])
                if data.get("heartbeat_at")
                else None
            )

            # Parse JSON fields
            execution_order = json.loads(data["execution_order"])
            completed_steps = json.loads(data["completed_steps"])
            step_doc_ids = json.loads(data["step_doc_ids"])
            context_doc_ids = json.loads(data["context_doc_ids"])
            scenario_config = json.loads(data["scenario_config"])
            persona_config = json.loads(data["persona_config"])

            # Construct DreamingState
            state = DreamingState(
                pipeline_id=data["pipeline_id"],
                agent_id=data["agent_id"],
                status=DreamStatus(data["status"]),
                created_at=created_at,
                updated_at=updated_at,
                completed_at=completed_at,
                scenario_name=data["scenario_name"],
                execution_order=execution_order,
                query=data.get("query"),
                guidance=data.get("guidance"),
                conversation_id=data["conversation_id"],
                base_model=data["base_model"],
                step_index=int(data["step_index"]),
                completed_steps=completed_steps,
                step_doc_ids=step_doc_ids,
                context_doc_ids=context_doc_ids,
                current_step_attempts=int(data["current_step_attempts"]),
                max_step_retries=int(data["max_step_retries"]),
                next_retry_at=next_retry_at,
                last_error=data.get("last_error"),
                heartbeat_at=heartbeat_at,
                heartbeat_timeout_seconds=int(data["heartbeat_timeout_seconds"]),
                scenario_config=scenario_config,
                persona_config=persona_config,
            )

            return state

        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.error(
                f"Failed to deserialize DreamingState for {agent_id}: {e}",
                exc_info=True,
            )
            return None

    async def save_dreaming_state(
        self: "MUDAgentWorker",
        state: "DreamingState",
    ) -> None:
        """Save dream state to Redis.

        Args:
            state: DreamingState to persist.
        """
        from aim_mud_types.redis_keys import RedisKeys

        key = RedisKeys.agent_dreaming_state(state.agent_id)

        # Serialize to Redis hash (all values must be strings)
        data = {
            "pipeline_id": state.pipeline_id,
            "agent_id": state.agent_id,
            "status": state.status.value,
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat(),
            "completed_at": state.completed_at.isoformat() if state.completed_at else "",
            "scenario_name": state.scenario_name,
            "execution_order": json.dumps(state.execution_order),
            "query": state.query or "",
            "guidance": state.guidance or "",
            "conversation_id": state.conversation_id,
            "base_model": state.base_model,
            "step_index": str(state.step_index),
            "completed_steps": json.dumps(state.completed_steps),
            "step_doc_ids": json.dumps(state.step_doc_ids),
            "context_doc_ids": json.dumps(state.context_doc_ids),
            "current_step_attempts": str(state.current_step_attempts),
            "max_step_retries": str(state.max_step_retries),
            "next_retry_at": state.next_retry_at.isoformat() if state.next_retry_at else "",
            "last_error": state.last_error or "",
            "heartbeat_at": state.heartbeat_at.isoformat() if state.heartbeat_at else "",
            "heartbeat_timeout_seconds": str(state.heartbeat_timeout_seconds),
            "scenario_config": json.dumps(state.scenario_config),
            "persona_config": json.dumps(state.persona_config),
        }

        await self.redis.hset(key, mapping=data)

        logger.debug(
            f"Saved DreamingState {state.pipeline_id} for {state.agent_id} "
            f"(step {state.step_index + 1}/{len(state.execution_order)})"
        )

    async def delete_dreaming_state(
        self: "MUDAgentWorker",
        agent_id: str,
    ) -> None:
        """Delete dream state from Redis.

        Args:
            agent_id: Agent identifier.
        """
        from aim_mud_types.redis_keys import RedisKeys

        key = RedisKeys.agent_dreaming_state(agent_id)
        await self.redis.delete(key)

        logger.debug(f"Deleted DreamingState for {agent_id}")

    async def archive_dreaming_state(
        self: "MUDAgentWorker",
        state: "DreamingState",
    ) -> None:
        """Archive completed dream to history.

        Stores the final state in a Redis list for debugging/analytics.
        Keeps last 100 entries (FIFO).

        Args:
            state: Completed DreamingState to archive.
        """
        from aim_mud_types.redis_keys import RedisKeys

        key = RedisKeys.agent_dreaming_history(state.agent_id)

        # Serialize as JSON for list storage
        archive_data = state.model_dump_json()

        # Push to list and trim to 100 entries
        await self.redis.lpush(key, archive_data)
        await self.redis.ltrim(key, 0, 99)

        logger.info(
            f"Archived DreamingState {state.pipeline_id} for {state.agent_id} "
            f"(status: {state.status.value})"
        )

    async def update_dreaming_heartbeat(
        self: "MUDAgentWorker",
        agent_id: str,
    ) -> None:
        """Update heartbeat timestamp for active dream.

        Args:
            agent_id: Agent identifier.
        """
        from aim_mud_types.redis_keys import RedisKeys

        key = RedisKeys.agent_dreaming_state(agent_id)
        now = datetime.now(timezone.utc)

        # Update both heartbeat_at and updated_at
        await self.redis.hset(
            key,
            mapping={
                "heartbeat_at": now.isoformat(),
                "updated_at": now.isoformat(),
            },
        )

        logger.debug(f"Updated dreaming heartbeat for {agent_id}")
