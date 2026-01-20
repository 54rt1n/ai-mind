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

from aim_mud_types.helper import _datetime_to_unix, model_to_redis_hash

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
        from aim_mud_types.coordination import DreamingState
        from aim_mud_types.redis_keys import RedisKeys

        key = RedisKeys.agent_dreaming_state(agent_id)
        data = await self.redis.hgetall(key)

        if not data:
            return None

        # Decode bytes to strings
        decoded: dict = {}
        for k, v in data.items():
            k_str = k.decode("utf-8") if isinstance(k, bytes) else str(k)
            v_str = v.decode("utf-8") if isinstance(v, bytes) else str(v)
            decoded[k_str] = v_str

        try:
            # Debug: log what we're loading
            framework_val = decoded.get("framework", "")
            state_val = decoded.get("state", "")
            logger.debug(
                f"Loading DreamingState: status={decoded.get('status')}, "
                f"framework_len={len(framework_val)}, state_len={len(state_val)}"
            )

            # Use Pydantic's model_validate (field validators handle datetime and JSON parsing)
            return DreamingState.model_validate(decoded)

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

        # Use helper to convert model to Redis hash format
        data = model_to_redis_hash(state)

        # Debug: verify framework/state are being saved
        framework_len = len(data.get("framework", ""))
        state_len = len(data.get("state", ""))
        logger.debug(
            f"Saving DreamingState {state.pipeline_id}: status={data.get('status')}, "
            f"framework_len={framework_len}, state_len={state_len}"
        )

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
        now = _datetime_to_unix(datetime.now(timezone.utc))

        # Update both heartbeat_at and updated_at (Unix timestamps)
        await self.redis.hset(
            key,
            mapping={
                "heartbeat_at": str(now),
                "updated_at": str(now),
            },
        )

        logger.debug(f"Updated dreaming heartbeat for {agent_id}")
