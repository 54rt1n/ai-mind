# aim/app/mud/worker/actions.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Action emission for the MUD worker.

Handles emitting actions to the Redis action stream.
Extracted from worker.py lines 1230-1269
"""

import json
import logging
from typing import TYPE_CHECKING

import redis.asyncio as redis

from aim_mud_types import MUDAction
from .utils import _utc_now


if TYPE_CHECKING:
    from .main import MUDAgentWorker


logger = logging.getLogger(__name__)


class ActionsMixin:
    """Mixin for action emission methods.

    These methods are mixed into MUDAgentWorker in main.py.
    """

    async def _emit_actions(self: "MUDAgentWorker", actions: list[MUDAction]) -> None:
        """Emit actions to the Redis mud:actions stream.

        Originally from worker.py lines 1230-1269

        Args:
            actions: List of MUDAction objects to emit.
        """
        for action in actions:
            try:
                command = action.to_command().strip()
                if not command:
                    logger.warning(
                        "Skipping action with empty command: %s(%s)",
                        action.tool,
                        action.args,
                    )
                    continue
                data = action.to_redis_dict(self.config.agent_id)
                action_id = await self.redis.xadd(
                    self.config.action_stream,
                    {"data": json.dumps(data)},
                )
                if isinstance(action_id, bytes):
                    action_id = action_id.decode("utf-8")
                await self._update_agent_profile(last_action_id=str(action_id))
                logger.info(
                    f"Emitted action: {action.tool} -> {command}"
                )
            except redis.RedisError as e:
                logger.error(f"Failed to emit action {action.tool}: {e}")

        # Trim old actions from stream (keep last 1000)
        try:
            await self.redis.xtrim(
                self.config.action_stream,
                maxlen=1000,
                approximate=True,
            )
        except redis.RedisError as e:
            logger.warning(f"Failed to trim action stream: {e}")
