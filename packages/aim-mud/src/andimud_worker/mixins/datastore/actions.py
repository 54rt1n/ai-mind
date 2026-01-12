# aim/app/mud/worker/actions.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Action emission for the MUD worker.

Handles emitting actions to the Redis action stream.
Extracted from worker.py lines 1230-1269
"""

import json
import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDAction

if TYPE_CHECKING:
    from ...worker import MUDAgentWorker


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
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)
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
                action_id = await client.append_mud_action(
                    {"data": json.dumps(data)},
                    stream_key=self.config.action_stream,
                )
                await self._update_agent_profile(last_action_id=str(action_id))
                logger.info(
                    f"Emitted action: {action.tool} -> {command}"
                )
            except Exception as e:
                logger.error(f"Failed to emit action {action.tool}: {e}")

        # Trim old actions from stream (keep last 1000)
        try:
            await client.trim_mud_actions_maxlen(
                maxlen=1000,
                approximate=True,
                stream_key=self.config.action_stream,
            )
        except Exception as e:
            logger.warning(f"Failed to trim action stream: {e}")
