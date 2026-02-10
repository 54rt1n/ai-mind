# aim/app/mud/worker/actions.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Action emission for the MUD worker.

Handles emitting actions to the Redis action stream.
Extracted from worker.py lines 1230-1269
"""

import json
import logging
import secrets
import time
from typing import TYPE_CHECKING

from aim_mud_types import MUDAction

if TYPE_CHECKING:
    from ...worker import MUDAgentWorker


logger = logging.getLogger(__name__)


def generate_action_id() -> str:
    """Generate a unique action_id for correlating actions with their echo events.

    Format: act_{timestamp_ms}_{random_hex}
    - timestamp_ms: milliseconds since epoch for ordering/debugging
    - random_hex: 4 bytes of randomness for uniqueness

    Returns:
        Unique action identifier string.
    """
    timestamp_ms = int(time.time() * 1000)
    random_suffix = secrets.token_hex(4)
    return f"act_{timestamp_ms}_{random_suffix}"


class ActionsMixin:
    """Mixin for action emission methods.

    These methods are mixed into MUDAgentWorker in main.py.

    Attributes:
        _last_emitted_action_ids: List of action_ids from the most recent _emit_actions() call.
            Used by commands that call take_turn() to retrieve action_ids for pending tracking.
        _last_emitted_expects_echo: Whether any action in the last emit expects an echo.
    """

    # Class-level attribute declaration for type hints
    _last_emitted_action_ids: list[str]
    _last_emitted_expects_echo: bool

    async def _emit_actions(
        self: "MUDAgentWorker", actions: list[MUDAction]
    ) -> tuple[list[str], bool]:
        """Emit actions to the Redis mud:actions stream.

        Originally from worker.py lines 1230-1269

        Generates a unique action_id for each action to enable correlation
        between emitted actions and their echo events from Evennia.

        Args:
            actions: List of MUDAction objects to emit.

        Returns:
            Tuple of (action_ids, expects_echo):
            - action_ids: List of action_ids for successfully emitted actions
            - expects_echo: True if any action expects an echo event (not non_published)
        """
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)
        emitted_action_ids: list[str] = []

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

                # Generate action_id for correlation with echo events
                action_id = generate_action_id()

                data = action.to_redis_dict(self.config.agent_id)
                data["action_id"] = action_id

                stream_msg_id = await client.append_mud_action(
                    {"data": json.dumps(data)},
                    stream_key=self.config.action_stream,
                )
                await self._update_agent_profile(last_action_id=str(stream_msg_id))
                emitted_action_ids.append(action_id)
                logger.info(
                    f"Emitted action: {action.tool} -> {command} (action_id={action_id})"
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

        # Check if any action expects an echo (not non_published)
        expects_echo = any(action.expects_echo() for action in actions)

        # Store for commands that call take_turn() and need action_ids
        self._last_emitted_action_ids = emitted_action_ids
        self._last_emitted_expects_echo = expects_echo

        # If no actions expect echo (all non-published), transition turn to READY immediately
        if emitted_action_ids and not expects_echo:
            from aim_mud_types import TurnRequestStatus
            from aim_mud_types.client import AsyncRedisMUDClient

            current = await self._get_turn_request()
            if current and current.status not in (TurnRequestStatus.DONE, TurnRequestStatus.ABORTED):
                client = AsyncRedisMUDClient(self.redis)
                await client.transition_turn_request_to_ready(
                    self.config.agent_id,
                    current,
                    expected_turn_id=current.turn_id,
                    status_reason="Non-published actions emitted (no echo expected)",
                    new_turn_id=True,
                    update_heartbeat=True,
                )
                logger.info("Turn set to READY (non-published actions only)")

        return emitted_action_ids, expects_echo
