# aim/app/mud/worker/mixins/datastore/state.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Worker state checking methods.

Handles pause state, abort requests, and spontaneous action triggers.
Extracted from worker.py lines 537-809
"""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import TurnRequestStatus
from aim_mud_types.helper import _utc_now

if TYPE_CHECKING:
    from ...worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class StateMixin:
    """Mixin for worker state checking methods.

    These methods are mixed into MUDAgentWorker in worker.py.
    """

    async def _is_paused(self: "MUDAgentWorker") -> bool:
        """Check if worker is paused via Redis flag.

        Originally from worker.py lines 537-546

        Returns:
            bool: True if paused, False if running.
        """
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)
        return await client.is_paused(self.config.pause_key)

    async def _check_abort_requested(self: "MUDAgentWorker") -> bool:
        """Check if current turn has abort requested.

        Originally from worker.py lines 637-655

        Returns:
            bool: True if abort requested, False otherwise.
        """
        turn_request = await self._get_turn_request()

        if turn_request is None:
            return False

        if turn_request.status == TurnRequestStatus.ABORT_REQUESTED:
            from aim_mud_types.turn_request_helpers import (
                transition_turn_request_and_update_async,
            )
            await transition_turn_request_and_update_async(
                self.redis,
                self.config.agent_id,
                turn_request,
                expected_turn_id=turn_request.turn_id,
                status=TurnRequestStatus.ABORTED,
                message="Aborted by user request",
                update_heartbeat=True,
            )
            return True
        return False

    def _should_act_spontaneously(self: "MUDAgentWorker") -> bool:
        """Determine if agent should act without new events.

        Originally from worker.py lines 659-685

        Checks time since last action against the spontaneous action interval.

        Returns:
            bool: True if spontaneous action should be triggered.
        """
        if self.session is None:
            return False

        if self.session.last_event_time is None:
            # No events yet - don't trigger spontaneous action
            return False

        now = _utc_now()
        elapsed_since_event = (now - self.session.last_event_time).total_seconds()
        if elapsed_since_event < self.config.spontaneous_action_interval:
            return False

        if self.session.last_action_time is None:
            return True

        elapsed_since_action = (now - self.session.last_action_time).total_seconds()
        return elapsed_since_action >= self.config.spontaneous_action_interval
