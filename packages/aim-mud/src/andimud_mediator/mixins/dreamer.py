# andimud_mediator/mixins/dreamer.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Dreamer mixin for the mediator service."""

import logging
from typing import Optional

from aim_mud_types import MUDEvent, EventType, TurnRequestStatus, TurnReason, normalize_agent_id

from ..patterns import DREAMER_PATTERN

logger = logging.getLogger(__name__)


class DreamerMixin:
    """Dreamer mixin for the mediator service."""


    async def _handle_dreamer_command(
        self,
        agent_id: str,
        enabled: bool,
    ) -> bool:
        """Handle @dreamer command by toggling automatic dreaming.

        Updates the agent:{id}:dreamer hash with enabled state.
        If enabling for the first time, initializes default thresholds.

        Args:
            agent_id: Target agent ID.
            enabled: True to enable, False to disable.

        Returns:
            True if state was updated successfully.
        """
        # Check agent is registered
        if self.registered_agents and agent_id not in self.registered_agents:
            logger.warning(f"@dreamer: Agent '{agent_id}' not registered")
            return False

        # Use RedisMUDClient for updates
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)

        # Get current state to check if first enable
        current = await client.get_dreamer_state(agent_id)

        # Build fields to update
        fields = {"enabled": enabled}

        # If enabling and no existing thresholds, set defaults
        if enabled and not current:
            fields["idle_threshold_seconds"] = 3600
            fields["token_threshold"] = 10000

        await client.update_dreamer_state_fields(agent_id, **fields)

        logger.info(f"@dreamer: Set {agent_id} dreamer enabled={enabled}")
        return True

    async def _try_handle_control_command(self, event: MUDEvent) -> bool:
        """Check if event is a control command and handle it.

        Parses event content for dream command patterns. Only @dreamer is handled
        by mediator now - all dream commands are handled by worker.

        Args:
            event: The MUDEvent to check.

        Returns:
            True if event was a control command (handled), False otherwise.
        """
        # Only check SYSTEM events for commands
        if event.event_type != EventType.SYSTEM:
            return False

        content = event.content.strip() if event.content else ""
        if not content:
            return False

        # Try @dreamer command (only command mediator handles)
        match = DREAMER_PATTERN.match(content)
        if match:
            agent_id = normalize_agent_id(match.group(1))
            enabled = match.group(2).lower() == "on"
            await self._handle_dreamer_command(agent_id, enabled)
            return True

        return False
