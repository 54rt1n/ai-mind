# andimud_mediator/mixins/dreamer.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Dreamer mixin for the mediator service."""

import logging
from typing import Optional

from aim_mud_types import MUDEvent, EventType, TurnRequestStatus, TurnReason

from ..patterns import (
    DREAMER_PATTERN, ANALYZE_PATTERN, SUMMARY_PATTERN, JOURNAL_PATTERN,
    PONDER_PATTERN, DAYDREAM_PATTERN, CRITIQUE_PATTERN, RESEARCH_PATTERN,
    COMMAND_TO_SCENARIO, normalize_agent_id,
)

logger = logging.getLogger(__name__)


class DreamerMixin:
    """Dreamer mixin for the mediator service."""

    async def _handle_analysis_command(
        self,
        agent_id: str,
        scenario: str,
        conversation_id: str,
        guidance: Optional[str] = None,
    ) -> bool:
        """Handle @analyze and @summary commands by assigning a dream turn.

        Creates a turn_request with reason="dream" and sets scenario,
        conversation_id, and guidance fields for the worker to process.

        Args:
            agent_id: Target agent ID.
            scenario: Name of scenario to run ("analysis_dialogue" or "summarizer").
            conversation_id: ID of conversation to analyze (required).
            guidance: Optional guidance text.

        Returns:
            True if turn was assigned, False if agent unavailable.
        """
        cmd_name = "analyze" if scenario == "analysis_dialogue" else "summary"

        # Check agent is registered
        if self.registered_agents and agent_id not in self.registered_agents:
            logger.warning(f"@{cmd_name}: Agent '{agent_id}' not registered")
            return False

        # Get current turn request state
        current = await self._get_turn_request(agent_id)
        if not current:
            logger.warning(f"@{cmd_name}: Agent '{agent_id}' offline (no turn_request)")
            return False

        status = current.status

        # Validate critical parameters
        if status is None:
            logger.error(f"@{cmd_name}: Agent '{agent_id}' has corrupted turn_request (status is None)")
            return False

        if not scenario:
            logger.error(f"@{cmd_name}: Missing required parameter 'scenario'")
            return False

        if not conversation_id:
            logger.error(f"@{cmd_name}: Missing required parameter 'conversation_id'")
            return False

        # Block if agent is busy, crashed, or retrying
        if status == TurnRequestStatus.CRASHED:
            logger.warning(f"@{cmd_name}: Agent '{agent_id}' is crashed")
            return False

        if status in (TurnRequestStatus.ASSIGNED, TurnRequestStatus.IN_PROGRESS,
                      TurnRequestStatus.ABORT_REQUESTED, TurnRequestStatus.EXECUTING,
                      TurnRequestStatus.RETRY):
            logger.warning(f"@{cmd_name}: Agent '{agent_id}' is busy (status={status.value})")
            return False

        # Build metadata dict (exclude None values)
        metadata = {"scenario": scenario, "conversation_id": conversation_id}
        if guidance:
            metadata["guidance"] = guidance

        from aim_mud_types.turn_request_helpers import assign_turn_request_async
        success, turn_request, result = await assign_turn_request_async(
            self.redis,
            agent_id,
            TurnReason.DREAM,
            attempt_count=0,
            deadline_ms=1800000,
            status=TurnRequestStatus.ASSIGNED,
            expected_turn_id=current.turn_id,
            skip_availability_check=True,
            **metadata,
        )

        if success and turn_request:
            logger.info(
                f"@{cmd_name}: Assigned dream turn to {agent_id} "
                f"(sequence_id={turn_request.sequence_id}, "
                f"scenario={scenario}, conversation_id={conversation_id}, "
                f"guidance={guidance or 'none'})"
            )
            return True
        else:
            logger.debug(
                f"@{cmd_name}: Assign failed for {agent_id}: {result}"
            )
            return False

    async def _handle_creative_command(
        self,
        agent_id: str,
        scenario: str,
        query: Optional[str] = None,
        guidance: Optional[str] = None,
    ) -> bool:
        """Handle creative commands (@journal, @ponder, @daydream, @critique, @research).

        Creates a turn_request with reason="dream" and sets scenario,
        query, and guidance fields for the worker to process.
        These commands create their own standalone conversations.

        Args:
            agent_id: Target agent ID.
            scenario: Name of scenario to run (e.g., "journaler_dialogue").
            query: Optional query text for the scenario.
            guidance: Optional guidance text.

        Returns:
            True if turn was assigned, False if agent unavailable.
        """
        # Reverse lookup command name for logging
        cmd_name = next(
            (k for k, v in COMMAND_TO_SCENARIO.items() if v == scenario),
            scenario,
        )

        # Check agent is registered
        if self.registered_agents and agent_id not in self.registered_agents:
            logger.warning(f"@{cmd_name}: Agent '{agent_id}' not registered")
            return False

        # Get current turn request state
        current = await self._get_turn_request(agent_id)
        if not current:
            logger.warning(f"@{cmd_name}: Agent '{agent_id}' offline (no turn_request)")
            return False

        status = current.status

        # Validate critical parameters
        if status is None:
            logger.error(f"@{cmd_name}: Agent '{agent_id}' has corrupted turn_request (status is None)")
            return False

        if not scenario:
            logger.error(f"@{cmd_name}: Missing required parameter 'scenario'")
            return False

        # Block if agent is busy, crashed, or retrying
        if status == TurnRequestStatus.CRASHED:
            logger.warning(f"@{cmd_name}: Agent '{agent_id}' is crashed")
            return False

        if status in (TurnRequestStatus.ASSIGNED, TurnRequestStatus.IN_PROGRESS,
                      TurnRequestStatus.ABORT_REQUESTED, TurnRequestStatus.EXECUTING,
                      TurnRequestStatus.RETRY):
            logger.warning(f"@{cmd_name}: Agent '{agent_id}' is busy (status={status.value})")
            return False

        # Build metadata dict (exclude None values)
        metadata = {"scenario": scenario}
        if query:
            metadata["query"] = query
        if guidance:
            metadata["guidance"] = guidance

        from aim_mud_types.turn_request_helpers import assign_turn_request_async
        success, turn_request, result = await assign_turn_request_async(
            self.redis,
            agent_id,
            TurnReason.DREAM,
            attempt_count=0,
            deadline_ms=1800000,
            status=TurnRequestStatus.ASSIGNED,
            expected_turn_id=current.turn_id,
            skip_availability_check=True,
            **metadata,
        )

        if success and turn_request:
            logger.info(
                f"@{cmd_name}: Assigned dream turn to {agent_id} "
                f"(sequence_id={turn_request.sequence_id}, "
                f"scenario={scenario}, query={query or 'none'}, "
                f"guidance={guidance or 'none'})"
            )
            return True
        else:
            logger.debug(
                f"@{cmd_name}: Assign failed for {agent_id}: {result}"
            )
            return False

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

        Parses event content for dream command patterns (@analyze, @summary,
        @journal, @ponder, @daydream, @critique, @research) and @dreamer.
        If matched, executes the command and returns True.

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

        # Try @analyze command (requires conversation_id)
        match = ANALYZE_PATTERN.match(content)
        if match:
            agent_id = normalize_agent_id(match.group(1))
            conversation_id = match.group(2).strip()
            guidance = match.group(3).strip() if match.group(3) else None
            await self._handle_analysis_command(
                agent_id, "analysis_dialogue", conversation_id, guidance
            )
            return True

        # Try @summary command (requires conversation_id)
        match = SUMMARY_PATTERN.match(content)
        if match:
            agent_id = normalize_agent_id(match.group(1))
            conversation_id = match.group(2).strip()
            await self._handle_analysis_command(
                agent_id, "summarizer", conversation_id, None
            )
            return True

        # Try creative commands (@journal, @ponder, @daydream, @critique, @research)
        for pattern, cmd_name in [
            (JOURNAL_PATTERN, "journal"),
            (PONDER_PATTERN, "ponder"),
            (DAYDREAM_PATTERN, "daydream"),
            (CRITIQUE_PATTERN, "critique"),
            (RESEARCH_PATTERN, "research"),
        ]:
            match = pattern.match(content)
            if match:
                agent_id = normalize_agent_id(match.group(1))
                query = match.group(2).strip() if match.group(2) else None
                guidance = match.group(3).strip() if match.group(3) else None
                await self._handle_creative_command(
                    agent_id,
                    COMMAND_TO_SCENARIO[cmd_name],
                    query,
                    guidance,
                )
                return True

        # Try @dreamer command
        match = DREAMER_PATTERN.match(content)
        if match:
            agent_id = normalize_agent_id(match.group(1))
            enabled = match.group(2).lower() == "on"
            await self._handle_dreamer_command(agent_id, enabled)
            return True

        return False
