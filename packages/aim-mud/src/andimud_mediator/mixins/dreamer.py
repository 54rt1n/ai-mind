# andimud_mediator/mixins/dreamer.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Dreamer mixin for the mediator service."""

import logging
from typing import Optional
import uuid

from aim_mud_types import MUDEvent, EventType, RedisKeys
from aim_mud_types.helper import _utc_now

from ..patterns import (
    DREAMER_PATTERN, ANALYZE_PATTERN, SUMMARY_PATTERN, JOURNAL_PATTERN,
    PONDER_PATTERN, DAYDREAM_PATTERN, CRITIQUE_PATTERN, RESEARCH_PATTERN,
    COMMAND_TO_SCENARIO,
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

        status = current.get("status")

        # Block if agent is busy or crashed
        if status == "crashed":
            logger.warning(f"@{cmd_name}: Agent '{agent_id}' is crashed")
            return False

        if status in ("assigned", "in_progress", "abort_requested"):
            logger.warning(f"@{cmd_name}: Agent '{agent_id}' is busy (status={status})")
            return False

        # Create dream turn assignment
        turn_id = str(uuid.uuid4())
        assigned_at = _utc_now().isoformat()
        current_turn_id = current.get("turn_id", "")
        # Dreams get 10 minutes (600000ms) - they're slow pipeline operations
        deadline_ms = "600000"

        # Use extended Lua script to set dream-specific fields
        lua_script = """
            local key = KEYS[1]
            local expected_turn_id = ARGV[1]
            local expected_status = ARGV[2]

            -- Check current state matches expectations
            local current_turn_id = redis.call('HGET', key, 'turn_id')
            local current_status = redis.call('HGET', key, 'status')

            if current_turn_id ~= expected_turn_id or current_status ~= expected_status then
                return 0  -- State changed, CAS failed
            end

            -- Atomically assign dream turn
            redis.call('HSET', key, 'turn_id', ARGV[3])
            redis.call('HSET', key, 'status', 'assigned')
            redis.call('HSET', key, 'reason', 'dream')
            redis.call('HSET', key, 'assigned_at', ARGV[4])
            redis.call('HSET', key, 'heartbeat_at', ARGV[4])
            redis.call('HSET', key, 'deadline_ms', ARGV[5])
            redis.call('HSET', key, 'attempt_count', '0')
            redis.call('HSET', key, 'scenario', ARGV[6])
            redis.call('HSET', key, 'conversation_id', ARGV[7])
            redis.call('HSET', key, 'guidance', ARGV[8])

            return 1  -- Success
        """

        result = await self.redis.eval(
            lua_script,
            1,  # number of keys
            RedisKeys.agent_turn_request(agent_id),
            current_turn_id or "",
            status,
            turn_id,
            assigned_at,
            deadline_ms,
            scenario,
            conversation_id,
            guidance or "",
        )

        if result == 1:
            # CAS succeeded - set TTL (10 minutes for dreams)
            await self.redis.expire(
                RedisKeys.agent_turn_request(agent_id),
                600,  # 10 minutes
            )
            logger.info(
                f"@{cmd_name}: Assigned dream turn to {agent_id} "
                f"(scenario={scenario}, conversation_id={conversation_id}, "
                f"guidance={guidance or 'none'})"
            )
            return True
        else:
            logger.debug(
                f"@{cmd_name}: CAS failed for {agent_id}, state changed during assignment"
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

        status = current.get("status")

        # Block if agent is busy or crashed
        if status == "crashed":
            logger.warning(f"@{cmd_name}: Agent '{agent_id}' is crashed")
            return False

        if status in ("assigned", "in_progress", "abort_requested"):
            logger.warning(f"@{cmd_name}: Agent '{agent_id}' is busy (status={status})")
            return False

        # Create dream turn assignment
        turn_id = str(uuid.uuid4())
        assigned_at = _utc_now().isoformat()
        current_turn_id = current.get("turn_id", "")
        # Dreams get 10 minutes (600000ms) - they're slow pipeline operations
        deadline_ms = "600000"

        # Use extended Lua script to set dream-specific fields
        lua_script = """
            local key = KEYS[1]
            local expected_turn_id = ARGV[1]
            local expected_status = ARGV[2]

            -- Check current state matches expectations
            local current_turn_id = redis.call('HGET', key, 'turn_id')
            local current_status = redis.call('HGET', key, 'status')

            if current_turn_id ~= expected_turn_id or current_status ~= expected_status then
                return 0  -- State changed, CAS failed
            end

            -- Atomically assign dream turn
            redis.call('HSET', key, 'turn_id', ARGV[3])
            redis.call('HSET', key, 'status', 'assigned')
            redis.call('HSET', key, 'reason', 'dream')
            redis.call('HSET', key, 'assigned_at', ARGV[4])
            redis.call('HSET', key, 'heartbeat_at', ARGV[4])
            redis.call('HSET', key, 'deadline_ms', ARGV[5])
            redis.call('HSET', key, 'attempt_count', '0')
            redis.call('HSET', key, 'scenario', ARGV[6])
            redis.call('HSET', key, 'query', ARGV[7])
            redis.call('HSET', key, 'guidance', ARGV[8])

            return 1  -- Success
        """

        result = await self.redis.eval(
            lua_script,
            1,  # number of keys
            RedisKeys.agent_turn_request(agent_id),
            current_turn_id or "",
            status,
            turn_id,
            assigned_at,
            deadline_ms,
            scenario,
            query or "",
            guidance or "",
        )

        if result == 1:
            # CAS succeeded - set TTL (10 minutes for dreams)
            await self.redis.expire(
                RedisKeys.agent_turn_request(agent_id),
                600,  # 10 minutes
            )
            logger.info(
                f"@{cmd_name}: Assigned dream turn to {agent_id} "
                f"(scenario={scenario}, query={query or 'none'}, "
                f"guidance={guidance or 'none'})"
            )
            return True
        else:
            logger.debug(
                f"@{cmd_name}: CAS failed for {agent_id}, state changed during assignment"
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

        key = RedisKeys.agent_dreamer(agent_id)

        # Get current state to check if first enable
        current = await self.redis.hgetall(key)

        mapping: dict[str, str] = {
            "enabled": "true" if enabled else "false",
        }

        # If enabling and no existing thresholds, set defaults
        if enabled and not current:
            mapping["idle_threshold_seconds"] = "3600"
            mapping["token_threshold"] = "10000"

        await self.redis.hset(key, mapping=mapping)

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
            agent_id = match.group(1).lower()
            conversation_id = match.group(2).strip()
            guidance = match.group(3).strip() if match.group(3) else None
            await self._handle_analysis_command(
                agent_id, "analysis_dialogue", conversation_id, guidance
            )
            return True

        # Try @summary command (requires conversation_id)
        match = SUMMARY_PATTERN.match(content)
        if match:
            agent_id = match.group(1).lower()
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
                agent_id = match.group(1).lower()
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
            agent_id = match.group(1).lower()
            enabled = match.group(2).lower() == "on"
            await self._handle_dreamer_command(agent_id, enabled)
            return True

        return False
