# andimud_mediator/mixins/agents.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Agents mixin for the mediator service."""

import json
import logging
from datetime import datetime, timezone
from typing import Optional
import uuid

from aim_mud_types import RedisKeys, MUDTurnRequest, TurnRequestStatus, TurnReason
from aim_mud_types.helper import _utc_now

logger = logging.getLogger(__name__)


class AgentsMixin:
    """Agents mixin for the mediator service."""

    async def _check_agent_states(self) -> None:
        """Check agent states when no events arrive (XREAD timeout).

        Checks for:
        1. Failed turns past retry time -> trigger retry
        2. Stale heartbeats (>5min) -> mark as crashed
        """
        for agent_id in self.registered_agents:
            try:
                turn_request = await self._get_turn_request(agent_id)
                if not turn_request:
                    continue

                status = turn_request.status

                # Case 1: Failed turn ready to retry
                if status == TurnRequestStatus.FAIL:
                    if turn_request.next_attempt_at:
                        next_attempt_at = datetime.fromisoformat(turn_request.next_attempt_at)
                        if datetime.now(timezone.utc) >= next_attempt_at:
                            logger.info(f"Retrying failed turn for {agent_id}")
                            await self._maybe_assign_turn(agent_id, reason=TurnReason.RETRY)

                # Case 2: Stale heartbeat (worker crashed)
                elif status == TurnRequestStatus.IN_PROGRESS:
                    if turn_request.heartbeat_at:
                        stale_seconds = (datetime.now(timezone.utc) - turn_request.heartbeat_at).total_seconds()
                        if stale_seconds > 300:  # 5 minutes
                            stale_duration = f"{int(stale_seconds // 60)}m{int(stale_seconds % 60)}s"

                            # Mark as crashed - use Lua script to prevent partial hash creation
                            lua_script = """
                                local key = KEYS[1]
                                local status_value = ARGV[1]
                                local status_reason = ARGV[2]

                                -- Only mark crashed if turn_request exists
                                if redis.call('EXISTS', key) == 1 then
                                    redis.call('HSET', key, 'status', status_value)
                                    redis.call('HSET', key, 'status_reason', status_reason)
                                    return 1
                                else
                                    return 0
                                end
                            """

                            result = await self.redis.eval(
                                lua_script,
                                1,  # number of keys
                                RedisKeys.agent_turn_request(agent_id),
                                TurnRequestStatus.CRASHED.value,
                                f"Heartbeat stale for {stale_duration}"
                            )

                            if result == 1:
                                logger.error(
                                    f"Worker {agent_id} crashed (no heartbeat for {stale_seconds:.0f}s)"
                                )
                            else:
                                logger.debug(
                                    f"Agent '{agent_id}' turn_request missing, skipping crash detection"
                                )
            except Exception as e:
                logger.error(f"Error checking state for {agent_id}: {e}", exc_info=True)
                continue

        # Check for auto-analysis trigger
        await self._check_auto_analysis_trigger()

    async def _maybe_assign_turn(self, agent_id: str, reason: "str | TurnReason" = TurnReason.EVENTS) -> bool:
        """Assign turn if agent is available (including ready to retry).

        Checks agent availability:
        - Worker offline: no turn_request hash exists
        - Worker paused: mud:agent:{id}:paused key is set to "1"
        - Worker crashed: turn_request status is "crashed"
        - Worker busy: turn_request status is "assigned", "in_progress", or "abort_requested"
        - Failed turn in backoff: status is "fail" and current time < next_attempt_at

        Returns:
            True if turn was assigned, False if agent is busy/offline/crashed/paused.
        """
        current = await self._get_turn_request(agent_id)

        # No turn_request = worker offline
        if not current:
            logger.debug(f"Agent {agent_id} offline (no turn_request)")
            return False

        # Check if agent is paused via Redis key
        pause_key = RedisKeys.agent_pause(agent_id)
        is_paused = await self.redis.get(pause_key)
        if is_paused == b"1":
            logger.debug(f"Agent {agent_id} paused, not assigning turn")
            return False

        status = current.status

        # Block crashed workers
        if status == TurnRequestStatus.CRASHED:
            logger.debug(f"Agent {agent_id} crashed, not assigning")
            return False

        # Block if actively processing or being aborted
        if status in (TurnRequestStatus.ASSIGNED, TurnRequestStatus.IN_PROGRESS, TurnRequestStatus.ABORT_REQUESTED):
            return False

        # Check if failed turn ready to retry
        if status == TurnRequestStatus.FAIL:
            if current.next_attempt_at:
                next_attempt_at = datetime.fromisoformat(current.next_attempt_at)
                if datetime.now(timezone.utc) < next_attempt_at:
                    # Still in backoff period
                    return False
            # Else: no next_attempt_at (max attempts) or past retry time, fall through to assign

        # Available: status is "ready" or "fail" past retry time
        # Create new turn assignment
        turn_id = str(uuid.uuid4())
        turn_sequence_id = await self._next_sequence_id()
        assigned_at = _utc_now().isoformat()
        current_turn_id = current.turn_id

        # Preserve attempt_count if retrying a failed turn
        attempt_count = "0"
        if status == TurnRequestStatus.FAIL:
            attempt_count = str(current.attempt_count)

        # Convert reason to string value for Lua script
        reason_str = reason.value if isinstance(reason, TurnReason) else reason

        # Use CAS pattern to atomically assign if state hasn't changed
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

            -- Atomically assign turn
            redis.call('HSET', key, 'turn_id', ARGV[3])
            redis.call('HSET', key, 'status', ARGV[4])
            redis.call('HSET', key, 'reason', ARGV[5])
            redis.call('HSET', key, 'assigned_at', ARGV[6])
            redis.call('HSET', key, 'heartbeat_at', ARGV[6])
            redis.call('HSET', key, 'deadline_ms', ARGV[7])
            redis.call('HSET', key, 'attempt_count', ARGV[8])
            redis.call('HSET', key, 'sequence_id', ARGV[9])

            return 1  -- Success
        """

        result = await self.redis.eval(
            lua_script,
            1,  # number of keys
            RedisKeys.agent_turn_request(agent_id),
            current_turn_id or "",
            status.value if isinstance(status, TurnRequestStatus) else status,
            turn_id,
            TurnRequestStatus.ASSIGNED.value,
            reason_str,
            assigned_at,
            str(self.config.turn_request_ttl_seconds * 1000),
            attempt_count,
            str(turn_sequence_id)
        )

        if result == 1:
            # CAS succeeded - conditionally set TTL (0 = no TTL)
            if self.config.turn_request_ttl_seconds > 0:
                await self.redis.expire(
                    RedisKeys.agent_turn_request(agent_id),
                    self.config.turn_request_ttl_seconds
                )
            logger.info(f"Assigned turn to {agent_id} (sequence_id={turn_sequence_id}, reason={reason_str}, attempt={attempt_count})")
            return True
        else:
            # CAS failed - state changed between check and assign
            logger.debug(f"CAS failed for {agent_id}, state changed during assignment")
            return False

    async def _get_turn_request(self, agent_id: str) -> Optional[MUDTurnRequest]:
        """Fetch the current turn request hash for an agent."""
        return await MUDTurnRequest.from_redis(self.redis, agent_id)

    async def _agents_from_room_profile(self, room_id: str) -> list[str]:
        """Lookup agent_ids present in a room profile."""
        if not room_id:
            return []
        try:
            raw = await self.redis.hget(RedisKeys.room_profile(room_id), "entities_present")
        except Exception as e:
            logger.error(f"Failed to read room profile for {room_id}: {e}")
            return []
        if not raw:
            return []
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            entities = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Invalid room profile entities for {room_id}")
            return []
        agent_ids: set[str] = set()
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            if entity.get("entity_type") != "ai":
                continue
            agent_id = entity.get("agent_id")
            if agent_id:
                agent_ids.add(str(agent_id))
        return list(agent_ids)

    async def _agent_id_from_actor(self, room_id: str, actor_id: str) -> Optional[str]:
        """Lookup agent_id for an actor by their entity_id (dbref).

        Args:
            room_id: The room where the event occurred.
            actor_id: The actor's entity_id (e.g., "#3").

        Returns:
            The agent_id if the actor is an AI agent, None otherwise.
        """
        if not room_id or not actor_id:
            return None
        try:
            raw = await self.redis.hget(RedisKeys.room_profile(room_id), "entities_present")
        except Exception as e:
            logger.error(f"Failed to read room profile for {room_id}: {e}")
            return None
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            entities = json.loads(raw)
        except json.JSONDecodeError:
            return None
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            if entity.get("entity_id") == actor_id:
                return entity.get("agent_id")
        return None

    def register_agent(self, agent_id: str, initial_room: str = "") -> None:
        """Register an agent with the mediator.

        Args:
            agent_id: Unique identifier for the agent.
            initial_room: Optional initial room ID for the agent.
        """
        self.registered_agents.add(agent_id)
        logger.info(
            f"Registered agent {agent_id}"
        )

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the mediator.

        Args:
            agent_id: Agent ID to unregister.
        """
        self.registered_agents.discard(agent_id)
        logger.info(f"Unregistered agent {agent_id}")

    async def _check_auto_analysis_trigger(self) -> None:
        """Check if we should trigger auto-analysis.

        Conditions checked:
        1. Auto-analysis enabled in config
        2. Mediator not paused
        3. Cooldown period has elapsed since last check
        4. All non-sleeping agents are idle (ready/done/fail status, not in backoff)
        5. Idle threshold duration reached

        If all conditions met, triggers conversation scanning.
        Sleeping agents are excluded from idle detection but CAN receive analysis tasks.
        """
        if not self.config.auto_analysis_enabled:
            return

        # Don't trigger auto-analysis if mediator is paused
        if await self._is_paused():
            return

        now = _utc_now()

        # Check cooldown
        elapsed = (now - self._last_auto_analysis_check).total_seconds()
        if elapsed < self.config.auto_analysis_cooldown_seconds:
            return

        # Check if ALL agents are idle (and not sleeping)
        all_idle = True
        for agent_id in self.registered_agents:
            # Check if agent is sleeping
            agent_profile = await self.redis.hgetall(RedisKeys.agent_profile(agent_id))
            is_sleeping_raw = agent_profile.get(b"is_sleeping") or agent_profile.get("is_sleeping")
            if is_sleeping_raw:
                is_sleeping = is_sleeping_raw.decode() if isinstance(is_sleeping_raw, bytes) else is_sleeping_raw
                if is_sleeping.lower() == "true":
                    # Skip sleeping agents - they should not trigger auto-analysis
                    continue

            turn_request = await self._get_turn_request(agent_id)
            if not turn_request:
                all_idle = False
                break

            status = turn_request.status
            if status in (TurnRequestStatus.ASSIGNED, TurnRequestStatus.IN_PROGRESS, TurnRequestStatus.ABORT_REQUESTED):
                all_idle = False
                break

            # Check if failed turn is still in backoff period
            if status == TurnRequestStatus.FAIL:
                next_attempt_at_str = turn_request.next_attempt_at
                if next_attempt_at_str:
                    try:
                        next_attempt_at = datetime.fromisoformat(next_attempt_at_str)
                        if datetime.now(timezone.utc) < next_attempt_at:
                            all_idle = False
                            break
                    except (ValueError, TypeError):
                        # Invalid timestamp, treat as idle
                        pass

        if not all_idle:
            # Reset idle timer if system not fully idle
            self._auto_analysis_idle_start = None
            return

        # Track idle duration
        if self._auto_analysis_idle_start is None:
            self._auto_analysis_idle_start = now
            logger.debug("Auto-analysis: system idle detected, starting timer")
            return

        idle_duration = (now - self._auto_analysis_idle_start).total_seconds()
        if idle_duration < self.config.auto_analysis_idle_seconds:
            logger.debug(
                f"Auto-analysis: system idle for {idle_duration:.0f}s "
                f"(threshold: {self.config.auto_analysis_idle_seconds}s)"
            )
            return

        # Threshold reached!
        logger.info(f"Auto-analysis: triggered after {idle_duration:.0f}s idle")
        self._last_auto_analysis_check = now
        self._auto_analysis_idle_start = None  # Reset for next cycle

        await self._scan_for_unanalyzed_conversations()

    async def _scan_for_unanalyzed_conversations(self) -> None:
        """Scan agents for conversations needing analysis.

        Uses round-robin to assign analysis to ONE agent per trigger.
        Tries agents in order starting from self._turn_index until one
        successfully receives an assignment.

        Uses existing turn assignment infrastructure via _handle_analysis_command().
        """
        # Convert set to list for indexing
        agents_list = list(self.registered_agents)
        if not agents_list:
            logger.debug("Auto-analysis: no registered agents")
            return

        n = len(agents_list)
        assigned_agent = None

        # Try each agent in round-robin order
        for i in range(n):
            agent_id = agents_list[(self._turn_index + i) % n]

            try:
                # Load cached conversation report from Redis
                report_key = RedisKeys.agent_conversation_report(agent_id)
                report_json = await self.redis.get(report_key)

                if not report_json:
                    logger.debug(f"Auto-analysis: no conversation report for {agent_id}")
                    continue

                if isinstance(report_json, bytes):
                    report_json = report_json.decode("utf-8")

                report = json.loads(report_json)

                if not isinstance(report, dict):
                    logger.warning(
                        f"Auto-analysis: invalid report format for {agent_id} "
                        f"(expected dict, got {type(report).__name__})"
                    )
                    continue

                # Find unanalyzed conversations
                # Document types come from conversation report structure:
                # - "mud-world": documents from world events (user turns)
                # - "mud-agent": documents from agent actions (assistant turns)
                # - "analysis": analysis documents created by analysis_dialogue scenario
                # These column names come from ConversationModel.get_conversation_report()
                unanalyzed = []
                for conversation_id, doc_counts in report.items():
                    if not isinstance(doc_counts, dict):
                        logger.warning(
                            f"Auto-analysis: invalid doc_counts for {conversation_id} in {agent_id}"
                        )
                        continue

                    has_mud_docs = (
                        doc_counts.get("mud-world", 0) > 0
                        or doc_counts.get("mud-agent", 0) > 0
                    )
                    has_analysis = doc_counts.get("analysis", 0) > 0

                    if has_mud_docs and not has_analysis:
                        timestamp = doc_counts.get("timestamp_max", "")
                        unanalyzed.append((conversation_id, timestamp))

                if not unanalyzed:
                    logger.debug(f"Auto-analysis: no unanalyzed conversations for {agent_id}")
                    continue

                # Sort by timestamp (oldest first)
                unanalyzed.sort(key=lambda x: x[1])
                conversation_id, timestamp = unanalyzed[0]

                logger.info(
                    f"Auto-analysis: found {len(unanalyzed)} unanalyzed conversation(s) "
                    f"for {agent_id}, triggering analysis for oldest: {conversation_id}"
                )

                # TODO - Phase 4: Implement self-turns where agents can initiate
                # their own processing without explicit conversation targets. This
                # will enable agents to autonomously explore topics, reflect on
                # recent experiences, or pursue creative initiatives during idle time.
                # Note: Self-turns should respect sleeping state (no self-turns for
                # sleeping agents), but analysis tasks are allowed for sleeping agents.

                # Assign analysis turn using existing infrastructure
                success = await self._handle_analysis_command(
                    agent_id=agent_id,
                    scenario="analysis_dialogue",
                    conversation_id=conversation_id,
                    guidance=None,
                )

                if success:
                    assigned_agent = agent_id
                    self._turn_index = (self._turn_index + i + 1) % n
                    logger.info(
                        f"Auto-analysis: assigned analysis to {agent_id} "
                        f"(round-robin index updated to {self._turn_index})"
                    )
                    break  # Stop after first successful assignment
                else:
                    logger.debug(
                        f"Auto-analysis: {agent_id} unavailable, trying next agent"
                    )

            except Exception as e:
                logger.error(
                    f"Auto-analysis: error scanning {agent_id}: {e}",
                    exc_info=True
                )
                continue

        if not assigned_agent:
            logger.debug("Auto-analysis: no agents had unanalyzed conversations or were available")
