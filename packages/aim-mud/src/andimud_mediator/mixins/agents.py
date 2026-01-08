# andimud_mediator/mixins/agents.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Agents mixin for the mediator service."""

import json
import logging
from datetime import datetime, timezone
from typing import Optional
import uuid

from aim_mud_types import RedisKeys
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

                status = turn_request.get("status")

                # Case 1: Failed turn ready to retry
                if status == "fail":
                    next_attempt_at_str = turn_request.get("next_attempt_at", "")
                    if next_attempt_at_str:
                        next_attempt_at = datetime.fromisoformat(next_attempt_at_str)
                        if datetime.now(timezone.utc) >= next_attempt_at:
                            logger.info(f"Retrying failed turn for {agent_id}")
                            await self._maybe_assign_turn(agent_id, reason="retry")

                # Case 2: Stale heartbeat (worker crashed)
                elif status == "in_progress":
                    heartbeat_at_str = turn_request.get("heartbeat_at", "")
                    if heartbeat_at_str:
                        heartbeat_at = datetime.fromisoformat(heartbeat_at_str)
                        stale_seconds = (datetime.now(timezone.utc) - heartbeat_at).total_seconds()
                        if stale_seconds > 300:  # 5 minutes
                            logger.error(
                                f"Worker {agent_id} crashed (no heartbeat for {stale_seconds:.0f}s)"
                            )
                            stale_duration = f"{int(stale_seconds // 60)}m{int(stale_seconds % 60)}s"
                            await self.redis.hset(
                                RedisKeys.agent_turn_request(agent_id),
                                mapping={
                                    "status": "crashed",
                                    "status_reason": f"Heartbeat stale for {stale_duration}"
                                }
                            )
            except Exception as e:
                logger.error(f"Error checking state for {agent_id}: {e}", exc_info=True)
                continue

    async def _maybe_assign_turn(self, agent_id: str, reason: str = "events") -> bool:
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

        status = current.get("status")

        # Block crashed workers
        if status == "crashed":
            logger.debug(f"Agent {agent_id} crashed, not assigning")
            return False

        # Block if actively processing or being aborted
        if status in ("assigned", "in_progress", "abort_requested"):
            return False

        # Check if failed turn ready to retry
        if status == "fail":
            next_attempt_at_str = current.get("next_attempt_at", "")
            if next_attempt_at_str:
                next_attempt_at = datetime.fromisoformat(next_attempt_at_str)
                if datetime.now(timezone.utc) < next_attempt_at:
                    # Still in backoff period
                    return False
            # Else: no next_attempt_at (max attempts) or past retry time, fall through to assign

        # Available: status is "ready" or "fail" past retry time
        # Create new turn assignment
        turn_id = str(uuid.uuid4())
        turn_sequence_id = await self._next_sequence_id()
        assigned_at = _utc_now().isoformat()
        current_turn_id = current.get("turn_id", "")

        # Preserve attempt_count if retrying a failed turn
        attempt_count = "0"
        if status == "fail":
            attempt_count = current.get("attempt_count", "0")

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
            redis.call('HSET', key, 'status', 'assigned')
            redis.call('HSET', key, 'reason', ARGV[4])
            redis.call('HSET', key, 'assigned_at', ARGV[5])
            redis.call('HSET', key, 'heartbeat_at', ARGV[5])
            redis.call('HSET', key, 'deadline_ms', ARGV[6])
            redis.call('HSET', key, 'attempt_count', ARGV[7])
            redis.call('HSET', key, 'sequence_id', ARGV[8])

            return 1  -- Success
        """

        result = await self.redis.eval(
            lua_script,
            1,  # number of keys
            RedisKeys.agent_turn_request(agent_id),
            current_turn_id or "",
            status,
            turn_id,
            reason,
            assigned_at,
            str(self.config.turn_request_ttl_seconds * 1000),
            attempt_count,
            str(turn_sequence_id)
        )

        if result == 1:
            # CAS succeeded - set TTL
            await self.redis.expire(
                RedisKeys.agent_turn_request(agent_id),
                self.config.turn_request_ttl_seconds
            )
            logger.info(f"Assigned turn to {agent_id} (sequence_id={turn_sequence_id}, reason={reason}, attempt={attempt_count})")
            return True
        else:
            # CAS failed - state changed between check and assign
            logger.debug(f"CAS failed for {agent_id}, state changed during assignment")
            return False

    async def _get_turn_request(self, agent_id: str) -> dict[str, str]:
        """Fetch the current turn request hash for an agent."""
        key = RedisKeys.agent_turn_request(agent_id)
        data = await self.redis.hgetall(key)
        if not data:
            return {}
        result: dict[str, str] = {}
        for k, v in data.items():
            if isinstance(k, bytes):
                k = k.decode("utf-8")
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            result[str(k)] = str(v)
        return result

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
