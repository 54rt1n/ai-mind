# andimud_mediator/mixins/agents.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Agents mixin for the mediator service."""

import logging
from datetime import datetime, timezone
from typing import Optional

from aim.conversation.model import ConversationModel
from aim_mud_types import MUDTurnRequest, TurnRequestStatus, TurnReason
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

                # Case 1: Failed/retry turn ready to retry
                if status in (TurnRequestStatus.RETRY, TurnRequestStatus.FAIL):
                    if turn_request.next_attempt_at:
                        # next_attempt_at is already a datetime after Pydantic deserialization
                        if datetime.now(timezone.utc) >= turn_request.next_attempt_at:
                            # Check if ANY agent is busy before assigning retry
                            # This enforces one-at-a-time concurrency
                            #if await self._any_agent_processing():
                            #    logger.debug(f"Skipping retry for {agent_id} - another agent is processing")
                            #    continue

                            logger.info(f"Retrying turn for {agent_id} (was {status.value})")
                            await self._maybe_assign_turn(agent_id, reason=TurnReason.RETRY)

                # Case 2: Stale heartbeat (worker crashed)
                elif status == TurnRequestStatus.IN_PROGRESS:
                    if turn_request.heartbeat_at:
                        stale_seconds = (datetime.now(timezone.utc) - turn_request.heartbeat_at).total_seconds()
                        if stale_seconds > 300:  # 5 minutes
                            stale_duration = f"{int(stale_seconds // 60)}m{int(stale_seconds % 60)}s"

                            from aim_mud_types.turn_request_helpers import (
                                transition_turn_request_and_update_async,
                            )
                            updated = await transition_turn_request_and_update_async(
                                self.redis,
                                agent_id,
                                turn_request,
                                expected_turn_id=turn_request.turn_id,
                                status=TurnRequestStatus.CRASHED,
                                status_reason=f"Heartbeat stale for {stale_duration}",
                                set_completed=True,
                                update_heartbeat=False,
                            )

                            if updated:
                                logger.error(
                                    f"Worker {agent_id} crashed (no heartbeat for {stale_seconds:.0f}s)"
                                )
                            else:
                                logger.debug(
                                    f"Agent '{agent_id}' turn_request missing or changed, skipping crash detection"
                                )
            except Exception as e:
                logger.error(f"Error checking state for {agent_id}: {e}", exc_info=True)
                continue

        # Case 3: System idle - assign idle turn if all agents ready for duration
        all_ready = await self._all_agents_ready()

        if all_ready:
            # Track when system entered "all ready" state
            if self._system_ready_since is None:
                self._system_ready_since = datetime.now(timezone.utc)

            # Check if system has been ready for threshold duration
            ready_duration = (datetime.now(timezone.utc) - self._system_ready_since).total_seconds()
            if ready_duration >= self.config.system_idle_seconds:
                # System idle - assign turn
                agents_list = sorted(self.registered_agents)
                if agents_list:
                    n = len(agents_list)
                    for i in range(n):
                        candidate = agents_list[(self._turn_index + i) % n]
                        assigned = await self._maybe_assign_turn(candidate, reason=TurnReason.IDLE)
                        if assigned:
                            self._turn_index = (self._turn_index + i + 1) % n
                            logger.debug(
                                f"Assigned idle turn to {candidate} "
                                f"(system ready for {ready_duration:.1f}s)"
                            )
                            break
        else:
            # Reset system ready timestamp when agents become busy
            self._system_ready_since = None


    async def _any_agent_processing(self) -> bool:
        """Check if any agent is currently processing a turn.

        Used to enforce one-at-a-time concurrency across all assignment paths.

        Returns:
            True if any agent is in ASSIGNED, IN_PROGRESS, EXECUTING, or EXECUTE status
        """
        for agent_id in self.registered_agents:
            turn_request = await self._get_turn_request(agent_id)
            if not turn_request:
                continue

            if turn_request.status in (
                TurnRequestStatus.ASSIGNED,
                TurnRequestStatus.IN_PROGRESS,
                TurnRequestStatus.EXECUTING,
                TurnRequestStatus.EXECUTE,
                TurnRequestStatus.ABORT_REQUESTED,
            ):
                return True

        return False

    async def _all_agents_ready(self) -> bool:
        """Check if ALL registered agents are in READY status."""
        for agent_id in self.registered_agents:
            turn_request = await self._get_turn_request(agent_id)
            if not turn_request or turn_request.status != TurnRequestStatus.READY:
                return False
        return True

    async def _is_player_activity_idle(self, idle_seconds: int) -> bool:
        """Check if player activity has been idle for at least idle_seconds.

        Uses tracked timestamp from LAST_PLAYER_ACTIVITY key instead of
        stream last-generated-id. Only non-AI events (PLAYER, NPC, SYSTEM)
        update this timestamp.

        Args:
            idle_seconds: Minimum idle time in seconds.

        Returns:
            True if player activity idle for at least idle_seconds,
            or if no player activity recorded yet.
        """
        try:
            from aim_mud_types import RedisKeys

            last_activity_ms = await self.redis.get(RedisKeys.LAST_PLAYER_ACTIVITY)
            if not last_activity_ms:
                return True  # No player activity yet = idle

            # Decode bytes if necessary
            if isinstance(last_activity_ms, bytes):
                last_activity_ms = last_activity_ms.decode('utf-8')

            last_ts_ms = int(last_activity_ms)
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            idle_ms = now_ms - last_ts_ms
            return idle_ms >= (idle_seconds * 1000)
        except Exception:
            # If key missing or error, treat as idle
            return True

    async def _is_stream_idle(self, stream_key: str, idle_seconds: int) -> bool:
        """Check if stream has been idle for at least idle_seconds.

        Args:
            stream_key: Redis stream key to check.
            idle_seconds: Minimum idle time in seconds.

        Returns:
            True if stream has been idle for at least idle_seconds,
            or if stream doesn't exist/is empty.
        """
        try:
            info = await self.redis.xinfo_stream(stream_key)
            last_id = info.get("last-generated-id") or info.get(b"last-generated-id")
            if isinstance(last_id, bytes):
                last_id = last_id.decode("utf-8")
            if not last_id or last_id in ("0", "0-0"):
                return True  # Empty stream = idle

            last_ts_ms = int(last_id.split("-")[0])
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            idle_ms = now_ms - last_ts_ms
            return idle_ms >= (idle_seconds * 1000)
        except Exception:
            # Stream doesn't exist = idle
            return True

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

        # Check if agent is paused via Redis
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)
        is_paused = await client.is_agent_paused(agent_id)
        if is_paused:
            logger.debug(f"Agent {agent_id} paused, not assigning turn")
            return False

        status = current.status

        # Block crashed workers
        if status == TurnRequestStatus.CRASHED:
            logger.debug(f"Agent {agent_id} crashed, not assigning")
            return False

        # Block if actively processing or being aborted
        if status in (TurnRequestStatus.ASSIGNED, TurnRequestStatus.IN_PROGRESS,
                      TurnRequestStatus.ABORT_REQUESTED, TurnRequestStatus.EXECUTING):
            return False

        # Check if retry/fail turn in backoff period
        if status in (TurnRequestStatus.RETRY, TurnRequestStatus.FAIL):
            if current.next_attempt_at:
                # next_attempt_at is already a datetime after Pydantic deserialization
                if datetime.now(timezone.utc) < current.next_attempt_at:
                    # Still in backoff period - not available
                    return False
            # Else: no next_attempt_at (max attempts) or past retry time, fall through to assign

        # Available: status is "ready" or "fail" past retry time
        # Convert reason to TurnReason enum
        turn_reason_enum = reason if isinstance(reason, TurnReason) else TurnReason(reason)

        from aim_mud_types.turn_request_helpers import assign_turn_request_async

        # Determine initial status based on reason
        # Immediate commands (FLUSH, CLEAR, NEW) get EXECUTE status for priority handling
        if turn_reason_enum.is_immediate_command():
            initial_status = TurnRequestStatus.EXECUTE
        else:
            initial_status = TurnRequestStatus.ASSIGNED

        # Preserve attempt_count and metadata if retrying a failed turn
        attempt_count = 0
        metadata = None
        if status in (TurnRequestStatus.RETRY, TurnRequestStatus.FAIL):
            attempt_count = current.attempt_count
            metadata = current.metadata  # Preserve metadata on retry

            # Validate: DREAM turns require metadata with scenario
            if current.reason == TurnReason.DREAM and not metadata:
                logger.warning(
                    f"DREAM turn for {agent_id} has no metadata, marking as FAIL"
                )
                from aim_mud_types.turn_request_helpers import (
                    transition_turn_request_and_update_async,
                )
                await transition_turn_request_and_update_async(
                    self.redis,
                    agent_id,
                    current,
                    expected_turn_id=current.turn_id,
                    status=TurnRequestStatus.FAIL,
                    message="Dream turn missing metadata",
                    next_attempt_at=None,
                    set_completed=True,
                    update_heartbeat=False,
                )
                return False
            if current.reason == TurnReason.DREAM and metadata and not metadata.get("scenario"):
                logger.warning(
                    f"DREAM turn for {agent_id} metadata missing scenario, marking as FAIL"
                )
                from aim_mud_types.turn_request_helpers import (
                    transition_turn_request_and_update_async,
                )
                await transition_turn_request_and_update_async(
                    self.redis,
                    agent_id,
                    current,
                    expected_turn_id=current.turn_id,
                    status=TurnRequestStatus.FAIL,
                    message="Dream turn missing scenario in metadata",
                    next_attempt_at=None,
                    set_completed=True,
                    update_heartbeat=False,
                )
                return False

        success, turn_request, result = await assign_turn_request_async(
            self.redis,
            agent_id,
            turn_reason_enum,
            attempt_count=attempt_count,
            status=initial_status,
            expected_turn_id=current.turn_id,
            skip_availability_check=True,
            **(metadata or {}),
        )

        if success and turn_request:
            logger.info(
                f"Assigned turn to {agent_id} "
                f"(sequence_id={turn_request.sequence_id}, "
                f"status={turn_request.status.value}, "
                f"reason={turn_request.reason.value}, "
                f"attempt={turn_request.attempt_count})"
            )
            return True
        else:
            # CAS failed - state changed between check and assign
            logger.debug(f"Assign failed for {agent_id}: {result}")
            return False

    async def _get_turn_request(self, agent_id: str) -> Optional[MUDTurnRequest]:
        """Fetch the current turn request hash for an agent."""
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)
        return await client.get_turn_request(agent_id)

    async def _agents_from_room_profile(self, room_id: str) -> list[str]:
        """Lookup agent_ids present in a room profile."""
        if not room_id:
            return []
        try:
            from aim_mud_types.client import RedisMUDClient
            client = RedisMUDClient(self.redis)
            room_profile = await client.get_room_profile(room_id)
            if not room_profile:
                return []

            agent_ids: set[str] = set()
            for entity in room_profile.entities:
                if entity.entity_type != "ai":
                    continue
                if entity.agent_id:
                    agent_ids.add(str(entity.agent_id))
            return list(agent_ids)
        except Exception as e:
            logger.error(f"Failed to read room profile for {room_id}: {e}")
            return []

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
            from aim_mud_types.client import RedisMUDClient
            client = RedisMUDClient(self.redis)
            room_profile = await client.get_room_profile(room_id)
            if not room_profile:
                return None

            for entity in room_profile.entities:
                if entity.entity_id == actor_id:
                    return entity.agent_id
            return None
        except Exception as e:
            logger.error(f"Failed to read room profile for {room_id}: {e}")
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

