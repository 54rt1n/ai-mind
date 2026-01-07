# andimud_mediator/mixins/events.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Events mixin for the mediator service."""

import asyncio
import json
import logging
from typing import Optional

from aim_mud_types import MUDEvent, RedisKeys
from aim_mud_types.helper import _utc_now

logger = logging.getLogger(__name__)


class EventsMixin:
    """Events mixin for the mediator service."""

    async def run_event_router(self) -> None:
        """Read mud:events, filter by room, distribute to agents.

        Main event routing loop:
        1. XREAD from mud:events stream (blocking with timeout)
        2. For each event:
           - Parse into MUDEvent
           - Look up which agents are in that room
           - Enrich event with room state
           - XADD to each relevant agent's stream
        3. Track agent locations from movement events
        """
        logger.info("Event router started")

        while self.running:
            try:
                # Check if paused
                if await self._is_paused():
                    logger.debug("Event router paused, sleeping...")
                    await asyncio.sleep(1)
                    continue

                # Block-read from event stream
                result = await self.redis.xread(
                    {self.config.event_stream: self.last_event_id},
                    block=int(self.config.event_poll_timeout * 1000),
                    count=100,
                )

                if not result:
                    # No events, check agent states for retry/crash
                    await self._check_agent_states()
                    continue

                for stream_name, messages in result:
                    for msg_id, data in messages:
                        # Update last event ID
                        if isinstance(msg_id, bytes):
                            msg_id = msg_id.decode("utf-8")
                        self.last_event_id = msg_id

                        try:
                            await self._process_event(msg_id, data)
                        except Exception as e:
                            logger.error(
                                f"Error processing event {msg_id}: {e}",
                                exc_info=True,
                            )

                # Trim processed events from stream (based on hash)
                await self._trim_processed_events()

            except asyncio.CancelledError:
                logger.info("Event router cancelled")
                break
            except Exception as e:
                logger.error(f"Error in event router: {e}", exc_info=True)
                continue

        logger.info("Event router stopped")

    async def _process_event(
        self, msg_id: str, data: dict[bytes, bytes] | dict[str, str]
    ) -> None:
        """Process a single event from the stream.

        Args:
            msg_id: Redis stream message ID.
            data: Raw event data from Redis.
        """
        # Check if already processed (idempotency check)
        try:
            already_processed = await self.redis.hexists(
                RedisKeys.EVENTS_PROCESSED,
                msg_id,
            )
            if already_processed:
                logger.debug(f"Event {msg_id} already processed, skipping")
                return
        except Exception as e:
            logger.error(f"Failed to check processed hash for {msg_id}: {e}")
            # Continue anyway - better to duplicate than to lose

        # Parse event data
        raw_data = data.get(b"data") or data.get("data")
        if raw_data is None:
            logger.warning(f"Event {msg_id} missing data field")
            return

        if isinstance(raw_data, bytes):
            raw_data = raw_data.decode("utf-8")

        event_dict = json.loads(raw_data)
        event = MUDEvent.from_dict(event_dict)
        event.event_id = msg_id

        logger.debug(
            f"Processing event {msg_id}: {event.event_type.value} "
            f"from {event.actor} in {event.room_id}"
        )

        # Check for control commands (@dream, @dreamer)
        if await self._try_handle_control_command(event):
            # Command was handled - mark processed and don't distribute
            await self._mark_event_processed(msg_id, [])
            return

        # Enrich event with room state
        enriched = await self.enrich_event(event)

        # Determine which agents should receive this event
        agents_to_notify = await self._agents_from_room_profile(event.room_id)
        if self.registered_agents:
            agents_to_notify = [
                a for a in agents_to_notify if a in self.registered_agents
            ]

        # Identify self-agent but don't filter them out completely
        actor_agent_id = await self._agent_id_from_actor(event.room_id, event.actor_id)

        # Separate self-agent from others (self-agent gets event but no turn assignment)
        self_agent: Optional[str] = None
        if actor_agent_id and actor_agent_id in agents_to_notify:
            self_agent = actor_agent_id
            agents_to_notify = [a for a in agents_to_notify if a != actor_agent_id]

        # Filter out sleeping agents (they receive NO events while sleeping)
        sleeping_agents = []
        for agent_id in agents_to_notify:
            agent_profile = await self.redis.hgetall(RedisKeys.agent_profile(agent_id))
            is_sleeping_raw = agent_profile.get(b"is_sleeping") or agent_profile.get("is_sleeping")
            if is_sleeping_raw:
                is_sleeping = is_sleeping_raw.decode() if isinstance(is_sleeping_raw, bytes) else is_sleeping_raw
                if is_sleeping.lower() == "true":
                    sleeping_agents.append(agent_id)
        agents_to_notify = [a for a in agents_to_notify if a not in sleeping_agents]
        if sleeping_agents:
            logger.debug(f"Filtered out sleeping agents: {sleeping_agents}")

        if not agents_to_notify:
            logger.debug(f"No agents to notify for event in room {event.room_id}")
            # Still mark as processed (we've looked at it)
            await self._mark_event_processed(msg_id, [])
            return

        # Round-robin: only assign turn to ONE available agent
        # Only the agent who gets the turn (or is first in line if all busy)
        # receives the event in their stream. This prevents multiple agents
        # from responding to the same event.
        assigned_agent: Optional[str] = None
        first_candidate: Optional[str] = None
        if agents_to_notify:
            n = len(agents_to_notify)
            for i in range(n):
                candidate = agents_to_notify[(self._turn_index + i) % n]
                if first_candidate is None:
                    first_candidate = candidate
                assigned = await self._maybe_assign_turn(candidate, reason="events")
                if assigned:
                    assigned_agent = candidate
                    self._turn_index = (self._turn_index + i + 1) % n
                    break

        # Distribute event to the agent who got the turn, or if all are busy,
        # to the first candidate (they'll process it when they finish).
        target_agent = assigned_agent or first_candidate
        if target_agent:
            stream_key = RedisKeys.agent_events(target_agent)
            await self.redis.xadd(
                stream_key,
                {"data": json.dumps(enriched)},
                maxlen=self.config.agent_events_maxlen,
                approximate=True,
            )
            logger.debug(f"Distributed event {msg_id} to {target_agent}")

        # Push self-event to actor's stream (no turn assignment - just for awareness)
        if self_agent:
            # Check if self-agent is sleeping
            self_agent_profile = await self.redis.hgetall(RedisKeys.agent_profile(self_agent))
            is_sleeping_raw = self_agent_profile.get(b"is_sleeping") or self_agent_profile.get("is_sleeping")
            is_self_sleeping = False
            if is_sleeping_raw:
                is_sleeping = is_sleeping_raw.decode() if isinstance(is_sleeping_raw, bytes) else is_sleeping_raw
                is_self_sleeping = is_sleeping.lower() == "true"

            if not is_self_sleeping:
                enriched_self = enriched.copy()
                enriched_self["is_self_action"] = True
                stream_key = RedisKeys.agent_events(self_agent)
                await self.redis.xadd(
                    stream_key,
                    {"data": json.dumps(enriched_self)},
                    maxlen=self.config.agent_events_maxlen,
                    approximate=True,
                )
                logger.debug(f"Pushed self-action event {msg_id} to {self_agent}")

        # Mark event as processed with the target agent
        await self._mark_event_processed(msg_id, [target_agent] if target_agent else [])

    async def _mark_event_processed(
        self, msg_id: str, agents: list[str]
    ) -> None:
        """Mark an event as processed in the hash.

        Args:
            msg_id: Redis stream message ID.
            agents: List of agent IDs that received the event.
        """
        try:
            timestamp = _utc_now().isoformat()
            agent_list = ",".join(agents) if agents else ""
            value = f"{timestamp}|{agent_list}"

            await self.redis.hset(
                RedisKeys.EVENTS_PROCESSED,
                msg_id,
                value,
            )

            logger.debug(
                f"Marked event {msg_id} as processed (agents: {agent_list or 'none'})"
            )
        except Exception as e:
            logger.error(f"Failed to mark event {msg_id} as processed: {e}")

    async def _trim_processed_events(self) -> None:
        """Trim processed events from the stream based on hash.

        Only trims events that are confirmed processed (in the hash).
        Uses the minimum ID in the hash as the trim point.
        """
        if self.last_event_id == "0":
            return

        try:
            # Get all processed event IDs
            processed_ids = await self.redis.hkeys(RedisKeys.EVENTS_PROCESSED)

            if not processed_ids:
                return

            # Decode and find minimum
            ids = []
            for key in processed_ids:
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
                ids.append(key)

            # Trim stream up to minimum processed event
            min_id = min(ids)
            await self.redis.xtrim(
                self.config.event_stream,
                minid=min_id,
                approximate=True,
            )
            logger.debug(f"Trimmed event stream up to {min_id}")
        except Exception as e:
            logger.error(f"Failed to trim event stream: {e}")

    async def _cleanup_processed_hash(self) -> None:
        """Remove old entries from processed hash, keeping most recent N.

        Called periodically to prevent unbounded hash growth.
        """
        try:
            # Get all processed event IDs
            processed_ids = await self.redis.hkeys(RedisKeys.EVENTS_PROCESSED)

            keep_count = self.config.events_processed_hash_max
            if len(processed_ids) <= keep_count:
                return  # No cleanup needed

            # Decode and sort
            ids = []
            for key in processed_ids:
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
                ids.append(key)

            ids.sort()  # Event IDs are sortable timestamps

            # Remove oldest entries
            to_remove = ids[:-keep_count]
            if to_remove:
                await self.redis.hdel(RedisKeys.EVENTS_PROCESSED, *to_remove)
                logger.info(
                    f"Cleaned up {len(to_remove)} old processed event entries"
                )
        except Exception as e:
            logger.error(f"Failed to cleanup processed hash: {e}")
