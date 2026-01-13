# andimud_mediator/mixins/events.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Events mixin for the mediator service."""

import asyncio
import json
import logging
from typing import Any, Optional

from aim_mud_types import EventType, MUDEvent, RedisKeys, TurnReason, TurnRequestStatus

logger = logging.getLogger(__name__)


class EventsMixin:
    """Events mixin for the mediator service."""

    async def load_last_event_id_from_hash(self) -> None:
        """Load last processed event ID from processing hash."""
        try:
            from aim_mud_types.client import RedisMUDClient
            client = RedisMUDClient(self.redis)
            max_id = await client.get_max_processed_mud_event_id()

            if not max_id:
                logger.info("No processed events hash found, starting from 0")
                self.last_event_id = "0"
                return

            self.last_event_id = max_id

            processed_count = len(await client.get_mud_event_processed_ids())
            logger.info(
                "Loaded last_event_id from processed hash: %s (%d events in hash)",
                self.last_event_id,
                processed_count,
            )
        except Exception as e:
            logger.error(f"Failed to load last_event_id from hash: {e}")
            self.last_event_id = "0"

    async def enrich_event(self, event: MUDEvent) -> dict[str, Any]:
        """Add room state to event.

        Currently a placeholder that returns the event with empty room_state.
        Future implementation will query Evennia REST API for current room state.

        Args:
            event: The MUDEvent to enrich.

        Returns:
            Dictionary with event data and enrichment fields.
        """
        # Build base event dictionary
        enriched = event.to_redis_dict()
        enriched["id"] = event.event_id

        # No enrichment when world_state is absent (worker pulls from agent profile)
        enriched["enriched"] = False
        return enriched

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
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)

        while self.running:
            try:
                # Check if paused
                if await self._is_paused():
                    logger.debug("Event router paused, sleeping...")
                    await asyncio.sleep(1)
                    continue

                # Block-read from event stream
                messages = await client.read_mud_events(
                    self.last_event_id,
                    block_ms=int(self.config.event_poll_timeout * 1000),
                    count=100,
                    stream_key=self.config.event_stream,
                )

                if not messages:
                    # No events, check agent states for retry/crash
                    await self._check_agent_states()
                    continue

                for msg_id, data in messages:
                    # Update last event ID
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
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)
        # Check if already processed (idempotency check)
        try:
            already_processed = await client.is_mud_event_processed(msg_id)
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

        # Check for control commands (@dream, @dreamer, @planner, @plan, @update)
        if await self._try_handle_control_command(event):
            # Command was handled - mark processed and don't distribute
            await self._mark_event_processed(msg_id, [])
            return

        # Check for planner commands (@planner, @plan, @update)
        if await self._try_handle_planner_command(event):
            # Command was handled - mark processed and don't distribute
            await self._mark_event_processed(msg_id, [])
            return

        # Enrich event with room state
        enriched = await self.enrich_event(event)

        # Assign sequence ID for chronological ordering
        sequence_id = await self._next_sequence_id()
        enriched["sequence_id"] = sequence_id
        logger.debug(f"Event {msg_id} assigned sequence_id={sequence_id}")

        # Determine which agents should receive this event
        agents_to_notify = await self._agents_from_room_profile(event.room_id)
        if self.registered_agents:
            agents_to_notify = [
                a for a in agents_to_notify if a in self.registered_agents
            ]

        # Identify self-agent: prefer metadata (no lookup), fall back to room profile
        # Metadata approach avoids race conditions from stale room profiles
        actor_agent_id = event.metadata.get("actor_agent_id")
        if actor_agent_id is None:
            # Fallback: lookup from room profile (may fail for arrival events)
            actor_agent_id = await self._agent_id_from_actor(event.room_id, event.actor_id)

        # Determine if this should be treated as a self-action for the actor.
        # Self-actions are delivered only to the actor with a flag for formatting.
        self_action_agent_id: Optional[str] = None
        if actor_agent_id:
            if self.registered_agents and actor_agent_id not in self.registered_agents:
                actor_agent_id = None
            elif event.event_type in (EventType.OBJECT, EventType.EMOTE):
                self_action_agent_id = actor_agent_id
            elif event.event_type == EventType.MOVEMENT:
                destination_room = event.metadata.get("destination_room")
                destination_name = event.metadata.get("destination_room_name")
                if destination_room and event.room_id == destination_room:
                    self_action_agent_id = actor_agent_id
                elif destination_name and event.room_name == destination_name:
                    self_action_agent_id = actor_agent_id

        # Filter actor from broadcast list (self-action delivered separately)
        if actor_agent_id and actor_agent_id in agents_to_notify:
            agents_to_notify = [a for a in agents_to_notify if a != actor_agent_id]

        # Build delivery list (others + optional self-action)
        agents_for_delivery = list(agents_to_notify)
        if self_action_agent_id and self_action_agent_id not in agents_for_delivery:
            agents_for_delivery.append(self_action_agent_id)

        # Filter out sleeping agents (they receive NO events while sleeping)
        sleeping_agents = []
        for agent_id in agents_for_delivery:
            is_sleeping = await client.get_agent_is_sleeping(agent_id)
            if is_sleeping:
                sleeping_agents.append(agent_id)
        agents_for_delivery = [a for a in agents_for_delivery if a not in sleeping_agents]
        agents_to_notify = [a for a in agents_to_notify if a not in sleeping_agents]
        if sleeping_agents:
            logger.debug(f"Filtered out sleeping agents: {sleeping_agents}")

        if not agents_for_delivery:
            logger.debug(f"No agents to notify for event in room {event.room_id}")
            # Still mark as processed (we've looked at it)
            await self._mark_event_processed(msg_id, [])
            return

        # Phase 1: Distribute event to agents (broadcast + optional self-action)
        for agent_id in agents_for_delivery:
            stream_key = RedisKeys.agent_events(agent_id)
            payload = dict(enriched)
            if self_action_agent_id and agent_id == self_action_agent_id:
                payload["is_self_action"] = True
            await client.append_agent_event(
                agent_id,
                {"data": json.dumps(payload)},
                maxlen=self.config.agent_events_maxlen,
                approximate=True,
                stream_key=stream_key,
            )
            logger.debug(f"Distributed event {msg_id} (seq={sequence_id}) to {agent_id}")

        # Phase 2: Assign turn ONLY if no agents are currently processing
        assigned_agent: Optional[str] = None

        # Check if any agent is processing (prevents parallel execution)
        any_processing = False
        for agent_id in agents_to_notify:
            turn_request = await self._get_turn_request(agent_id)
            if turn_request and turn_request.status == TurnRequestStatus.IN_PROGRESS:
                any_processing = True
                logger.debug(f"Agent {agent_id} is processing, blocking turn assignment")
                break

        if not any_processing and agents_to_notify:
            # Debounce: Wait briefly to ensure status is stable (not mid-transition)
            # This prevents assignment during the window when an agent completes
            # their turn and writes self-events that trigger another cycle
            await asyncio.sleep(0.1)

            # Re-check processing status to catch any transitions that occurred
            any_processing_recheck = False
            for agent_id in agents_to_notify:
                turn_request = await self._get_turn_request(agent_id)
                if turn_request and turn_request.status == TurnRequestStatus.IN_PROGRESS:
                    any_processing_recheck = True
                    logger.debug(
                        f"Agent {agent_id} started processing during debounce, "
                        f"blocking turn assignment"
                    )
                    break

            # Only assign if status remains stable
            if not any_processing_recheck:
                # System idle - assign turn via round-robin
                n = len(agents_to_notify)
                for i in range(n):
                    candidate = agents_to_notify[(self._turn_index + i) % n]
                    assigned = await self._maybe_assign_turn(candidate, reason=TurnReason.EVENTS)
                    if assigned:
                        assigned_agent = candidate
                        self._turn_index = (self._turn_index + i + 1) % n
                        logger.info(f"Assigned turn to {assigned_agent} for event {msg_id} (seq={sequence_id})")
                        break
        else:
            if any_processing:
                logger.debug(f"Event {msg_id} queued (agents busy, no turn assigned)")
            else:
                logger.debug(f"No agents ready for turn assignment")

        # NOTE: Self-actions are distributed only to the actor with is_self_action flag.
        if self_action_agent_id:
            logger.debug(f"Event {msg_id} is self-action for {self_action_agent_id}")

        # Mark event as processed with the assigned agent
        await self._mark_event_processed(
            msg_id,
            [assigned_agent] if assigned_agent else []
        )

    async def _mark_event_processed(
        self, msg_id: str, agents: list[str]
    ) -> None:
        """Mark an event as processed in the hash.

        Args:
            msg_id: Redis stream message ID.
            agents: List of agent IDs that received the event.
        """
        try:
            from aim_mud_types.client import RedisMUDClient
            client = RedisMUDClient(self.redis)
            await client.mark_mud_event_processed(msg_id, agents)

            agent_list = ",".join(agents) if agents else ""
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
            from aim_mud_types.client import RedisMUDClient
            client = RedisMUDClient(self.redis)
            min_id = await client.get_min_processed_mud_event_id()
            if not min_id:
                return
            await client.trim_mud_events_minid(
                min_id=min_id,
                approximate=True,
                stream_key=self.config.event_stream,
            )
            logger.debug(f"Trimmed event stream up to {min_id}")
        except Exception as e:
            logger.error(f"Failed to trim event stream: {e}")

    async def _cleanup_processed_hash(self) -> None:
        """Remove old entries from processed hash, keeping most recent N.

        Called periodically to prevent unbounded hash growth.
        """
        try:
            keep_count = self.config.events_processed_hash_max
            from aim_mud_types.client import RedisMUDClient
            client = RedisMUDClient(self.redis)
            removed = await client.trim_processed_mud_event_ids(keep_count)
            if removed:
                logger.info(
                    f"Cleaned up {removed} old processed event entries"
                )
        except Exception as e:
            logger.error(f"Failed to cleanup processed hash: {e}")
