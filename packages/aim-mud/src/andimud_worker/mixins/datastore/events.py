# aim/app/mud/worker/events.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Event draining for the MUD worker.

Handles consuming events from Redis streams with settling logic.
Extracted from worker.py lines 646-777
"""

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Optional

from aim_mud_types import MUDEvent, EventType
from ...adapter import format_self_action_guidance

if TYPE_CHECKING:
    from ...worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class EventsMixin:
    """Mixin for event draining methods.

    These methods are mixed into MUDAgentWorker in main.py.
    """

    async def drain_events(
        self: "MUDAgentWorker", timeout: float, accumulate_self_actions: bool = True, max_sequence_id: Optional[int] = None
    ) -> list[MUDEvent]:
        """Block until events arrive on agent's stream.

        Originally from worker.py lines 646-735

        Events are already enriched by mediator with room state.

        Args:
            timeout: Maximum seconds to block waiting for events.

        Returns:
            List of MUDEvent objects parsed from the stream.
        """
        events_unsorted: list[tuple[int, MUDEvent]] = []
        max_id = None
        scan_last_id: Optional[str] = None
        last_consumed_id: Optional[str] = None
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)

        refreshed_world_state = False

        async def _parse_result(result) -> None:
            nonlocal scan_last_id, last_consumed_id, refreshed_world_state
            for _stream_name, messages in result:
                for msg_id, data in messages:
                    # Track last seen id for scan cursor (always advance scan)
                    # msg_id may be bytes or str depending on Redis client config
                    if isinstance(msg_id, bytes):
                        msg_id = msg_id.decode("utf-8")
                    scan_last_id = msg_id
                    should_consume = True

                    # Parse event data
                    try:
                        # Data field contains the JSON payload
                        raw_data = data.get(b"data") or data.get("data")
                        if raw_data is None:
                            logger.warning(f"Event {msg_id} missing data field")
                            continue

                        if isinstance(raw_data, bytes):
                            raw_data = raw_data.decode("utf-8")

                        enriched = json.loads(raw_data)

                        # Extract sequence_id for sorting
                        sequence_id = enriched.get("sequence_id", 0)
                        if sequence_id == 0:
                            logger.warning(f"Event {msg_id} missing sequence_id")

                        # Skip events with sequence_id >= max_sequence_id
                        if max_sequence_id is not None and sequence_id >= max_sequence_id:
                            logger.debug(f"Skipping event {msg_id} (seq={sequence_id} >= {max_sequence_id})")
                            should_consume = False
                            continue

                        # Check for self-action flag (set by mediator for actor's own events)
                        is_self_action = enriched.pop("is_self_action", False)

                        event = MUDEvent.from_dict(enriched)
                        event.event_id = msg_id
                        event.metadata["sequence_id"] = sequence_id

                        if is_self_action:
                            if not refreshed_world_state:
                                room_id, character_id = await self._load_agent_world_state()
                                await self._load_room_profile(room_id, character_id)
                                refreshed_world_state = True
                            # Mark in metadata so conversation manager can format in first person
                            event.metadata["is_self_action"] = True
                            logger.debug(
                                f"Received self-action event (seq={sequence_id}): {event.event_type.value}"
                            )
                            # Format self-action guidance if not already formatted
                            content = event.content or ""
                            if (
                                not content.strip().startswith("═")
                                and "YOUR RECENT ACTION" not in content
                                and event.event_type in (EventType.MOVEMENT, EventType.OBJECT, EventType.EMOTE)
                            ):
                                event.content = format_self_action_guidance(
                                    [event],
                                    world_state=self.session.world_state,
                                )
                            events_unsorted.append((sequence_id, event))
                        else:
                            # Append as tuple with sequence_id for sorting
                            events_unsorted.append((sequence_id, event))
                            logger.debug(
                                f"Parsed event (seq={sequence_id}): {event.event_type.value} "
                                f"from {event.actor}"
                            )

                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.error(f"Failed to parse event {msg_id}: {e}")
                        # Still consume malformed events to avoid stalling the stream
                        should_consume = True
                    finally:
                        if should_consume:
                            last_consumed_id = msg_id

        # Snapshot the max stream id at drain start to avoid chasing new events
        try:
            max_id = await client.get_agent_events_last_id(
                self.config.agent_id,
                stream_key=self.config.agent_stream,
            )
        except Exception as e:
            # "no such key" is expected when stream doesn't exist yet (no events arrived)
            if "no such key" in str(e).lower():
                logger.debug(f"Event stream not yet created: {e}")
            else:
                logger.error(f"Redis error in drain_events (xinfo): {e}")
            return []

        if not max_id:
            return []

        # Drain events up to snapshot max_id
        start_id = self.session.last_event_id or "0"
        min_id = f"({start_id}" if start_id != "0" else "-"
        while True:
            try:
                result = await client.range_agent_events(
                    self.config.agent_id,
                    min_id,
                    max_id,
                    count=100,
                    stream_key=self.config.agent_stream,
                )
            except Exception as e:
                logger.error(f"Redis error in drain_events (xrange): {e}")
                break

            if not result:
                break

            await _parse_result([(self.config.agent_stream, result)])
            last_msg_id = scan_last_id
            if not last_msg_id or last_msg_id == max_id:
                break
            min_id = f"({last_msg_id}"

        if last_consumed_id:
            self.session.last_event_id = last_consumed_id

        # NOTE: Do NOT persist last_event_id here. Persistence happens after the speech
        # check determines whether events should be consumed:
        # - Speech turn: persisted in worker.py after speech check confirms speech
        # - Non-speech turn: persisted via _restore_event_position() (rollback)
        # - Exception: persisted via _restore_event_position() in exception handler
        # This ensures events are never lost if the worker crashes during processing.

        # Sort events by sequence_id to ensure chronological order
        events_unsorted.sort(key=lambda x: x[0])
        events = [event for _, event in events_unsorted]

        if events:
            logger.info(
                f"Drained {len(events)} events, sorted by sequence_id: "
                f"{[e.metadata.get('sequence_id', 0) for e in events]}"
            )

        return events

    async def _drain_with_settle(self: "MUDAgentWorker", max_sequence_id: Optional[int] = None) -> list[MUDEvent]:
        """Drain events with settling delay for cascading events.

        Originally from worker.py lines 736-777

        Drains events, waits settle_seconds, drains again.
        Repeats until a drain returns zero events. This allows
        cascading events (e.g., someone entering a room and immediately
        speaking) to be batched together.

        Args:
            max_sequence_id: Optional cap for first drain only. Subsequent
                drains during settle window ignore this cap to capture
                cascading events.

        Returns:
            All accumulated events from multiple drains.
        """
        settle_time = self.config.event_settle_seconds
        final_settle_time = settle_time // 3
        all_events: list[MUDEvent] = []
        drain_count = 0
        empty_after_events = 0

        while True:
            # Only cap the first drain with turn request's sequence_id
            # Subsequent drains during settle have no cap to catch cascades
            cap = max_sequence_id if drain_count == 0 else None
            events = await self.drain_events(timeout=0, max_sequence_id=cap)
            drain_count += 1
            if not events:
                if not all_events:
                    # No new events - cascade has settled
                    if drain_count == 1:
                        logger.warning(
                            "First drain returned 0 events - turn assigned but stream empty?"
                        )
                    break

                empty_after_events += 1
                if empty_after_events >= 2:
                    logger.info(
                        "Event cascade settled after %.1fs with %d total events",
                        settle_time,
                        len(all_events),
                    )
                    break

                logger.info(
                    "Drained 0 events after %d total, waiting %.1fs for stragglers",
                    len(all_events),
                    final_settle_time,
                )
                await asyncio.sleep(final_settle_time)
                continue

            all_events.extend(events)
            empty_after_events = 0
            logger.info(
                "Drained %d events (total %d), waiting %.1fs for more",
                len(events),
                len(all_events),
                settle_time,
            )
            await asyncio.sleep(settle_time)

        return all_events

    async def _restore_event_position(
        self: "MUDAgentWorker", saved_event_id: Optional[str]
    ) -> None:
        """Restore event stream position for non-speech turns or failed processing.

        Called when turn processing fails or when events should not be consumed
        (non-speech turns). Rolls back the in-memory event position to the saved
        position so the next drain will receive the same events.

        Redis persistence is NOT needed here because Redis never advanced during
        drain - it only advances in memory. Redis persistence only happens after
        speech check confirms a speech action occurred.

        Args:
            saved_event_id: Event ID to restore, or None to skip restoration
        """
        if saved_event_id is None:
            logger.debug("Skipping event position restore (events were consumed)")
            return

        logger.info(
            f"Restoring event position from {self.session.last_event_id} back to {saved_event_id}"
        )

        # Rollback in-memory session state
        # Redis already has the correct position (it was never advanced during drain)
        self.session.last_event_id = saved_event_id

        # Clear pending buffers - these events will be re-drained on next turn
        self.pending_events = []

    async def _apply_events_to_session(
        self: "MUDAgentWorker",
        events: list[MUDEvent],
        *,
        extend: bool = True,
    ) -> None:
        """Apply events to session state and conversation history.

        Shared logic for setting up turn context with events. Handles:
        1. Setting/extending worker.pending_events
        2. Setting/extending session.pending_events
        3. Updating session.last_event_time
        4. Pushing events as user turn to conversation_manager

        Args:
            events: Events to apply
            extend: If True, extend existing pending_events. If False, replace them.
        """
        if not events:
            return

        # Update worker.pending_events
        if extend:
            self.pending_events.extend(events)
        else:
            self.pending_events = events

        # Update session.pending_events
        if self.session:
            if extend:
                if not self.session.pending_events:
                    self.session.pending_events = []
                self.session.pending_events.extend(events)
            else:
                self.session.pending_events = events

            # Update last_event_time from latest event
            latest = events[-1]
            self.session.last_event_time = latest.timestamp

        # Push events as user turn to conversation history
        if self.conversation_manager:
            signature = self._compute_drain_signature(events)
            if signature and signature == self._last_conversation_signature:
                logger.info("Skipping duplicate conversation push for same event batch")
                return

            new_events: list[MUDEvent] = []
            for event in events:
                if event.event_id:
                    if event.event_id in self._conversation_event_ids:
                        continue
                    new_events.append(event)
                else:
                    new_events.append(event)

            if not new_events:
                logger.info("Skipping conversation push; all events already recorded")
                return
            room_id = None
            room_name = None
            if self.session and self.session.current_room:
                room_id = self.session.current_room.room_id
                room_name = self.session.current_room.name

            await self.conversation_manager.push_user_turn(
                events=new_events,
                world_state=self.session.world_state if self.session else None,
                room_id=room_id,
                room_name=room_name,
            )
            for event in new_events:
                if not event.event_id:
                    continue
                if event.event_id in self._conversation_event_ids:
                    continue
                if len(self._conversation_event_id_queue) >= 10000:
                    oldest = self._conversation_event_id_queue.popleft()
                    self._conversation_event_ids.discard(oldest)
                self._conversation_event_id_queue.append(event.event_id)
                self._conversation_event_ids.add(event.event_id)

            if signature:
                self._last_conversation_signature = signature

    async def _drain_to_turn(self: "MUDAgentWorker") -> list[MUDEvent]:
        """Re-drain events that arrived during Phase 1 processing.

        This method should be called after DecisionProcessor completes and before
        SpeakingProcessor or ThinkingTurnProcessor runs. It captures events that
        arrived while Phase 1 was executing, ensuring Phase 2 has the most current
        context.

        The method:
        1. Drains new events with settling (same logic as initial drain)
        2. Appends to worker.pending_events and session.pending_events
        3. Pushes new events as another user turn to conversation_manager
        4. Returns the new events for logging/inspection

        Returns:
            List of newly drained events (empty if no new events arrived).

        Example:
            ```python
            # After Phase 1 decision
            decision = worker._last_decision

            if decision.decision_type == DecisionType.SPEAK:
                # Re-drain for new events
                new_events = await worker._drain_to_turn()
                if new_events:
                    logger.info(f"Captured {len(new_events)} new events for Phase 2")

                # Run Phase 2 with combined events
                speaking_processor = SpeakingProcessor(worker)
                await speaking_processor.execute(turn_request, worker.pending_events)
            ```
        """
        # Drain new events with settling (no max_sequence_id cap for redrain)
        new_events = await self._drain_with_settle()

        if not new_events:
            logger.debug("No new events arrived during Phase 1")
            return []

        logger.info(
            f"Re-drained {len(new_events)} events for Phase 2, "
            f"sequence_ids: {[e.metadata.get('sequence_id', 0) for e in new_events]}"
        )

        # Apply events to session (extend mode)
        await self._apply_events_to_session(new_events, extend=True)
        logger.debug("Pushed re-drained events to conversation history")

        return new_events
