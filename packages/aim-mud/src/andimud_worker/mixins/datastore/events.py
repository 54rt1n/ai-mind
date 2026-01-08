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

import redis.asyncio as redis

from aim_mud_types import MUDEvent

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

        def _parse_result(result) -> None:
            for _stream_name, messages in result:
                for msg_id, data in messages:
                    # Update last event ID for resumption
                    # msg_id may be bytes or str depending on Redis client config
                    if isinstance(msg_id, bytes):
                        msg_id = msg_id.decode("utf-8")
                    self.session.last_event_id = msg_id

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
                            continue

                        # Check for self-action flag (set by mediator for actor's own events)
                        is_self_action = enriched.pop("is_self_action", False)

                        event = MUDEvent.from_dict(enriched)
                        event.event_id = msg_id
                        event.metadata["sequence_id"] = sequence_id

                        if is_self_action:
                            # Mark in metadata so conversation manager can format in first person
                            event.metadata["is_self_action"] = True
                            # Store in session for guidance injection (don't process as regular event)
                            # Skip accumulation for @agent turns (accumulate_self_actions=False)
                            if accumulate_self_actions and self.session:
                                self.session.pending_self_actions.append(event)
                                logger.debug(
                                    f"Stored self-action event (seq={sequence_id}): {event.event_type.value}"
                                )
                            else:
                                logger.debug(
                                    f"Skipped self-action event (seq={sequence_id}): {event.event_type.value}"
                                )
                        else:
                            # Append as tuple with sequence_id for sorting
                            events_unsorted.append((sequence_id, event))
                            logger.debug(
                                f"Parsed event (seq={sequence_id}): {event.event_type.value} "
                                f"from {event.actor}"
                            )

                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.error(f"Failed to parse event {msg_id}: {e}")
                        continue

        # Snapshot the max stream id at drain start to avoid chasing new events
        try:
            info = await self.redis.xinfo_stream(self.config.agent_stream)
            max_id = info.get("last-generated-id") or info.get(b"last-generated-id")
            if isinstance(max_id, bytes):
                max_id = max_id.decode("utf-8")
        except redis.RedisError as e:
            logger.error(f"Redis error in drain_events (xinfo): {e}")
            return []

        if not max_id or max_id == "0":
            return []

        # Drain events up to snapshot max_id
        start_id = self.session.last_event_id or "0"
        min_id = f"({start_id}" if start_id != "0" else "-"
        while True:
            try:
                result = await self.redis.xrange(
                    self.config.agent_stream,
                    min=min_id,
                    max=max_id,
                    count=100,
                )
            except redis.RedisError as e:
                logger.error(f"Redis error in drain_events (xrange): {e}")
                break

            if not result:
                break

            _parse_result([(self.config.agent_stream, result)])
            last_msg_id = self.session.last_event_id
            if not last_msg_id or last_msg_id == max_id:
                break
            min_id = f"({last_msg_id}"

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

        Returns:
            All accumulated events from multiple drains.
        """
        settle_time = self.config.event_settle_seconds
        all_events: list[MUDEvent] = []
        drain_count = 0

        while True:
            events = await self.drain_events(timeout=0, max_sequence_id=max_sequence_id)
            drain_count += 1
            if not events:
                # No new events - cascade has settled
                if all_events:
                    logger.info(
                        "Event cascade settled after %.1fs with %d total events",
                        settle_time,
                        len(all_events),
                    )
                elif drain_count == 1:
                    logger.warning(
                        "First drain returned 0 events - turn assigned but stream empty?"
                    )
                break

            all_events.extend(events)
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
        self.session.pending_self_actions = []
