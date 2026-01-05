# aim/app/mud/worker/events.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Event draining for the MUD worker.

Handles consuming events from Redis streams with settling logic.
Extracted from worker.py lines 646-777
"""

import asyncio
import json
import logging
from typing import TYPE_CHECKING

import redis.asyncio as redis

from ..session import MUDEvent


if TYPE_CHECKING:
    from .main import MUDAgentWorker


logger = logging.getLogger(__name__)


class EventsMixin:
    """Mixin for event draining methods.

    These methods are mixed into MUDAgentWorker in main.py.
    """

    async def drain_events(self: "MUDAgentWorker", timeout: float) -> list[MUDEvent]:
        """Block until events arrive on agent's stream.

        Originally from worker.py lines 646-735

        Events are already enriched by mediator with room state.

        Args:
            timeout: Maximum seconds to block waiting for events.

        Returns:
            List of MUDEvent objects parsed from the stream.
        """
        events: list[MUDEvent] = []
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
                        event = MUDEvent.from_dict(enriched)
                        event.event_id = msg_id
                        events.append(event)

                        logger.debug(
                            f"Parsed event {msg_id}: {event.event_type.value} "
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

        if self.session and self.session.last_event_id:
            await self._update_agent_profile(last_event_id=self.session.last_event_id)

        return events

    async def _drain_with_settle(self: "MUDAgentWorker") -> list[MUDEvent]:
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
            events = await self.drain_events(timeout=0)
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
