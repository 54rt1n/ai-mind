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

from aim_mud_types import MUDEvent, MUDConversationEntry, EventType
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

        DEPRECATED: This method is replaced by get_new_conversation_entries().
        The mediator now compiles events into conversation entries with embeddings.
        Only retained for PENDING status handling (matching action_id echoes).

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

        DEPRECATED: This method is replaced by get_new_conversation_entries().
        The mediator now compiles events into conversation entries with embeddings.
        Use get_new_conversation_entries() + collapse_consecutive_entries() instead.

        Originally from worker.py lines 736-777

        Drains events, waits settle_seconds, drains again.
        Repeats until a drain returns zero events. This allows
        cascading events (e.g., someone entering a room and immediately
        speaking) to be batched together.

        Args:
            max_sequence_id: Optional cap for ALL drains. This prevents
                capturing events from future turns during the settle window.

        Returns:
            All accumulated events from multiple drains.
        """
        settle_time = self.config.event_settle_seconds
        final_settle_time = settle_time // 3
        all_events: list[MUDEvent] = []
        drain_count = 0
        empty_after_events = 0

        while True:
            # Cap ALL drains at max_sequence_id to prevent cross-turn event leakage
            events = await self.drain_events(timeout=0, max_sequence_id=max_sequence_id)
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

    async def _push_events_to_conversation(
        self: "MUDAgentWorker",
        events: list[MUDEvent],
    ) -> None:
        """Push events to conversation history.

        DEPRECATED: This method is replaced by mediator event compilation.
        The mediator now compiles events into MUDConversationEntry objects
        with pre-computed embeddings and pushes them directly to the
        conversation list. Use get_new_conversation_entries() to read them.

        Only retained for PENDING status handling (pushing action echo events).

        Simplified method that:
        1. Updates session.last_event_time
        2. Pushes events as user turn to conversation_manager

        Args:
            events: Events to push to conversation history
        """
        if not events:
            return

        # Update session.last_event_time from latest event
        if self.session:
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
                last_event_id=new_events[-1].event_id if new_events else None,
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

        DEPRECATED: This method is replaced by get_new_conversation_entries().
        The mediator now compiles events into conversation entries with embeddings.
        Use get_new_conversation_entries() + collapse_consecutive_entries() instead.

        This method should be called after DecisionProcessor completes and before
        SpeakingProcessor or ThinkingTurnProcessor runs. It captures events that
        arrived while Phase 1 was executing, ensuring Phase 2 has the most current
        context.

        The method:
        1. Drains new events with settling (same logic as initial drain)
        2. Pushes new events to conversation history
        3. Returns the new events to be combined with original events

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
                await speaking_processor.execute(turn_request, events + new_events)
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

        # Push events to conversation history
        await self._push_events_to_conversation(new_events)
        logger.debug("Pushed re-drained events to conversation history")

        return new_events

    # =========================================================================
    # MUDLOGIC V2: Conversation Entry Reading (replaces event draining)
    # =========================================================================

    async def get_new_conversation_entries(
        self: "MUDAgentWorker",
        settle: bool = False,
    ) -> list["MUDConversationEntry"]:
        """Read new conversation entries with optional settling.

        Reads entries from conversation list that were compiled by mediator.
        Entries already have embeddings computed.

        Args:
            settle: If True, wait for entries to stop arriving before returning.
                   Uses event_settle_seconds for timing between reads.

        Returns:
            List of new MUDConversationEntry objects.
        """
        if not settle:
            # Fast path - read once and return
            return await self._read_entries_once()

        # Settling path - wait for entries to stop arriving
        settle_time = self.config.event_settle_seconds
        final_settle_time = settle_time / 3
        all_entries: list[MUDConversationEntry] = []
        drain_count = 0
        empty_after_entries = 0

        while True:
            entries = await self._read_entries_once()
            drain_count += 1

            if not entries:
                if not all_entries:
                    # No new entries - cascade has settled
                    if drain_count == 1:
                        logger.warning(
                            "First read returned 0 entries - turn assigned but list empty?"
                        )
                    break

                empty_after_entries += 1
                if empty_after_entries >= 2:
                    logger.info(
                        "Entry cascade settled after %.1fs with %d total entries",
                        settle_time,
                        len(all_entries),
                    )
                    break

                logger.info(
                    "Read 0 entries after %d total, waiting %.1fs for stragglers",
                    len(all_entries),
                    final_settle_time,
                )
                await asyncio.sleep(final_settle_time)
                continue

            all_entries.extend(entries)
            empty_after_entries = 0
            logger.info(
                "Read %d entries (total %d), waiting %.1fs for more",
                len(entries),
                len(all_entries),
                settle_time,
            )
            await asyncio.sleep(settle_time)

        return all_entries

    async def _read_entries_once(
        self: "MUDAgentWorker",
    ) -> list["MUDConversationEntry"]:
        """Read entries once without settling.

        Internal helper for get_new_conversation_entries().

        Returns:
            List of new MUDConversationEntry objects.
        """
        from aim_mud_types.client import RedisMUDClient
        from aim_mud_types import MUDConversationEntry

        client = RedisMUDClient(self.redis)

        # Get current position
        last_read = self.session.last_conversation_index

        # Fetch new entries starting from last_read (read all available)
        raw_entries = await client.get_conversation_entries(
            self.config.agent_id,
            start=last_read,
            end=-1,
        )

        entries: list[MUDConversationEntry] = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                entry = MUDConversationEntry.model_validate_json(raw)
                entries.append(entry)
            except Exception as e:
                logger.warning(f"Failed to parse conversation entry: {e}")

        # Update tracking position
        if entries:
            new_position = last_read + len(entries)
            self.session.last_conversation_index = new_position
            logger.debug(f"Read {len(entries)} entries, position now {new_position}")

        return entries

    def collapse_consecutive_entries(
        self: "MUDAgentWorker",
        entries: list["MUDConversationEntry"],
    ) -> list["MUDConversationEntry"]:
        """Collapse consecutive mud-world entries by same speaker_id.

        Groups consecutive DOC_MUD_WORLD entries with same speaker_id
        into single entries for cleaner LLM context display.

        Args:
            entries: List of conversation entries from mediator.

        Returns:
            Collapsed list (may be shorter than input).
        """
        from aim.constants import DOC_MUD_WORLD
        from aim_mud_types import MUDConversationEntry

        if not entries:
            return []

        collapsed: list[MUDConversationEntry] = []
        current_group: list[MUDConversationEntry] = []

        for entry in entries:
            # Only collapse DOC_MUD_WORLD entries
            if entry.document_type != DOC_MUD_WORLD:
                # Flush current group
                if current_group:
                    collapsed.append(self._merge_entries(current_group))
                    current_group = []
                collapsed.append(entry)
                continue

            # Check if we can add to current group
            if current_group:
                last = current_group[-1]
                # Same speaker_id and same document type?
                if last.speaker_id == entry.speaker_id:
                    current_group.append(entry)
                    continue
                else:
                    # Different speaker - flush and start new
                    collapsed.append(self._merge_entries(current_group))
                    current_group = []

            current_group.append(entry)

        # Flush final group
        if current_group:
            collapsed.append(self._merge_entries(current_group))

        return collapsed

    def _merge_entries(
        self: "MUDAgentWorker",
        entries: list["MUDConversationEntry"],
    ) -> "MUDConversationEntry":
        """Merge multiple entries into one.

        Args:
            entries: List of entries to merge (must be non-empty).

        Returns:
            Single merged MUDConversationEntry.
        """
        from aim_mud_types import MUDConversationEntry

        if len(entries) == 1:
            return entries[0]

        first = entries[0]
        last = entries[-1]

        # Join content
        content = "\n\n".join(e.content for e in entries)

        # Sum tokens
        total_tokens = sum(e.tokens for e in entries)

        # Merge metadata
        merged_metadata = dict(first.metadata)
        merged_metadata["event_count"] = sum(
            e.metadata.get("event_count", 1) for e in entries
        )
        merged_metadata["merged_entry_count"] = len(entries)

        # Collect all event IDs
        all_event_ids: list[str] = []
        for e in entries:
            ids = e.metadata.get("event_ids", [])
            if isinstance(ids, list):
                all_event_ids.extend(ids)
        merged_metadata["event_ids"] = all_event_ids

        # Use first entry's embedding (representative)
        # Could average embeddings, but first is simpler and sufficient
        embedding = first.embedding

        return MUDConversationEntry(
            role=first.role,
            content=content,
            tokens=total_tokens,
            document_type=first.document_type,
            conversation_id=first.conversation_id,
            sequence_no=first.sequence_no,
            metadata=merged_metadata,
            speaker_id=first.speaker_id,
            timestamp=first.timestamp,
            last_event_id=last.last_event_id,
            embedding=embedding,
            saved=False,
            skip_save=first.skip_save,
        )

    # =========================================================================
    # Legacy event draining (kept for PENDING status handling)
    # =========================================================================

    async def _cleanup_drained_events(self: "MUDAgentWorker") -> bool:
        """Delete the event stream if all events have been drained to conversation.

        Atomically checks if the stream's last-generated-id matches the
        conversation's last entry's last_event_id. If they match, deletes
        the stream key. This cleans up processed events and prevents stale
        data on restarts.

        Should be called after a turn completes successfully.

        Returns:
            True if the stream was deleted (fully drained).
            False if stream has newer events or doesn't exist.
        """
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)

        deleted = await client.delete_agent_events_if_drained(
            self.config.agent_id,
            stream_key=self.config.agent_stream,
        )

        if deleted:
            logger.info("Cleaned up event stream - all events drained to conversation")

        return deleted
