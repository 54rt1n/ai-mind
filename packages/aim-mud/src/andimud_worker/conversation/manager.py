# aim/app/mud/conversation.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Redis-backed conversation list for MUD agent turns.

This module provides a single source of truth for both LLM context
and CVM (Conversation Vector Memory) persistence. Each agent has
its own conversation list in Redis that:

1. Stores user turns (world events) and assistant turns (agent responses)
2. Pre-counts tokens for efficient budget management
3. Tracks which entries have been flushed to CVM
4. Auto-trims old saved entries when over token budget
5. Survives restarts (persisted in Redis)

The design follows the plan in CODEX_PHASE_2_4_MUD_Retrieval.md.
"""

import json
import logging
import secrets
from typing import Literal, Optional

from pydantic import BaseModel, Field
from redis.asyncio import Redis

from aim_mud_types import MUDEvent, MUDAction, WorldState, MUDConversationEntry, EventType
from aim_mud_types.helper import _utc_now

from ..adapter import format_event, format_self_event
from aim.constants import (
    DOC_MUD_WORLD,
    DOC_MUD_AGENT,
    DOC_MUD_ACTION,
    DOC_CODE_ACTION,
    DOC_CODE_FILE,
    LISTENER_ALL,
)
from aim.conversation.message import ConversationMessage
from aim.utils.tokens import count_tokens

logger = logging.getLogger(__name__)

CODE_EVENT_MAX_CHARS = 13000
CODE_EVENT_SAVE_HEAD = 256
CODE_EVENT_SAVE_TAIL = 256


def _truncate_head(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    return text[:max_chars] if len(text) > max_chars else text


def _truncate_head_tail(text: str, head: int, tail: int) -> str:
    if head <= 0 and tail <= 0:
        return ""
    if len(text) <= head + tail:
        return text
    return f"{text[:head]}{text[-tail:]}"


def _compare_stream_ids(a: str, b: str) -> int:
    """Compare two Redis stream IDs numerically.

    Redis stream IDs are in format "timestamp-sequence" (e.g., "1704096000000-42").
    Lexicographic comparison doesn't work correctly for sequences with different
    digit counts ("5" > "42" lexicographically). This function compares them
    by parsing into (timestamp, sequence) tuples.

    Args:
        a: First stream ID
        b: Second stream ID

    Returns:
        -1 if a < b, 0 if a == b, 1 if a > b
    """
    try:
        a_ts, a_seq = a.split("-", 1)
        b_ts, b_seq = b.split("-", 1)
        a_tuple = (int(a_ts), int(a_seq))
        b_tuple = (int(b_ts), int(b_seq))
        if a_tuple < b_tuple:
            return -1
        elif a_tuple > b_tuple:
            return 1
        return 0
    except (ValueError, AttributeError):
        # Fallback to lexicographic if parsing fails
        if a < b:
            return -1
        elif a > b:
            return 1
        return 0


class MUDConversationManager:
    """Manages Redis conversation list for an agent.

    The conversation list is the single source of truth for:
    - LLM context (via get_history with token budget)
    - CVM persistence (via flush_to_cvm)

    Entries are stored as JSON in a Redis list, with newest at the end.
    Auto-trimming removes old saved entries when over the token budget.

    Attributes:
        redis: Async Redis client.
        agent_id: Unique identifier for this agent.
        persona_id: The persona ID for this agent.
        max_tokens: Maximum total tokens to retain in the list.
        key: Redis key for this agent's conversation list.
    """

    def __init__(
        self,
        redis: Redis,
        agent_id: str,
        persona_id: str,
        max_tokens: int = 50000,
    ):
        """Initialize the conversation manager.

        Args:
            redis: Async Redis client.
            agent_id: Unique identifier for this agent.
            persona_id: The persona ID for this agent.
            max_tokens: Maximum total tokens before auto-trimming.
        """
        self.redis = redis
        self.agent_id = agent_id
        self.persona_id = persona_id
        self.max_tokens = max_tokens
        self.key = f"mud:agent:{agent_id}:conversation"
        self._sequence_no = 0
        self._conversation_id: Optional[str] = None

    def _get_conversation_id(self) -> str:
        """Get or create conversation_id for grouping turns.

        Creates a new conversation_id on first call, which persists
        for the lifetime of this manager instance. Format is:
        andimud_{timestamp_ms}_{random_hex}

        Returns:
            The conversation_id string.
        """
        if not self._conversation_id:
            ts = int(_utc_now().timestamp() * 1000)
            suffix = secrets.token_hex(4)
            self._conversation_id = f"andimud_{ts}_{suffix}"
        return self._conversation_id

    @property
    def conversation_id(self) -> str:
        """Get or create conversation_id.

        Public property that provides natural access to the conversation_id.
        Creates one lazily on first access if not already set.

        Returns:
            The conversation_id string.
        """
        return self._get_conversation_id()

    def _next_sequence_no(self) -> int:
        """Get and increment the sequence number.

        Returns:
            The next sequence number.
        """
        seq = self._sequence_no
        self._sequence_no += 1
        return seq

    async def push_user_turn(
        self,
        events: list[MUDEvent],
        world_state: Optional[WorldState] = None,
        room_id: Optional[str] = None,
        room_name: Optional[str] = None,
        last_event_id: Optional[str] = None,
    ) -> MUDConversationEntry:
        """Compile events into a user turn and push to list.

        DEPRECATED: Event compilation is now handled by the mediator.
        The mediator compiles events into MUDConversationEntry objects with
        pre-computed embeddings and pushes them directly to the conversation list.
        This method is retained for backward compatibility with PENDING status
        handling and legacy code paths.

        Creates a DOC_MUD_WORLD or DOC_MUD_ACTION entry per consecutive
        actor group with rich metadata about the room and actors involved.

        Args:
            events: List of MUDEvent objects to compile.
            world_state: Optional world state snapshot for context.
            room_id: Room identifier for metadata.
            room_name: Room name for metadata.
            last_event_id: Fallback last_event_id if group has no events with IDs.

        Returns:
            The created MUDConversationEntry.
        """
        # Phase 5: Self-action events are now the single source of truth for agent speech.
        # No longer filtering self-speech echoes - they create DOC_MUD_ACTION entries.
        filtered_events = events
        last_entry: Optional[MUDConversationEntry] = None
        group_events: list[MUDEvent] = []
        group_actor_key: Optional[str] = None
        group_is_self: Optional[bool] = None

        async def flush_group() -> None:
            nonlocal last_entry, group_events, group_actor_key, group_is_self
            if not group_events:
                return

            last_event = group_events[-1]

            # Compute last_event_id from this group's events
            group_last_event_id = group_events[-1].event_id if group_events else last_event_id
            entry_room_id = last_event.room_id
            entry_room_name = last_event.room_name
            if not entry_room_id and world_state and world_state.room_state:
                entry_room_id = world_state.room_state.room_id
            if not entry_room_name and world_state and world_state.room_state:
                entry_room_name = world_state.room_state.name

            actors: list[str] = []
            actor_ids: list[str] = []
            event_ids: list[str] = []
            event_types: list[str] = []
            targets: list[str] = []
            target_ids: list[str] = []
            event_metadatas: list[dict] = []
            content_parts: list[str] = []

            for event in group_events:
                if group_is_self:
                    content_parts.append(event.content or "[No content]")
                else:
                    content_parts.append(format_event(event) or "[No content]")

                if event.actor and event.actor not in actors:
                    actors.append(event.actor)
                if event.actor_id and event.actor_id not in actor_ids:
                    actor_ids.append(event.actor_id)
                if event.event_id:
                    event_ids.append(event.event_id)
                event_types.append(event.event_type.value)
                if event.target:
                    targets.append(event.target)
                if event.target_id:
                    target_ids.append(event.target_id)
                event_metadatas.append(event.metadata)

            content = "\n\n".join(content_parts)
            tokens = count_tokens(content)

            # Determine document type
            if group_is_self:
                # Self-speech/narrative → DOC_MUD_AGENT, other self-actions → DOC_MUD_ACTION
                speech_types = {EventType.SPEECH.value, EventType.NARRATIVE.value}
                if any(et in speech_types for et in event_types):
                    doc_type = DOC_MUD_AGENT
                else:
                    doc_type = DOC_MUD_ACTION
            else:
                doc_type = DOC_MUD_WORLD

            metadata = {
                "room_id": entry_room_id or room_id,
                "room_name": entry_room_name or room_name,
                "event_count": len(group_events),
                "actors": actors,
                "actor_ids": actor_ids,
                "event_ids": event_ids,
                "event_type": event_types[-1],
                "event_types": event_types,
                "actor": last_event.actor,
                "actor_id": last_event.actor_id,
                "target": last_event.target,
                "target_id": last_event.target_id,
                "targets": targets,
                "target_ids": target_ids,
                "event_metadata": event_metadatas[0] if len(event_metadatas) == 1 else event_metadatas,
            }

            entry = MUDConversationEntry(
                role="user",
                content=content,
                tokens=tokens,
                document_type=doc_type,
                conversation_id=self._get_conversation_id(),
                sequence_no=self._next_sequence_no(),
                metadata=metadata,
                speaker_id="world",
                timestamp=last_event.timestamp,
                last_event_id=group_last_event_id,
            )
            await self._push_and_trim(entry)
            last_entry = entry

            group_events = []
            group_actor_key = None
            group_is_self = None

        for event in filtered_events:
            if event.event_type in (EventType.CODE_ACTION, EventType.CODE_FILE):
                await flush_group()
                raw_content = event.content or ""
                if not raw_content:
                    raw_content = "(no output)"
                content = _truncate_head(raw_content, CODE_EVENT_MAX_CHARS)
                tokens = count_tokens(content)
                doc_type = DOC_CODE_ACTION if event.event_type == EventType.CODE_ACTION else DOC_CODE_FILE
                metadata = {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "room_id": event.room_id,
                    "room_name": event.room_name,
                    "actor": event.actor,
                    "actor_id": event.actor_id,
                    "target": event.target,
                    "target_id": event.target_id,
                    "event_metadata": event.metadata,
                }
                entry = MUDConversationEntry(
                    role="user",
                    content=content,
                    tokens=tokens,
                    document_type=doc_type,
                    conversation_id=self._get_conversation_id(),
                    sequence_no=self._next_sequence_no(),
                    metadata=metadata,
                    speaker_id="code",
                    timestamp=event.timestamp,
                    last_event_id=event.event_id or last_event_id,
                )
                await self._push_and_trim(entry)
                last_entry = entry
                continue

            # Don't group narrative events - each gets its own entry
            if event.event_type == EventType.NARRATIVE:
                await flush_group()  # Flush any pending non-narrative events

                is_self = event.metadata.get("is_self_action", False)
                content = format_event(event) or "[No content]"
                tokens = count_tokens(content)
                metadata = {
                    "room_id": event.room_id or room_id,
                    "room_name": event.room_name or room_name,
                    "event_count": 1,
                    "actors": [event.actor] if event.actor else [],
                    "actor_ids": [event.actor_id] if event.actor_id else [],
                    "event_ids": [event.event_id] if event.event_id else [],
                    "event_type": event.event_type.value,
                    "event_types": [event.event_type.value],
                    "actor": event.actor,
                    "actor_id": event.actor_id,
                    "target": event.target,
                    "target_id": event.target_id,
                    "targets": [event.target] if event.target else [],
                    "target_ids": [event.target_id] if event.target_id else [],
                    "event_metadata": event.metadata,
                }
                entry = MUDConversationEntry(
                    role="user",
                    content=content,
                    tokens=tokens,
                    document_type=DOC_MUD_AGENT if is_self else DOC_MUD_WORLD,
                    conversation_id=self._get_conversation_id(),
                    sequence_no=self._next_sequence_no(),
                    metadata=metadata,
                    speaker_id=self.agent_id if is_self else "world",
                    timestamp=event.timestamp,
                    last_event_id=event.event_id or last_event_id,
                )
                await self._push_and_trim(entry)
                last_entry = entry
                continue

            is_self = event.metadata.get("is_self_action", False)
            actor_key = event.actor_id or event.actor
            if group_events and (group_actor_key != actor_key or group_is_self != is_self):
                await flush_group()

            if not group_events:
                group_actor_key = actor_key
                group_is_self = is_self
            group_events.append(event)

        await flush_group()

        if last_entry is None:
            # No events (or all self-speech filtered) -> placeholder entry
            last_entry = MUDConversationEntry(
                role="user",
                content="[No events]",
                tokens=count_tokens("[No events]"),
                document_type=DOC_MUD_WORLD,
                conversation_id=self._get_conversation_id(),
                sequence_no=self._next_sequence_no(),
                metadata={"event_count": 0},
                speaker_id="world",
                last_event_id=last_event_id,
            )
            await self._push_and_trim(last_entry)

        return last_entry

    async def push_assistant_turn(
        self,
        content: str,
        think: Optional[str] = None,
        actions: Optional[list[MUDAction]] = None,
        skip_save: bool = False,
    ) -> MUDConversationEntry:
        """Push assistant response to list.

        Creates a DOC_MUD_AGENT entry with the agent's response
        and metadata about actions taken.

        Args:
            content: The agent's response content.
            think: Optional thinking/reasoning content.
            actions: List of MUDAction objects taken.
            skip_save: If True, this entry will never be persisted to CVM.

        Returns:
            The created MUDConversationEntry.
        """
        if actions is None:
            actions = []

        # Count tokens (include think content in count)
        full_content = content
        if think:
            full_content = f"<think>{think}</think>\n{content}"
        tokens = count_tokens(full_content)

        # Build metadata
        action_commands = [a.to_command() for a in actions if a.to_command()]
        metadata = {
            "actions": action_commands,
            "action_count": len(actions),
        }

        entry = MUDConversationEntry(
            role="assistant",
            content=content,
            tokens=tokens,
            document_type=DOC_MUD_AGENT,
            conversation_id=self._get_conversation_id(),
            sequence_no=self._next_sequence_no(),
            metadata=metadata,
            think=think,
            speaker_id=self.persona_id,
            skip_save=skip_save,
        )

        await self._push_and_trim(entry)
        return entry

    async def get_history(self, token_budget: int) -> list[MUDConversationEntry]:
        """Get recent entries within token budget.

        Returns entries in chronological order, selecting from newest
        to oldest until the budget is exceeded.

        Args:
            token_budget: Maximum total tokens to include.

        Returns:
            List of entries in chronological order.
        """
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)

        # Get all entries from Redis
        raw_entries = await client.get_conversation_entries(self.agent_id, 0, -1)

        if not raw_entries:
            return []

        # Parse entries
        entries: list[MUDConversationEntry] = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                entry = MUDConversationEntry.model_validate_json(raw)
                entries.append(entry)
            except Exception as e:
                logger.warning(f"Failed to parse conversation entry: {e}")
                continue

        # Select from newest to oldest within budget
        selected: list[MUDConversationEntry] = []
        total_tokens = 0

        for entry in reversed(entries):
            if total_tokens + entry.tokens > token_budget:
                break
            selected.append(entry)
            total_tokens += entry.tokens

        # Return in chronological order
        return list(reversed(selected))

    async def get_recent_events(self, count: int = 10) -> list[MUDConversationEntry]:
        """Get N most recent conversation entries (not token-budget limited).

        Unlike get_history() which uses token budget, this provides fixed-count
        access for debugging commands like @last.

        Args:
            count: Maximum number of entries to return.

        Returns:
            List of entries in chronological order (oldest first).
        """
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)

        # Get last N entries from Redis
        raw_entries = await client.get_conversation_entries(self.agent_id, -count, -1)

        if not raw_entries:
            return []

        # Parse entries
        entries: list[MUDConversationEntry] = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                entry = MUDConversationEntry.model_validate_json(raw)
                entries.append(entry)
            except Exception as e:
                logger.warning(f"Failed to parse conversation entry: {e}")
                continue

        return entries

    async def flush_to_cvm(self, cvm) -> int:
        """Flush unsaved entries to CVM and mark as saved.

        Temporarily loads vectorizer for embedding computation.

        Creates ConversationMessage objects for each unsaved entry
        and inserts them into the CVM. Updates the entries in Redis
        with saved=True and the assigned doc_id. Entries marked
        skip_save=True are marked saved without persistence.

        Args:
            cvm: ConversationModel instance for persistence.

        Returns:
            Number of entries flushed.
        """
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)
        raw_entries = await client.get_conversation_entries(self.agent_id, 0, -1)

        if not raw_entries:
            return 0

        # Load vectorizer for batch write
        cvm.load_vectorizer()

        try:
            flushed = 0

            for i, raw in enumerate(raw_entries):
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")

                try:
                    entry = MUDConversationEntry.model_validate_json(raw)
                except Exception as e:
                    logger.warning(f"Failed to parse entry for flush: {e}")
                    continue

                if entry.saved:
                    continue

                if entry.skip_save:
                    entry.saved = True
                    await client.set_conversation_entry(
                        self.agent_id,
                        i,
                        entry.model_dump_json(),
                    )
                    continue

                persist_content = entry.content
                if entry.document_type in (DOC_CODE_ACTION, DOC_CODE_FILE):
                    persist_content = _truncate_head_tail(
                        persist_content,
                        CODE_EVENT_SAVE_HEAD,
                        CODE_EVENT_SAVE_TAIL,
                    )

                # Create CVM message
                msg = ConversationMessage.create(
                    conversation_id=entry.conversation_id,
                    sequence_no=entry.sequence_no,
                    role=entry.role,
                    content=persist_content,
                    document_type=entry.document_type,
                    speaker_id=entry.speaker_id or ("world" if entry.role == "user" else self.persona_id),
                    listener_id=LISTENER_ALL,
                    user_id=self._resolve_user_id(entry),
                    persona_id=self.persona_id,
                    think=entry.think,
                    metadata=json.dumps(entry.metadata) if entry.metadata else "",
                    timestamp=int(entry.timestamp.timestamp()),
                )

                # Insert WITHOUT embedding parameter - let add_document compute all
                cvm.insert(msg)

                # Update entry as saved (doc_id is on the message, not returned by insert)
                entry.saved = True
                entry.doc_id = msg.doc_id

                # Update in Redis
                await client.set_conversation_entry(
                    self.agent_id,
                    i,
                    entry.model_dump_json(),
                )
                flushed += 1

                logger.debug(f"Flushed entry {msg.doc_id} to CVM")

            if flushed > 0:
                logger.info(f"Flushed {flushed} entries to CVM")

            return flushed

        finally:
            # Always release vectorizer, even on error
            cvm.release_vectorizer()

    @staticmethod
    def _resolve_user_id(entry: MUDConversationEntry) -> str:
        """Resolve user_id for persistence based on entry metadata."""
        if entry.role != "user":
            return "mud"

        if entry.document_type in (DOC_MUD_ACTION, DOC_CODE_ACTION, DOC_CODE_FILE):
            return "mud"

        metadata = entry.metadata or {}
        actor = metadata.get("actor") or metadata.get("actor_id")
        return actor or "mud"

    async def _push_and_trim(self, entry: MUDConversationEntry) -> None:
        """Push entry to list and auto-trim old saved entries if over budget.

        Only trims entries where saved=True to avoid data loss.
        Entries are trimmed from oldest first.

        Args:
            entry: The entry to push.
        """
        # Push the entry
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)
        await client.append_conversation_entry(
            self.agent_id,
            entry.model_dump_json(),
        )

        await self._trim_saved_entries()

    async def _trim_saved_entries(self) -> None:
        """Trim oldest saved entries until within token budget."""
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)
        total_tokens = await self.get_total_tokens()

        while total_tokens > self.max_tokens:
            raw = await client.get_conversation_entry(self.agent_id, 0)
            if not raw:
                break

            try:
                oldest = MUDConversationEntry.model_validate_json(raw)
            except Exception:
                await client.pop_conversation_entry(self.agent_id)
                continue

            if not oldest.saved:
                logger.debug(
                    "Cannot trim unsaved entry; remaining at %d tokens (max %d)",
                    total_tokens,
                    self.max_tokens,
                )
                break

            await client.pop_conversation_entry(self.agent_id)
            total_tokens -= oldest.tokens
            logger.debug(
                f"Trimmed saved entry {oldest.doc_id}, now at {total_tokens} tokens"
            )

    async def get_total_tokens(self) -> int:
        """Sum tokens across all entries.

        Returns:
            Total token count.
        """
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)
        raw_entries = await client.get_conversation_entries(self.agent_id, 0, -1)

        total = 0
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                entry = MUDConversationEntry.model_validate_json(raw)
                total += entry.tokens
            except Exception:
                continue

        return total

    async def get_entry_count(self) -> int:
        """Get the number of entries in the conversation list.

        Returns:
            Number of entries.
        """
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)
        return await client.get_conversation_length(self.agent_id)

    async def clear(self) -> None:
        """Clear the conversation list.

        Used for testing or session reset.
        """
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)
        await client.delete_conversation(self.agent_id)
        self._sequence_no = 0
        self._conversation_id = None
        logger.info(f"Cleared conversation list for agent {self.agent_id}")

    def get_current_conversation_id(self) -> Optional[str]:
        """Get the current conversation_id without creating one.

        Returns:
            Current conversation_id or None if not set.
        """
        return self._conversation_id

    def set_conversation_id(self, conversation_id: str) -> None:
        """Set the conversation_id for this manager.

        Updates the instance variable. Does not re-tag existing entries.
        Call retag_unsaved_entries() separately to update entries.

        Args:
            conversation_id: The new conversation_id to use.
        """
        self._conversation_id = conversation_id
        logger.info(f"Set conversation_id to {conversation_id}")

    async def get_last_event_id(self) -> Optional[str]:
        """Get the maximum last_event_id across all conversation entries.

        Scans all entries and returns the maximum event_id. Event IDs are
        Redis stream IDs in format "timestamp-sequence" which can be compared
        lexicographically.

        Returns:
            The maximum last_event_id across all entries, or None if no entries.
        """
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)

        raw_entries = await client.get_conversation_entries(self.agent_id, 0, -1)
        if not raw_entries:
            return None

        max_event_id: Optional[str] = None
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                entry = MUDConversationEntry.model_validate_json(raw)
                if entry.last_event_id:
                    if max_event_id is None or _compare_stream_ids(entry.last_event_id, max_event_id) > 0:
                        max_event_id = entry.last_event_id
            except Exception:
                continue

        return max_event_id

    async def retag_unsaved_entries(self, new_conversation_id: str) -> int:
        """Re-tag all unsaved entries with new conversation_id and renumber from 0.

        Only updates entries where saved=False. Saved entries remain in CVM
        with their original conversation_id.

        Args:
            new_conversation_id: The conversation_id to apply.

        Returns:
            Number of entries re-tagged.
        """
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)
        raw_entries = await client.get_conversation_entries(self.agent_id, 0, -1)

        if not raw_entries:
            return 0

        retagged = 0
        new_sequence_no = 0
        updated_entries = []

        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")

            try:
                entry = MUDConversationEntry.model_validate_json(raw)
            except Exception as e:
                logger.warning(f"Failed to parse entry for retag: {e}")
                updated_entries.append(raw)  # Keep as-is
                continue

            # Only update unsaved entries
            if not entry.saved:
                entry.conversation_id = new_conversation_id
                entry.sequence_no = new_sequence_no
                new_sequence_no += 1
                retagged += 1

            updated_entries.append(entry.model_dump_json())

        # Replace entire list atomically
        if updated_entries:
            await client.replace_conversation_entries(
                self.agent_id,
                updated_entries,
            )

        # Update internal sequence counter
        self._sequence_no = new_sequence_no

        logger.info(f"Re-tagged {retagged} unsaved entries with conversation_id: {new_conversation_id}")
        return retagged
