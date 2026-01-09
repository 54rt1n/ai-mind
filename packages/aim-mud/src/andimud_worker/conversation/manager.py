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
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field
from redis.asyncio import Redis

from aim_mud_types import MUDEvent, MUDAction, WorldState, MUDConversationEntry
from aim_mud_types.helper import _utc_now

from ..adapter import format_event, format_self_event
from aim.constants import DOC_MUD_WORLD, DOC_MUD_AGENT, LISTENER_ALL
from aim.conversation.message import ConversationMessage
from aim.utils.tokens import count_tokens

logger = logging.getLogger(__name__)


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
    ) -> MUDConversationEntry:
        """Compile events into a user turn and push to list.

        Creates a DOC_MUD_WORLD entry with formatted events and
        rich metadata about the room and actors involved.

        Args:
            events: List of MUDEvent objects to compile.
            world_state: Optional world state snapshot for context.
            room_id: Room identifier for metadata.
            room_name: Room name for metadata.

        Returns:
            The created MUDConversationEntry.
        """
        # Filter out self-speech echoes - agent already has these in assistant turn
        filtered_events = [e for e in events if not e.is_self_speech_echo()]

        # Format events as pure prose (CVM-ready, no XML wrapper)
        # Use \n\n delimiter for paragraph separation
        # Self-actions are formatted in first person, others in third person
        if filtered_events:
            formatted_parts = []
            for event in filtered_events:
                # Check if this is a self-action (from metadata flag)
                is_self = event.metadata.get("is_self_action", False)
                if is_self:
                    formatted_parts.append(format_self_event(event))
                else:
                    formatted_parts.append(format_event(event))
            content = "\n\n".join(formatted_parts)
        else:
            content = "[No events]"

        # Count tokens
        tokens = count_tokens(content)

        # Build metadata
        actors = list({e.actor for e in filtered_events if e.actor})
        event_ids = [e.event_id for e in filtered_events if e.event_id]

        # Get room info from world_state if not provided
        if not room_id and world_state and world_state.room_state:
            room_id = world_state.room_state.room_id
        if not room_name and world_state and world_state.room_state:
            room_name = world_state.room_state.name
        if not room_id and events:
            room_id = events[-1].room_id
        if not room_name and events:
            room_name = events[-1].room_name

        metadata = {
            "room_id": room_id,
            "room_name": room_name,
            "event_count": len(filtered_events),
            "actors": actors,
            "event_ids": event_ids,
        }

        entry = MUDConversationEntry(
            role="user",
            content=content,
            tokens=tokens,
            document_type=DOC_MUD_WORLD,
            conversation_id=self._get_conversation_id(),
            sequence_no=self._next_sequence_no(),
            metadata=metadata,
            speaker_id="world",
        )

        await self._push_and_trim(entry)
        return entry

    async def push_assistant_turn(
        self,
        content: str,
        think: Optional[str] = None,
        actions: Optional[list[MUDAction]] = None,
    ) -> MUDConversationEntry:
        """Push assistant response to list.

        Creates a DOC_MUD_AGENT entry with the agent's response
        and metadata about actions taken.

        Args:
            content: The agent's response content.
            think: Optional thinking/reasoning content.
            actions: List of MUDAction objects taken.

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
        # Get all entries from Redis
        raw_entries = await self.redis.lrange(self.key, 0, -1)

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

    async def flush_to_cvm(self, cvm) -> int:
        """Flush unsaved entries to CVM and mark as saved.

        Creates ConversationMessage objects for each unsaved entry
        and inserts them into the CVM. Updates the entries in Redis
        with saved=True and the assigned doc_id.

        Args:
            cvm: ConversationModel instance for persistence.

        Returns:
            Number of entries flushed.
        """
        raw_entries = await self.redis.lrange(self.key, 0, -1)

        if not raw_entries:
            return 0

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

            # Create CVM message
            msg = ConversationMessage.create(
                conversation_id=entry.conversation_id,
                sequence_no=entry.sequence_no,
                role=entry.role,
                content=entry.content,
                document_type=entry.document_type,
                speaker_id=entry.speaker_id or ("world" if entry.role == "user" else self.persona_id),
                listener_id=LISTENER_ALL,
                user_id="mud",
                persona_id=self.persona_id,
                think=entry.think,
                metadata=json.dumps(entry.metadata) if entry.metadata else "",
                timestamp=int(entry.timestamp.timestamp()),
            )

            # Insert into CVM
            doc_id = cvm.insert(msg)

            # Update entry as saved
            entry.saved = True
            entry.doc_id = doc_id

            # Update in Redis
            await self.redis.lset(self.key, i, entry.model_dump_json())
            flushed += 1

            logger.debug(f"Flushed entry {doc_id} to CVM")

        if flushed > 0:
            logger.info(f"Flushed {flushed} entries to CVM")

        return flushed

    async def _push_and_trim(self, entry: MUDConversationEntry) -> None:
        """Push entry to list and auto-trim old saved entries if over budget.

        Only trims entries where saved=True to avoid data loss.
        Entries are trimmed from oldest first.

        Args:
            entry: The entry to push.
        """
        # Push the entry
        await self.redis.rpush(self.key, entry.model_dump_json())

        # Check if we need to trim
        total_tokens = await self.get_total_tokens()

        while total_tokens > self.max_tokens:
            # Get the oldest entry
            raw = await self.redis.lindex(self.key, 0)
            if not raw:
                break

            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")

            try:
                oldest = MUDConversationEntry.model_validate_json(raw)
            except Exception:
                # Can't parse, remove it anyway
                await self.redis.lpop(self.key)
                continue

            # Only trim saved entries
            if not oldest.saved:
                # Can't trim unsaved - stop here
                logger.debug(
                    "Cannot trim unsaved entry; remaining at %d tokens (max %d)",
                    total_tokens,
                    self.max_tokens,
                )
                break

            # Remove the oldest saved entry
            await self.redis.lpop(self.key)
            total_tokens -= oldest.tokens
            logger.debug(
                f"Trimmed saved entry {oldest.doc_id}, now at {total_tokens} tokens"
            )

    async def get_total_tokens(self) -> int:
        """Sum tokens across all entries.

        Returns:
            Total token count.
        """
        raw_entries = await self.redis.lrange(self.key, 0, -1)

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
        return await self.redis.llen(self.key)

    async def clear(self) -> None:
        """Clear the conversation list.

        Used for testing or session reset.
        """
        await self.redis.delete(self.key)
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

    async def retag_unsaved_entries(self, new_conversation_id: str) -> int:
        """Re-tag all unsaved entries with new conversation_id and renumber from 0.

        Only updates entries where saved=False. Saved entries remain in CVM
        with their original conversation_id.

        Args:
            new_conversation_id: The conversation_id to apply.

        Returns:
            Number of entries re-tagged.
        """
        raw_entries = await self.redis.lrange(self.key, 0, -1)

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
            pipe = self.redis.pipeline()
            pipe.delete(self.key)
            for entry_json in updated_entries:
                pipe.rpush(self.key, entry_json)
            await pipe.execute()

        # Update internal sequence counter
        self._sequence_no = new_sequence_no

        logger.info(f"Re-tagged {retagged} unsaved entries with conversation_id: {new_conversation_id}")
        return retagged
