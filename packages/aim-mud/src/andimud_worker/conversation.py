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

from aim_mud_types import MUDEvent, MUDAction, WorldState

from .adapter import format_event
from aim.constants import DOC_MUD_WORLD, DOC_MUD_AGENT, LISTENER_ALL
from aim.conversation.message import ConversationMessage
from aim.utils.tokens import count_tokens

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


class MUDConversationEntry(BaseModel):
    """Single entry in the Redis conversation list.

    Each entry represents either a user turn (world events compiled into
    a single document) or an assistant turn (agent response with actions).

    Attributes:
        role: Either "user" for world events or "assistant" for agent response.
        content: Formatted content ready for LLM consumption.
        timestamp: When the entry was created.
        tokens: Pre-counted tokens for budget management.
        saved: True after @write flushes to CVM.
        doc_id: Set after CVM insert; used for deduplication.
        document_type: DOC_MUD_WORLD or DOC_MUD_AGENT.
        conversation_id: Groups related turns together.
        sequence_no: Order within the conversation.
        metadata: Rich metadata (room info, actions, event details).
        think: Assistant's <think> content if present.
        speaker_id: "world" for user turns or persona_id for assistant.
    """

    # Core fields
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=_utc_now)

    # Token management
    tokens: int

    # Persistence tracking
    saved: bool = False
    doc_id: Optional[str] = None

    # Document metadata (matches CVM schema)
    document_type: str
    conversation_id: str
    sequence_no: int

    # Rich metadata
    metadata: dict = Field(default_factory=dict)

    # Optional fields
    think: Optional[str] = None
    speaker_id: Optional[str] = None


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
        # Format events as pure prose (CVM-ready, no XML wrapper)
        # Use \n\n delimiter for paragraph separation
        if events:
            content = "\n\n".join(format_event(event) for event in events)
        else:
            content = "[No events]"

        # Count tokens
        tokens = count_tokens(content)

        # Build metadata
        actors = list({e.actor for e in events if e.actor})
        event_ids = [e.event_id for e in events if e.event_id]

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
            "event_count": len(events),
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
