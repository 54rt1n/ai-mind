# andimud_mediator/mixins/compiler.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Event compilation mixin for the mediator service.

Compiles MUD events into conversation entries with embeddings.
Events are compiled immediately into MUDConversationEntry objects.
"""

import asyncio
import logging
import secrets
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from aim_mud_types import (
    EventType,
    MUDConversationEntry,
    MUDEvent,
    RedisMUDClient,
)
from aim_mud_types.formatters import format_event, format_self_action_guidance
from aim_mud_types.helper import _utc_now
from aim.constants import DOC_CODE_ACTION, DOC_CODE_FILE, DOC_MUD_AGENT, DOC_MUD_WORLD
from aim.utils.tokens import count_tokens

if TYPE_CHECKING:
    import numpy as np
    from aim.conversation.embedding import HuggingFaceEmbedding

logger = logging.getLogger(__name__)


@dataclass
class PendingEventBatch:
    """Batch of events pending compilation for a room.

    Events in the same room are batched together. Third-person content is
    compiled once and published to all observers (embedding computed once).
    First-person content is compiled separately for the acting agent.

    Attributes:
        room_id: The room this batch is for (batch key).
        events: List of MUDEvent objects in this room.
        observer_agents: Set of agent IDs to receive third-person entries.
        self_action_events: List of (agent_id, event) pairs for first-person.
    """

    room_id: str
    events: list[MUDEvent] = field(default_factory=list)
    observer_agents: set[str] = field(default_factory=set)
    self_action_events: list[tuple[str, MUDEvent]] = field(default_factory=list)


class CompilerMixin:
    """Event compilation mixin for mediator.

    Compiles MUD events into conversation entries with embeddings.
    Events are compiled immediately upon arrival.

    Consecutive events from the same actor are grouped into one entry.
    """

    def _init_compiler(self) -> None:
        """Initialize compiler state. Call from service __init__."""
        self._pending_batches: dict[str, PendingEventBatch] = {}
        self._embedding_model: Optional["HuggingFaceEmbedding"] = None

    def _get_embedding_model(self) -> "HuggingFaceEmbedding":
        """Lazy-load embedding model.

        Returns:
            Initialized HuggingFaceEmbedding instance.
        """
        if self._embedding_model is None:
            from aim.conversation.embedding import HuggingFaceEmbedding

            model_name = getattr(
                self.config,
                "embedding_model",
                "mixedbread-ai/mxbai-embed-large-v1",
            )
            device = getattr(self.config, "embedding_device", "cpu")
            logger.info(
                "Initializing embedding model for mediator: %s (device=%s)",
                model_name,
                device,
            )
            self._embedding_model = HuggingFaceEmbedding(
                model_name=model_name,
                device=device,
            )
            logger.info("Embedding model ready")
        return self._embedding_model

    async def queue_event_for_compilation(
        self,
        event: MUDEvent,
        observer_agent_ids: list[str],
        self_action_agent_id: Optional[str],
    ) -> None:
        """Queue an event for immediate compilation.

        Third-person content is compiled once with one embedding computation,
        then published to all observers. First-person content is compiled
        separately for the self-action agent.

        Args:
            event: The MUDEvent to queue.
            observer_agent_ids: Agent IDs to receive third-person entries.
            self_action_agent_id: Agent ID to receive first-person entry, if any.
        """
        room_id = event.room_id or "unknown"
        batch = PendingEventBatch(room_id=room_id)

        batch.events.append(event)
        batch.observer_agents.update(observer_agent_ids)

        if self_action_agent_id:
            batch.self_action_events.append((self_action_agent_id, event))

        # Store batch temporarily for compilation
        self._pending_batches[room_id] = batch

        # Compile immediately
        await self._compile_batch(room_id)

    async def _compile_batch(self, room_id: str) -> None:
        """Compile pending batch into conversation entries.

        Pops the batch from pending, compiles third-person content once
        (with one embedding computation) and publishes to all observers,
        then compiles first-person content separately for self-action agents.

        Args:
            room_id: Room whose batch to compile.
        """
        batch = self._pending_batches.pop(room_id, None)
        if batch is None or (not batch.events and not batch.self_action_events):
            return

        logger.info(
            "Compiling batch of %d events for room %s (%d observers, %d self-actions)",
            len(batch.events),
            room_id,
            len(batch.observer_agents),
            len(batch.self_action_events),
        )

        client = RedisMUDClient(self.redis)
        total_entries_written = 0

        # Phase 1: Compile third-person entries for observers (embedding computed once)
        if batch.events and batch.observer_agents:
            # Build third-person entries (content + embedding computed ONCE)
            third_person_entries = await self._compile_events_to_entries_third_person(
                batch.events
            )

            # Publish same entries to all observers (each gets their own sequence_no)
            for agent_id in batch.observer_agents:
                for entry_template in third_person_entries:
                    # Get conversation tracking info for this specific agent
                    conversation_id = await self._get_agent_conversation_id(agent_id)
                    sequence_no = await self._next_conversation_sequence_no(agent_id)

                    # Create agent-specific entry with shared content/embedding
                    entry = MUDConversationEntry(
                        role=entry_template.role,
                        content=entry_template.content,
                        tokens=entry_template.tokens,
                        document_type=entry_template.document_type,
                        conversation_id=conversation_id,
                        sequence_no=sequence_no,
                        metadata=entry_template.metadata,
                        speaker_id="world",
                        timestamp=entry_template.timestamp,
                        last_event_id=entry_template.last_event_id,
                        embedding=entry_template.embedding,
                        saved=False,
                        skip_save=False,
                    )

                    await client.append_conversation_entry(
                        agent_id,
                        entry.model_dump_json(),
                    )
                    total_entries_written += 1

            logger.info(
                "Wrote %d third-person entries to %d observers for room %s",
                len(third_person_entries),
                len(batch.observer_agents),
                room_id,
            )

        # Phase 2: Compile first-person entries for self-action agents
        if batch.self_action_events:
            # Group self-action events by agent
            self_actions_by_agent: dict[str, list[MUDEvent]] = {}
            for agent_id, event in batch.self_action_events:
                if agent_id not in self_actions_by_agent:
                    self_actions_by_agent[agent_id] = []
                self_actions_by_agent[agent_id].append(event)

            for agent_id, events in self_actions_by_agent.items():
                entries = await self._compile_events_to_entries_first_person(
                    events, agent_id
                )
                for entry in entries:
                    await client.append_conversation_entry(
                        agent_id,
                        entry.model_dump_json(),
                    )
                    total_entries_written += 1

                logger.info(
                    "Wrote %d first-person entries for %s",
                    len(entries),
                    agent_id,
                )

        logger.info(
            "Compiled batch for room %s: %d total entries written",
            room_id,
            total_entries_written,
        )

    async def _compile_events_to_entries_third_person(
        self,
        events: list[MUDEvent],
    ) -> list[MUDConversationEntry]:
        """Compile events into third-person conversation entries (for observers).

        Groups consecutive events by actor, formats as third-person content,
        computes embeddings ONCE per group, and returns template entries.
        These templates are then personalized per agent (conversation_id, sequence_no)
        when published.

        Args:
            events: List of MUDEvent objects.

        Returns:
            List of MUDConversationEntry templates with embeddings.
        """
        # Group consecutive events by actor
        # Code events always get their own entry
        groups: list[list[MUDEvent]] = []
        current_group: list[MUDEvent] = []
        current_actor_key: Optional[str] = None

        for event in events:
            actor_key = event.actor_id or event.actor

            # Code events always get their own entry
            if event.event_type in (EventType.CODE_ACTION, EventType.CODE_FILE):
                if current_group:
                    groups.append(current_group)
                    current_group = []
                groups.append([event])
                current_actor_key = None
                continue

            # Check if we need to flush current group
            if current_group and actor_key != current_actor_key:
                groups.append(current_group)
                current_group = []

            current_group.append(event)
            current_actor_key = actor_key

        # Flush final group
        if current_group:
            groups.append(current_group)

        # Build entries from groups
        entries: list[MUDConversationEntry] = []
        for group_events in groups:
            entry = await self._build_entry_from_group_third_person(group_events)
            entries.append(entry)

        return entries

    async def _compile_events_to_entries_first_person(
        self,
        events: list[MUDEvent],
        agent_id: str,
    ) -> list[MUDConversationEntry]:
        """Compile events into first-person conversation entries (for actor).

        Groups consecutive events by actor, formats as first-person content,
        computes embeddings, and returns entries for the self-action agent.

        Args:
            events: List of MUDEvent objects.
            agent_id: The acting agent's ID.

        Returns:
            List of MUDConversationEntry objects with embeddings.
        """
        # Group consecutive events by actor
        # Code events always get their own entry
        groups: list[list[MUDEvent]] = []
        current_group: list[MUDEvent] = []
        current_actor_key: Optional[str] = None

        for event in events:
            actor_key = event.actor_id or event.actor

            # Code events always get their own entry
            if event.event_type in (EventType.CODE_ACTION, EventType.CODE_FILE):
                if current_group:
                    groups.append(current_group)
                    current_group = []
                groups.append([event])
                current_actor_key = None
                continue

            # Check if we need to flush current group
            if current_group and actor_key != current_actor_key:
                groups.append(current_group)
                current_group = []

            current_group.append(event)
            current_actor_key = actor_key

        # Flush final group
        if current_group:
            groups.append(current_group)

        # Build entries from groups
        entries: list[MUDConversationEntry] = []
        for group_events in groups:
            entry = await self._build_entry_from_group_first_person(
                group_events, agent_id
            )
            entries.append(entry)

        return entries

    async def _build_entry_from_group_third_person(
        self,
        events: list[MUDEvent],
    ) -> MUDConversationEntry:
        """Build a third-person conversation entry template from a group of events.

        This creates a template with content and embedding computed once.
        Per-agent fields (conversation_id, sequence_no) are filled in later
        when publishing to each observer.

        Args:
            events: List of events to compile into one entry.

        Returns:
            MUDConversationEntry template with computed embedding.
        """
        # Format content as third-person
        formatted_parts: list[str] = []
        for event in events:
            formatted_parts.append(format_event(event))

        content = "\n\n".join(formatted_parts)

        # Determine document type
        first_event = events[0]
        if first_event.event_type == EventType.CODE_ACTION:
            doc_type = DOC_CODE_ACTION
        elif first_event.event_type == EventType.CODE_FILE:
            doc_type = DOC_CODE_FILE
        else:
            doc_type = DOC_MUD_WORLD

        # Count tokens
        tokens = count_tokens(content)

        # Build metadata
        metadata = {
            "room_id": first_event.room_id,
            "room_name": first_event.room_name,
            "event_count": len(events),
            "actors": list({e.actor for e in events if e.actor}),
            "actor_ids": list({e.actor_id for e in events if e.actor_id}),
            "event_ids": [e.event_id for e in events if e.event_id],
            "event_types": [e.event_type.value for e in events],
        }

        # DOC_MUD_WORLD entries get written to CVM but without embeddings
        # (only agent's own actions are searchable/comparable via embedding index)
        is_world_event = doc_type == DOC_MUD_WORLD

        # Skip embedding computation for world events (saves compute, not searchable)
        if is_world_event:
            embedding_b64 = None
        else:
            # Compute embedding in thread pool (computed ONCE, shared to all observers)
            embedding_b64 = await self._compute_embedding(content)

        # Create template entry - conversation_id and sequence_no will be set per-agent
        entry = MUDConversationEntry(
            role="user",
            content=content,
            tokens=tokens,
            document_type=doc_type,
            conversation_id="",  # Placeholder - set per agent
            sequence_no=0,  # Placeholder - set per agent
            metadata=metadata,
            speaker_id="world",
            timestamp=first_event.timestamp,
            last_event_id=events[-1].event_id or "",
            embedding=embedding_b64,
            saved=False,
            skip_save=False,  # Still written to CVM (just without embedding index)
        )

        return entry

    async def _build_entry_from_group_first_person(
        self,
        events: list[MUDEvent],
        agent_id: str,
    ) -> MUDConversationEntry:
        """Build a first-person conversation entry from a group of events.

        This creates a complete entry for the acting agent with first-person
        formatting and its own embedding.

        Args:
            events: List of events to compile into one entry.
            agent_id: The acting agent's ID.

        Returns:
            MUDConversationEntry with computed embedding.
        """
        # Format content as first-person with visual banner
        content = format_self_action_guidance(events, world_state=None)

        # Determine document type
        first_event = events[0]
        if first_event.event_type == EventType.CODE_ACTION:
            doc_type = DOC_CODE_ACTION
        elif first_event.event_type == EventType.CODE_FILE:
            doc_type = DOC_CODE_FILE
        else:
            doc_type = DOC_MUD_AGENT

        # Count tokens
        tokens = count_tokens(content)

        # Build metadata
        metadata = {
            "room_id": first_event.room_id,
            "room_name": first_event.room_name,
            "event_count": len(events),
            "actors": list({e.actor for e in events if e.actor}),
            "actor_ids": list({e.actor_id for e in events if e.actor_id}),
            "event_ids": [e.event_id for e in events if e.event_id],
            "event_types": [e.event_type.value for e in events],
        }

        # Compute embedding in thread pool
        embedding_b64 = await self._compute_embedding(content)

        # Get conversation tracking info for this agent
        conversation_id = await self._get_agent_conversation_id(agent_id)
        sequence_no = await self._next_conversation_sequence_no(agent_id)

        entry = MUDConversationEntry(
            role="user",
            content=content,
            tokens=tokens,
            document_type=doc_type,
            conversation_id=conversation_id,
            sequence_no=sequence_no,
            metadata=metadata,
            speaker_id=agent_id,
            timestamp=first_event.timestamp,
            last_event_id=events[-1].event_id or "",
            embedding=embedding_b64,
            saved=False,
            skip_save=False,
        )

        return entry

    async def _compute_embedding(self, content: str) -> str:
        """Compute embedding for content in thread pool.

        Args:
            content: Text to embed.

        Returns:
            Base64-encoded embedding vector.
        """
        loop = asyncio.get_event_loop()
        embedding_model = self._get_embedding_model()

        # Run embedding computation in thread pool to avoid blocking
        embedding_vector: "np.ndarray" = await loop.run_in_executor(
            None,
            embedding_model,
            content,
        )

        return MUDConversationEntry.encode_embedding(embedding_vector)

    async def _get_agent_conversation_id(self, agent_id: str) -> str:
        """Get or create conversation ID for agent from Redis.

        Args:
            agent_id: Agent identifier.

        Returns:
            Existing or newly generated conversation ID.
        """
        client = RedisMUDClient(self.redis)
        profile = await client.get_agent_profile(agent_id)

        if profile is not None and profile.conversation_id:
            return profile.conversation_id

        # Generate new conversation ID
        ts = int(_utc_now().timestamp() * 1000)
        suffix = secrets.token_hex(4)
        new_id = f"andimud_{ts}_{suffix}"

        # Persist the new conversation ID to the profile
        if profile is not None:
            await client.update_agent_profile_fields(
                agent_id,
                conversation_id=new_id,
            )

        return new_id

    async def _next_conversation_sequence_no(self, agent_id: str) -> int:
        """Get next conversation sequence number for agent.

        Uses conversation list length as the sequence number, ensuring
        monotonically increasing values.

        Args:
            agent_id: Agent identifier.

        Returns:
            Next sequence number (current list length).
        """
        client = RedisMUDClient(self.redis)
        length = await client.get_conversation_length(agent_id)
        return length
