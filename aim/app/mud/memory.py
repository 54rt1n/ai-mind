# aim/app/mud/memory.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""MUD memory bucket, persistence, and retrieval helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import time
import uuid
import warnings
from typing import Optional, Iterable

from aim.utils.tokens import count_tokens
from aim.constants import (
    DOC_CONVERSATION,
    DOC_INSPIRATION,
    DOC_UNDERSTANDING,
    DOC_PONDERING,
    DOC_BRAINSTORM,
    DOC_MUD_WORLD,
    DOC_MUD_AGENT,
    CHUNK_LEVEL_256,
    CHUNK_LEVEL_768,
)
from aim.conversation.message import ConversationMessage
from aim.conversation.rerank import MemoryReranker, TaggedResult
from aim.utils.xml import XmlFormatter

from .session import MUDEvent, MUDAction, WorldState
from .adapter import format_event


SCENARIO_PREFIX = "andimud"
INSIGHT_DOC_TYPES = [DOC_INSPIRATION, DOC_UNDERSTANDING, DOC_PONDERING, DOC_BRAINSTORM]
LONG_CONTEXT_DOC_TYPES = [DOC_CONVERSATION, DOC_MUD_AGENT, DOC_MUD_WORLD]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_conversation_id(prefix: str = SCENARIO_PREFIX) -> str:
    """Generate a Dreamer-style conversation_id."""
    timestamp_ms = int(time.time() * 1000)
    random_suffix = uuid.uuid4().hex[:9]
    return f"{prefix}_{timestamp_ms}_{random_suffix}"


def _safe_iso(dt: Optional[datetime]) -> str:
    if not dt:
        return ""
    return dt.isoformat()


@dataclass
class MUDMemoryBucket:
    """Accumulates world inputs and agent outputs for persistence."""

    bucket_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = field(default_factory=_utc_now)
    last_event_at: Optional[datetime] = None
    events: list[MUDEvent] = field(default_factory=list)
    responses: list[str] = field(default_factory=list)
    actions: list[MUDAction] = field(default_factory=list)
    latest_world_state: Optional[WorldState] = None
    _event_ids: set[str] = field(default_factory=set, repr=False)

    def append_events(self, events: Iterable[MUDEvent]) -> None:
        """Append events in stream order, avoiding duplicates."""
        for event in events:
            if event.event_id and event.event_id in self._event_ids:
                continue
            self.events.append(event)
            if event.event_id:
                self._event_ids.add(event.event_id)
            self.last_event_at = event.timestamp

    def append_response(self, response: str, actions: Iterable[MUDAction]) -> None:
        """Append a raw model response and associated actions."""
        if response:
            self.responses.append(response)
        if actions:
            self.actions.extend(actions)

    def update_world_state(self, world_state: Optional[WorldState]) -> None:
        """Store the latest world state snapshot for the bucket."""
        if world_state:
            self.latest_world_state = world_state

    def has_content(self) -> bool:
        return bool(self.events or self.responses)

    def idle_seconds(self, now: Optional[datetime] = None) -> Optional[float]:
        if not self.last_event_at:
            if not self.responses:
                return None
            current = now or _utc_now()
            return (current - self.created_at).total_seconds()
        current = now or _utc_now()
        return (current - self.last_event_at).total_seconds()

    def should_flush(
        self,
        max_tokens: int,
        idle_flush_seconds: int,
        now: Optional[datetime] = None,
    ) -> bool:
        """Determine if bucket should flush by token or idle threshold."""
        if not self.has_content():
            return False

        token_estimate = self.estimate_tokens()
        if token_estimate >= max_tokens:
            return True

        idle_seconds = self.idle_seconds(now)
        if idle_seconds is not None and idle_seconds >= idle_flush_seconds:
            return True

        return False

    def estimate_tokens(self) -> int:
        """Estimate total tokens for world + agent content."""
        return count_tokens(self.render_world_content()) + count_tokens(self.render_agent_content())

    def render_world_content(self) -> str:
        """Render raw world inputs for persistence."""
        lines: list[str] = []

        if not self.events:
            return "No events recorded."

        # Bucket time range
        start_ts = self.events[0].timestamp
        end_ts = self.events[-1].timestamp
        lines.append(f"[Bucket {self.bucket_id}] {start_ts.isoformat()} -> {end_ts.isoformat()}")

        # Latest world snapshot
        last_world = self._latest_world_state()
        if last_world:
            lines.append(last_world.to_xml(include_self=False))

        lines.append(f"<events count=\"{len(self.events)}\">")
        event_lines = [self._format_event_line(event) for event in self.events]
        lines.append("\n\n".join(event_lines))
        lines.append("</events>")

        return "\n".join(lines)

    def render_agent_content(self) -> str:
        """Render raw agent outputs for persistence."""
        if not self.responses:
            return ""
        return "\n\n---\n\n".join(self.responses)

    def _latest_world_state(self) -> Optional[WorldState]:
        return self.latest_world_state

    def _format_event_line(self, event: MUDEvent) -> str:
        return format_event(event).strip()


class MUDMemoryPersister:
    """Persists MUD buckets to CVM."""

    def __init__(self, scenario_prefix: str = SCENARIO_PREFIX):
        self.scenario_prefix = scenario_prefix

    def persist_bucket(
        self,
        bucket: MUDMemoryBucket,
        cvm,
        persona_id: str,
        user_id: str = "mud",
        inference_model: Optional[str] = None,
    ) -> Optional[str]:
        if not bucket.has_content():
            return None

        conversation_id = generate_conversation_id(self.scenario_prefix)

        world_content = bucket.render_world_content()
        agent_content = bucket.render_agent_content()

        start_ts = bucket.events[0].timestamp if bucket.events else None
        end_ts = bucket.events[-1].timestamp if bucket.events else None

        metadata_base = {
            "bucket_id": bucket.bucket_id,
            "start_ts": _safe_iso(start_ts),
            "end_ts": _safe_iso(end_ts),
            "event_count": len(bucket.events),
            "action_count": len(bucket.actions),
            "room_ids": list({e.room_id for e in bucket.events if e.room_id}),
            "room_names": list({e.room_name for e in bucket.events if e.room_name}),
            "actors": list({e.actor for e in bucket.events if e.actor}),
            "event_ids": [e.event_id for e in bucket.events if e.event_id],
        }

        world_message = ConversationMessage.create(
            conversation_id=conversation_id,
            sequence_no=0,
            role="user",
            content=world_content,
            speaker_id="world",
            user_id=user_id,
            persona_id=persona_id,
            document_type=DOC_MUD_WORLD,
            metadata=json.dumps(metadata_base),
            inference_model=inference_model,
        )

        agent_metadata = dict(metadata_base)
        agent_metadata["actions"] = [a.to_command() for a in bucket.actions]

        agent_message = ConversationMessage.create(
            conversation_id=conversation_id,
            sequence_no=1,
            role="assistant",
            content=agent_content,
            speaker_id=persona_id,
            user_id=user_id,
            persona_id=persona_id,
            document_type=DOC_MUD_AGENT,
            metadata=json.dumps(agent_metadata),
            inference_model=inference_model,
        )

        cvm.insert(world_message)
        cvm.insert(agent_message)

        return conversation_id


class MUDMemoryRetriever:
    """Retrieves MUD memories using CVM query + reranker.

    DEPRECATED: Use MUDResponseStrategy instead, which integrates with
    XMLMemoryTurnStrategy for memory retrieval.

    This class is kept for backwards compatibility but should not be used
    in new code.
    """

    def __init__(self, cvm, top_n: int = 10):
        warnings.warn(
            "MUDMemoryRetriever is deprecated. Use MUDResponseStrategy instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.cvm = cvm
        self.top_n = top_n

    def build_memory_block(
        self,
        session,
        token_budget: int,
    ) -> tuple[str, int]:
        """Build MUD memory XML block within token budget."""
        queries = self._build_queries(session)
        if not queries or token_budget <= 0:
            return "", 0

        conv_results, insight_results, broad_results = self._query_by_buckets(
            queries=queries,
            source_tag="mud",
            seen_docs=set(),
            top_n=self.top_n,
        )

        all_long_context = conv_results + insight_results
        reranker = MemoryReranker(token_counter=count_tokens, lambda_param=0.7, conversation_budget_ratio=0.6)
        reranked = reranker.rerank(
            conversation_results=all_long_context,
            other_results=broad_results,
            token_budget=token_budget,
            seen_parent_ids=set(),
        )

        # Deduplicate by parent_doc_id/doc_id
        deduped: list[TaggedResult] = []
        seen_ids: set[str] = set()
        for source_tag, row in reranked:
            doc_id = row.get("parent_doc_id", row.get("doc_id", ""))
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            deduped.append((source_tag, row))

        block = self._format_memory_block(deduped)
        return block, count_tokens(block)

    def _build_queries(self, session) -> list[str]:
        queries: list[str] = []

        # Room
        if session.current_room:
            if session.current_room.name:
                queries.append(session.current_room.name)
            if session.current_room.description:
                queries.append(session.current_room.description)

        # Entities
        if session.entities_present:
            names = [e.name for e in session.entities_present if not e.is_self and e.name]
            if names:
                queries.append(" ".join(names))

        # Speech content
        speech_events = [e for e in session.pending_events if e.event_type.value == "speech" and e.content]
        if speech_events:
            queries.append(" ".join(e.content for e in speech_events))

        return [q for q in queries if q and q.strip()]

    def _query_by_buckets(
        self,
        queries: list[str],
        source_tag: str,
        seen_docs: set,
        top_n: int,
        length_boost: float = 0.0,
    ) -> tuple[list[TaggedResult], list[TaggedResult], list[TaggedResult]]:
        """Query memories in the same buckets as XMLMemoryTurnStrategy."""
        conversation_results: list[TaggedResult] = []
        insight_results: list[TaggedResult] = []
        broad_results: list[TaggedResult] = []

        conv_df = self.cvm.query(
            queries,
            filter_doc_ids=seen_docs,
            top_n=top_n,
            filter_metadocs=True,
            query_document_type=LONG_CONTEXT_DOC_TYPES,
            chunk_level=CHUNK_LEVEL_768,
            length_boost_factor=length_boost,
        )
        if not conv_df.empty:
            for _, row in conv_df.iterrows():
                conversation_results.append((f"{source_tag}_conv", row))

        insight_df = self.cvm.query(
            queries,
            filter_doc_ids=seen_docs,
            top_n=top_n,
            filter_metadocs=True,
            query_document_type=INSIGHT_DOC_TYPES,
            chunk_level=CHUNK_LEVEL_768,
            length_boost_factor=length_boost,
        )
        if not insight_df.empty:
            for _, row in insight_df.iterrows():
                insight_results.append((f"{source_tag}_insight", row))

        broad_df = self.cvm.query(
            queries,
            filter_doc_ids=seen_docs,
            top_n=top_n,
            filter_metadocs=True,
            chunk_level=CHUNK_LEVEL_256,
            length_boost_factor=length_boost,
        )
        if not broad_df.empty:
            for _, row in broad_df.iterrows():
                broad_results.append((f"{source_tag}_broad", row))

        return conversation_results, insight_results, broad_results

    def _format_memory_block(self, results: list[TaggedResult]) -> str:
        if not results:
            return ""

        xml = XmlFormatter()
        xml.add_element("MUDMemory", content="-- MUD Memory --", nowrap=True, priority=1)

        for source_tag, row in results:
            content = row.get("content", "")
            doc_type = row.get("document_type", "")
            date = row.get("date", "")
            xml.add_element(
                "MUDMemory",
                "Memory",
                content=content,
                type=doc_type,
                source=source_tag,
                date=date,
                noindent=True,
                priority=1,
            )

        return xml.render()
