# aim/dreamer/core/memory_dsl.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unified Memory DSL executor for pipeline context building.

This module implements the Memory Operations DSL that provides a consistent
interface for all memory operations in both seed-level and step-level contexts.

Action Categories:
- Retrieval: load_conversation, get_memory, search_memories
- Transform: sort, filter, truncate, drop
- Meta: flush, clear
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ...conversation.model import ConversationModel
    from .models import MemoryAction, PipelineState
    from .state import ScenarioState

logger = logging.getLogger(__name__)

# Type alias for state objects
StateType = Union["PipelineState", "ScenarioState"]


def execute_memory_actions(
    actions: list["MemoryAction"],
    state: StateType,
    cvm: "ConversationModel",
    query_text: str | None = None,
) -> list[str]:
    """Execute a sequence of memory actions, returning accumulated doc_ids.

    This is the unified entry point for all memory operations in the DSL.
    Actions execute sequentially, each operating on and modifying the
    accumulated doc_ids list.

    Args:
        actions: List of MemoryAction objects to execute
        state: Pipeline or Dialogue state for context
        cvm: ConversationModel for querying documents
        query_text: Optional query text for search_memories (if not in state)

    Returns:
        Deduplicated list of doc_ids in accumulated order
    """
    accumulated_doc_ids: list[str] = []

    for action in actions:
        if action.action == "load_conversation":
            doc_ids = _load_conversation(action, state, cvm)
            accumulated_doc_ids.extend(doc_ids)

        elif action.action == "get_memory":
            # Check min_memories threshold
            if action.min_memories and len(accumulated_doc_ids) >= action.min_memories:
                logger.debug(
                    f"get_memory: Skipping - already have {len(accumulated_doc_ids)} docs "
                    f"(min_memories={action.min_memories})"
                )
            else:
                doc_ids = _get_memory(action, state, cvm)
                accumulated_doc_ids.extend(doc_ids)

        elif action.action == "search_memories":
            # Check min_memories threshold
            if action.min_memories and len(accumulated_doc_ids) >= action.min_memories:
                logger.debug(
                    f"search_memories: Skipping - already have {len(accumulated_doc_ids)} docs "
                    f"(min_memories={action.min_memories})"
                )
            else:
                doc_ids = _search_memories(action, state, cvm, accumulated_doc_ids, query_text)
                accumulated_doc_ids.extend(doc_ids)

        elif action.action == "sort":
            accumulated_doc_ids = _sort_docs(action, accumulated_doc_ids, cvm)

        elif action.action == "filter":
            accumulated_doc_ids = _filter_docs(action, accumulated_doc_ids, cvm)

        elif action.action == "truncate":
            limit = action.limit or action.top_n or len(accumulated_doc_ids)
            if len(accumulated_doc_ids) > limit:
                logger.debug(f"truncate: Limiting from {len(accumulated_doc_ids)} to {limit} docs")
                accumulated_doc_ids = accumulated_doc_ids[:limit]

        elif action.action == "drop":
            accumulated_doc_ids = _drop_docs(action, accumulated_doc_ids, cvm)

        elif action.action in ("flush", "clear"):
            logger.debug(f"{action.action}: Clearing {len(accumulated_doc_ids)} accumulated docs")
            accumulated_doc_ids = []

    # Deduplicate while preserving order
    return _deduplicate(accumulated_doc_ids)


def _load_conversation(
    action: "MemoryAction",
    state: StateType,
    cvm: "ConversationModel",
) -> list[str]:
    """Load documents from a conversation.

    Bulk loads all documents matching criteria from a conversation.
    This is a chronological load, not a semantic search.

    Args:
        action: MemoryAction with load_conversation params
        state: Pipeline/Dialogue state for conversation_id
        cvm: ConversationModel

    Returns:
        List of doc_ids from the conversation
    """
    # Resolve conversation ID
    conv_id = state.conversation_id
    if action.target and action.target != "current":
        conv_id = action.target

    if not conv_id:
        logger.warning("load_conversation: No conversation_id available")
        return []

    # Get conversation history
    history_df = cvm.get_conversation_history(
        conv_id,
        query_document_type=action.document_types,
        filter_document_type=action.exclude_types,
    )

    if history_df.empty:
        logger.debug(f"load_conversation: No documents found for {conv_id}")
        return []

    doc_ids = history_df['doc_id'].tolist()
    logger.debug(f"load_conversation: Loaded {len(doc_ids)} docs from {conv_id}")
    return doc_ids


def _get_memory(
    action: "MemoryAction",
    state: StateType,
    cvm: "ConversationModel",
) -> list[str]:
    """Direct retrieval by criteria without semantic search.

    Queries the index for documents matching type/conversation criteria.
    Unlike search_memories, this does not use vector similarity.

    Args:
        action: MemoryAction with get_memory params
        state: Pipeline/Dialogue state
        cvm: ConversationModel

    Returns:
        List of doc_ids from the query
    """
    # Resolve conversation filter
    query_conv_id = None
    if action.conversation_id == "current":
        query_conv_id = state.conversation_id
    elif action.conversation_id and action.conversation_id != "all":
        query_conv_id = action.conversation_id
    # If conversation_id is None or "all", search across all conversations

    top_n = action.top_n or 100

    # Use index.search for direct document retrieval
    results = cvm.index.search(
        query_document_type=action.document_types,
        filter_document_type=action.exclude_types,
        query_conversation_id=query_conv_id,
        query_limit=top_n,
    )

    if results.empty:
        logger.debug("get_memory: No documents found")
        return []

    doc_ids = results['doc_id'].tolist()
    logger.debug(f"get_memory: Found {len(doc_ids)} docs")
    return doc_ids


def _search_memories(
    action: "MemoryAction",
    state: StateType,
    cvm: "ConversationModel",
    accumulated_doc_ids: list[str],
    override_query_text: str | None = None,
) -> list[str]:
    """Semantic vector search for memories.

    Uses embedding similarity to find relevant documents.

    Args:
        action: MemoryAction with search_memories params
        state: Pipeline/Dialogue state
        cvm: ConversationModel
        accumulated_doc_ids: Current accumulated docs (for use_context)
        override_query_text: Optional query text override

    Returns:
        List of doc_ids from semantic search
    """
    from aim.constants import CHUNK_LEVEL_768

    # Determine query text
    query_text = None

    if action.use_context and accumulated_doc_ids:
        # Build query from accumulated document content
        contents = []
        for doc_id in accumulated_doc_ids[-5:]:  # Use last 5 for context
            doc = cvm.get_by_doc_id(doc_id)
            if doc and 'content' in doc:
                contents.append(doc['content'][:500])  # Limit per doc
        if contents:
            query_text = " ".join(contents)
            logger.debug(f"search_memories: Using accumulated context as query ({len(query_text)} chars)")

    if not query_text:
        query_text = action.query_text or override_query_text or getattr(state, 'query_text', None)

    if not query_text:
        logger.warning("search_memories: No query text available, returning empty")
        return []

    # Resolve conversation filter
    query_conv_id = None
    if action.conversation_id == "current":
        query_conv_id = state.conversation_id
    elif action.conversation_id and action.conversation_id != "all":
        query_conv_id = action.conversation_id

    top_n = action.top_n or 10
    temporal_decay = action.temporal_decay if action.temporal_decay is not None else 0.99
    chunk_level = action.chunk_level or CHUNK_LEVEL_768

    # Semantic query
    memories_df = cvm.query(
        query_texts=[query_text],
        top_n=top_n,
        query_document_type=action.document_types,
        filter_document_type=action.exclude_types,
        query_conversation_id=query_conv_id,
        sort_by="relevance",
        temporal_decay=temporal_decay,
        chunk_level=chunk_level,
    )

    if memories_df.empty:
        logger.debug("search_memories: No memories found")
        return []

    doc_ids = memories_df['doc_id'].tolist()
    logger.debug(f"search_memories: Found {len(doc_ids)} memories")
    return doc_ids


def _sort_docs(
    action: "MemoryAction",
    doc_ids: list[str],
    cvm: "ConversationModel",
) -> list[str]:
    """Sort accumulated doc_ids by timestamp or other criteria.

    Args:
        action: MemoryAction with sort params
        doc_ids: Current accumulated doc_ids
        cvm: ConversationModel for looking up timestamps

    Returns:
        Sorted list of doc_ids
    """
    if not doc_ids:
        return doc_ids

    sort_by = action.by or "timestamp"
    ascending = (action.direction == "ascending")

    if sort_by == "timestamp":
        # Look up timestamps for each doc_id
        docs_with_ts = []
        for doc_id in doc_ids:
            doc = cvm.get_by_doc_id(doc_id)
            if doc:
                ts = doc.get('timestamp', 0)
                docs_with_ts.append((doc_id, ts))
            else:
                # Keep docs without timestamp at the end
                docs_with_ts.append((doc_id, float('inf') if ascending else 0))

        docs_with_ts.sort(key=lambda x: x[1], reverse=not ascending)
        sorted_ids = [doc_id for doc_id, _ in docs_with_ts]
        logger.debug(f"sort: Sorted {len(sorted_ids)} docs by timestamp ({'asc' if ascending else 'desc'})")
        return sorted_ids

    elif sort_by == "relevance":
        # For relevance sorting, we'd need scores from a query
        # This is a no-op if docs weren't loaded via query with scoring
        logger.debug("sort: Relevance sort requested but no scores available, keeping original order")
        return doc_ids

    else:
        logger.warning(f"sort: Unknown sort_by value '{sort_by}', keeping original order")
        return doc_ids


def _filter_docs(
    action: "MemoryAction",
    doc_ids: list[str],
    cvm: "ConversationModel",
) -> list[str]:
    """Filter accumulated doc_ids by document type.

    Args:
        action: MemoryAction with filter params
        doc_ids: Current accumulated doc_ids
        cvm: ConversationModel

    Returns:
        Filtered list of doc_ids
    """
    if not doc_ids:
        return doc_ids

    # Filter by document_types if specified
    if action.document_types:
        filtered = []
        for doc_id in doc_ids:
            doc = cvm.get_by_doc_id(doc_id)
            if doc and doc.get('document_type') in action.document_types:
                filtered.append(doc_id)
        logger.debug(f"filter: Kept {len(filtered)}/{len(doc_ids)} docs matching types {action.document_types}")
        return filtered

    # Filter by exclude_types if specified
    if action.exclude_types:
        filtered = []
        for doc_id in doc_ids:
            doc = cvm.get_by_doc_id(doc_id)
            if doc and doc.get('document_type') not in action.exclude_types:
                filtered.append(doc_id)
        logger.debug(f"filter: Kept {len(filtered)}/{len(doc_ids)} docs after excluding {action.exclude_types}")
        return filtered

    # Match pattern (future)
    if action.match:
        logger.debug(f"filter: Pattern matching not yet implemented, passing through {len(doc_ids)} docs")
        return doc_ids

    return doc_ids


def _drop_docs(
    action: "MemoryAction",
    doc_ids: list[str],
    cvm: "ConversationModel",
) -> list[str]:
    """Remove specific docs or types from accumulated context.

    Args:
        action: MemoryAction with drop params
        doc_ids: Current accumulated doc_ids
        cvm: ConversationModel

    Returns:
        List with specified docs removed
    """
    if not doc_ids:
        return doc_ids

    # Drop specific doc_ids
    if action.doc_ids:
        drop_set = set(action.doc_ids)
        filtered = [d for d in doc_ids if d not in drop_set]
        logger.debug(f"drop: Removed {len(doc_ids) - len(filtered)} docs by ID")
        return filtered

    # Drop by document_types
    if action.document_types:
        filtered = []
        for doc_id in doc_ids:
            doc = cvm.get_by_doc_id(doc_id)
            if doc and doc.get('document_type') not in action.document_types:
                filtered.append(doc_id)
        logger.debug(f"drop: Removed {len(doc_ids) - len(filtered)} docs of types {action.document_types}")
        return filtered

    return doc_ids


def _deduplicate(doc_ids: list[str]) -> list[str]:
    """Deduplicate doc_ids while preserving order of first occurrence.

    Args:
        doc_ids: List of doc_ids (may contain duplicates)

    Returns:
        Deduplicated list in order of first occurrence
    """
    seen = set()
    unique = []
    for doc_id in doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            unique.append(doc_id)
    return unique
