# aim/dreamer/context.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Context preparation DSL executor for pipeline steps.

This module implements the step-level context DSL that allows declarative
specification of how each step's input context is assembled.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aim.conversation.model import ConversationModel
    from aim.dreamer.models import ContextAction, StepDefinition, PipelineState

logger = logging.getLogger(__name__)


def prepare_step_context(
    step_def: "StepDefinition",
    state: "PipelineState",
    cvm: "ConversationModel",
) -> tuple[list[str], bool]:
    """Execute context DSL actions and return ordered doc_ids.

    Context accumulates through the pipeline:
    - First step with context DSL: executes DSL to build initial context
    - Subsequent steps: receive accumulated context (initial + all prior outputs)

    Args:
        step_def: Step definition with optional context DSL
        state: Current pipeline state
        cvm: ConversationModel for querying documents

    Returns:
        Tuple of (doc_ids, is_initial_context):
        - doc_ids: Ordered list of doc_ids to use as step context
        - is_initial_context: True if this is new context from DSL (should be stored in state)
    """
    if not step_def.context:
        # No context DSL - use accumulated context from prior steps
        # This includes initial context + all prior step outputs
        return list(state.context_doc_ids), False

    accumulated_doc_ids: list[str] = []

    for action in step_def.context:
        if action.action == "load_conversation":
            doc_ids = _load_conversation(action, state, cvm)
            accumulated_doc_ids.extend(doc_ids)

        elif action.action == "query":
            doc_ids = _query_documents(action, state, cvm)
            accumulated_doc_ids.extend(doc_ids)

        elif action.action == "sort":
            accumulated_doc_ids = _sort_doc_ids(action, accumulated_doc_ids, cvm)

        elif action.action == "filter":
            accumulated_doc_ids = _filter_doc_ids(action, accumulated_doc_ids, cvm)

    # Deduplicate while preserving order
    seen = set()
    unique_doc_ids = []
    for doc_id in accumulated_doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            unique_doc_ids.append(doc_id)

    logger.debug(f"Context DSL produced {len(unique_doc_ids)} doc_ids for step {step_def.id}")
    return unique_doc_ids, True  # True = this is initial context from DSL


def _load_conversation(
    action: "ContextAction",
    state: "PipelineState",
    cvm: "ConversationModel",
) -> list[str]:
    """Load documents from a conversation.

    Args:
        action: ContextAction with load_conversation params
        state: Pipeline state for conversation_id
        cvm: ConversationModel

    Returns:
        List of doc_ids from the conversation
    """
    # Resolve conversation ID
    conv_id = state.conversation_id
    if action.target and action.target != "current":
        conv_id = action.target

    # Build query params
    query_doc_type = action.document_types
    filter_doc_type = action.exclude_types

    # Get conversation history
    history_df = cvm.get_conversation_history(
        conv_id,
        query_document_type=query_doc_type,
        filter_document_type=filter_doc_type,
    )

    if history_df.empty:
        logger.debug(f"load_conversation: No documents found for {conv_id}")
        return []

    doc_ids = history_df['doc_id'].tolist()
    logger.debug(f"load_conversation: Loaded {len(doc_ids)} docs from {conv_id}")
    return doc_ids


def _query_documents(
    action: "ContextAction",
    state: "PipelineState",
    cvm: "ConversationModel",
) -> list[str]:
    """Query documents from the index.

    Unlike load_conversation which loads from a specific conversation,
    query searches across all conversations (or a specific one if specified).

    Args:
        action: ContextAction with query params
        state: Pipeline state
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

    # Use index.search for direct document retrieval (no semantic search)
    results = cvm.index.search(
        query_document_type=action.document_types,
        filter_document_type=action.exclude_types,
        query_conversation_id=query_conv_id,
        query_limit=top_n,
    )

    if results.empty:
        logger.debug(f"query: No documents found")
        return []

    doc_ids = results['doc_id'].tolist()
    logger.debug(f"query: Found {len(doc_ids)} docs")
    return doc_ids


def _sort_doc_ids(
    action: "ContextAction",
    doc_ids: list[str],
    cvm: "ConversationModel",
) -> list[str]:
    """Sort accumulated doc_ids by timestamp or other criteria.

    Args:
        action: ContextAction with sort params
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
        logger.debug(f"sort: Relevance sort requested but no scores available, keeping original order")
        return doc_ids

    else:
        logger.warning(f"sort: Unknown sort_by value '{sort_by}', keeping original order")
        return doc_ids


def _filter_doc_ids(
    action: "ContextAction",
    doc_ids: list[str],
    cvm: "ConversationModel",
) -> list[str]:
    """Filter accumulated doc_ids (future implementation).

    Args:
        action: ContextAction with filter params
        doc_ids: Current accumulated doc_ids
        cvm: ConversationModel

    Returns:
        Filtered list of doc_ids
    """
    # Future implementation - for now just pass through
    logger.debug(f"filter: Filter action not yet implemented, passing through {len(doc_ids)} docs")
    return doc_ids
