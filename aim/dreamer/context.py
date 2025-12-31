# aim/dreamer/context.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Context preparation DSL executor for pipeline steps.

This module implements the step-level context DSL that allows declarative
specification of how each step's input context is assembled.

Uses the unified Memory DSL executor from memory_dsl.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aim.conversation.model import ConversationModel
    from aim.dreamer.models import MemoryAction, StepDefinition, PipelineState

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

    Uses the unified Memory DSL executor for action processing.

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

    from .memory_dsl import execute_memory_actions

    # Execute context DSL using unified executor
    doc_ids = execute_memory_actions(
        actions=step_def.context,
        state=state,
        cvm=cvm,
        query_text=state.query_text,
    )

    logger.debug(f"Context DSL produced {len(doc_ids)} doc_ids for step {step_def.id}")
    return doc_ids, True  # True = this is initial context from DSL


# Legacy functions kept for backward compatibility with tests
# These delegate to memory_dsl functions internally

def _load_conversation(
    action: "MemoryAction",
    state: "PipelineState",
    cvm: "ConversationModel",
) -> list[str]:
    """Load documents from a conversation.

    Delegates to memory_dsl._load_conversation.
    """
    from .memory_dsl import _load_conversation as dsl_load_conversation
    return dsl_load_conversation(action, state, cvm)


def _query_documents(
    action: "MemoryAction",
    state: "PipelineState",
    cvm: "ConversationModel",
) -> list[str]:
    """Query documents from the index.

    Maps to memory_dsl._get_memory (renamed from query).
    """
    from .memory_dsl import _get_memory
    return _get_memory(action, state, cvm)


def _sort_doc_ids(
    action: "MemoryAction",
    doc_ids: list[str],
    cvm: "ConversationModel",
) -> list[str]:
    """Sort accumulated doc_ids by timestamp or other criteria.

    Delegates to memory_dsl._sort_docs.
    """
    from .memory_dsl import _sort_docs
    return _sort_docs(action, doc_ids, cvm)


def _filter_doc_ids(
    action: "MemoryAction",
    doc_ids: list[str],
    cvm: "ConversationModel",
) -> list[str]:
    """Filter accumulated doc_ids.

    Delegates to memory_dsl._filter_docs.
    """
    from .memory_dsl import _filter_docs
    return _filter_docs(action, doc_ids, cvm)
