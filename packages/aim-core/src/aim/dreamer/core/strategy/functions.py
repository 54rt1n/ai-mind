# aim/dreamer/core/strategy/functions.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Shared helper functions for step execution strategies."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import ScenarioExecutor
    from ..models import NewStepDefinition

logger = logging.getLogger(__name__)


def execute_seed_actions(executor: "ScenarioExecutor") -> None:
    """Execute scenario-level seed actions once and populate memory_refs.

    Seed actions run before the first step executes. They provide baseline
    context that step-level DSL can build on.

    Args:
        executor: The scenario executor with framework, state, and CVM
    """
    from ..memory_dsl import execute_memory_actions
    from ..state import DocRef

    if executor.state.seed_loaded:
        return

    if not executor.framework.seed:
        executor.state.seed_loaded = True
        return

    doc_ids = execute_memory_actions(
        actions=executor.framework.seed,
        state=executor.state,
        cvm=executor.cvm,
        query_text=executor.state.query_text,
    )

    existing_doc_ids = {ref.doc_id for ref in executor.state.memory_refs}

    for doc_id in doc_ids:
        if doc_id in existing_doc_ids:
            continue

        doc = executor.cvm.get_by_doc_id(doc_id)
        if doc:
            ref = DocRef(
                doc_id=doc_id,
                document_type=doc.get('document_type'),
            )
            executor.state.memory_refs.append(ref)
            existing_doc_ids.add(doc_id)

    executor.state.seed_loaded = True
    logger.debug(f"Loaded {len(doc_ids)} docs from scenario seed")


def execute_context_actions(
    executor: "ScenarioExecutor",
    step_def: "NewStepDefinition",
) -> None:
    """Execute memory DSL context actions and update state.memory_refs.

    Accumulates documents into memory_refs. Does NOT auto-clear - memory_refs
    persists across steps until the DSL explicitly uses 'flush' or 'clear'.

    Uses the memory_dsl.execute_memory_actions() function to process
    the context actions and retrieve documents.

    Args:
        executor: The scenario executor with state and CVM
        step_def: Step definition with optional context actions
    """
    from ..memory_dsl import execute_memory_actions
    from ..state import DocRef

    # NOTE: Do NOT auto-clear memory_refs here.
    # The DSL has explicit 'flush' and 'clear' actions for that purpose.
    # Auto-clearing defeats accumulated context across steps.

    # Execute memory DSL - returns doc_ids list
    doc_ids = execute_memory_actions(
        actions=step_def.context,
        state=executor.state,
        cvm=executor.cvm,
        query_text=executor.state.query_text,
    )

    # Convert doc_ids to DocRefs
    for doc_id in doc_ids:
        # Get document metadata from CVM
        doc = executor.cvm.get_by_doc_id(doc_id)
        if doc:
            ref = DocRef(
                doc_id=doc_id,
                document_type=doc.get('document_type'),
            )
            executor.state.memory_refs.append(ref)

    logger.debug(f"Loaded {len(doc_ids)} docs from context DSL")


def load_memory_docs(executor: "ScenarioExecutor") -> list[dict]:
    """Load memory documents from state.memory_refs.

    Retrieves full document dictionaries from the CVM for each
    reference in the executor's memory_refs list.

    Args:
        executor: The scenario executor with state and CVM

    Returns:
        List of document dicts with 'content' field
    """
    docs = []
    for ref in executor.state.memory_refs:
        doc = executor.cvm.get_by_doc_id(ref.doc_id)
        if doc:
            docs.append(doc)
    return docs


def load_step_docs(executor: "ScenarioExecutor") -> list[dict]:
    """Load prior step documents from state.step_doc_ids.

    Retrieves full document dictionaries from the CVM for each
    document ID created by prior steps in this scenario.

    Args:
        executor: The scenario executor with state and CVM

    Returns:
        List of document dicts with 'content' and 'think' fields
    """
    docs = []
    for doc_id in executor.state.step_doc_ids:
        doc = executor.cvm.get_by_doc_id(doc_id)
        if doc:
            docs.append({
                'content': doc.get('content', ''),
                'think': doc.get('think'),
            })
    return docs
