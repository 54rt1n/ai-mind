# aim/refiner/context.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Context gathering strategies for the refiner module.

Implements sophisticated dual-bucket querying with token-budget awareness,
MMR reranking for diversity, and paradigm-specific document retrieval.
Modeled after the XMLMemoryTurnStrategy for optimal memory retrieval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional, List, Tuple

import pandas as pd

from aim.conversation.rerank import MemoryReranker, TaggedResult
from aim.constants import (
    DOC_CONVERSATION,
    DOC_BRAINSTORM,
    DOC_PONDERING,
    DOC_DAYDREAM,
    DOC_SUMMARY,
    DOC_ANALYSIS,
    DOC_CODEX,
    DOC_SELF_RAG,
    DOC_JOURNAL,
    DOC_INSPIRATION,
    DOC_UNDERSTANDING,
    CHUNK_LEVEL_256,
    CHUNK_LEVEL_768,
)

if TYPE_CHECKING:
    from aim.conversation.model import ConversationModel

logger = logging.getLogger(__name__)


# Fields to include when converting DataFrame rows to context documents.
# Only include fields that are needed for context - excludes index_a/index_b
# (numpy embeddings) and other internal fields that can't be JSON serialized.
CONTEXT_DOC_FIELDS = {
    'doc_id', 'document_type', 'content', 'think', 'role',
    'conversation_id', 'persona_id', 'user_id', 'branch', 'sequence_no',
    'timestamp', 'weight', 'score',
}


def row_to_context_doc(row) -> dict:
    """
    Convert a DataFrame row to a context document dict.

    Only includes fields that are needed for context and can be JSON serialized.
    Excludes numpy array fields like index_a/index_b (embeddings).

    Args:
        row: pandas Series or dict-like object

    Returns:
        dict with only serializable context fields
    """
    source = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
    return {k: v for k, v in source.items() if k in CONTEXT_DOC_FIELDS and v is not None}


# Document type name to constant mapping
DOC_TYPE_MAP = {
    "brainstorm": DOC_BRAINSTORM,
    "pondering": DOC_PONDERING,
    "daydream": DOC_DAYDREAM,
    "journal": DOC_JOURNAL,
    "inspiration": DOC_INSPIRATION,
    "understanding": DOC_UNDERSTANDING,
    "summary": DOC_SUMMARY,
    "analysis": DOC_ANALYSIS,
    "codex": DOC_CODEX,
    "self-rag": DOC_SELF_RAG,
    "conversation": DOC_CONVERSATION,
}


def _resolve_doc_types(names: List[str]) -> List[str]:
    """Convert doc type names to constants."""
    return [DOC_TYPE_MAP.get(name, name) for name in names]


def get_paradigm_doc_types(paradigm: str) -> List[str]:
    """Get document types for a paradigm from config."""
    from aim.refiner.paradigm_config import get_paradigm_config

    config = get_paradigm_config(paradigm)
    if config:
        return _resolve_doc_types(config.doc_types)

    # Fallback defaults
    logger.warning(f"No config for paradigm '{paradigm}', using defaults")
    return [DOC_BRAINSTORM, DOC_PONDERING]


def get_paradigm_queries(paradigm: str) -> List[dict]:
    """Get queries for a paradigm from config."""
    from aim.refiner.paradigm_config import get_paradigm_config

    config = get_paradigm_config(paradigm)
    if config:
        return config.queries

    # Fallback defaults
    return [{"text": "unexplored ideas", "weight": 1.0}]


def get_approach_doc_types(approach: str, paradigm: str = "") -> List[str]:
    """Get document types for an approach from config."""
    from aim.refiner.paradigm_config import get_paradigm_config

    # Try to get from the paradigm config first
    if paradigm:
        config = get_paradigm_config(paradigm)
        if config:
            doc_types = config.get_approach_doc_types(approach)
            return _resolve_doc_types(doc_types)

    # Try loading the approach as a paradigm (e.g., "critique" approach uses critique config)
    config = get_paradigm_config(approach)
    if config:
        doc_types = config.get_approach_doc_types(approach)
        return _resolve_doc_types(doc_types)

    # Fallback defaults
    logger.warning(f"No config for approach '{approach}', using defaults")
    return [DOC_PONDERING, DOC_BRAINSTORM]


@dataclass
class GatheredContext:
    """Container for gathered context documents with metadata."""

    documents: List[dict] = field(default_factory=list)
    paradigm: str = "unknown"
    tokens_used: int = 0
    conversation_count: int = 0
    other_count: int = 0

    @property
    def empty(self) -> bool:
        """Check if no documents were gathered."""
        return len(self.documents) == 0

    @property
    def doc_count(self) -> int:
        """Total document count."""
        return len(self.documents)

    def to_records(self) -> List[dict]:
        """Return documents as list of dicts."""
        return self.documents


class ContextGatherer:
    """
    Gathers context documents from the ConversationModel using sophisticated
    dual-bucket querying with MMR reranking for diversity.

    Implements the same patterns as XMLMemoryTurnStrategy:
    - Dual queries: conversations at chunk_768, others at chunk_256
    - Token-budget aware searching
    - Multiple query sources with different weights
    - MMR reranking for diversity within budget

    Provides strategies for:
    - Broad gathering: Cast wide net with paradigm-specific doc types
    - Targeted gathering: Focused query using topic and approach
    """

    def __init__(
        self,
        cvm: "ConversationModel",
        token_counter: Callable[[str], int],
        lambda_param: float = 0.7,
        conversation_budget_ratio: float = 0.5,
    ):
        """
        Initialize the ContextGatherer.

        Args:
            cvm: ConversationModel for document queries
            token_counter: Function to count tokens in a string
            lambda_param: MMR parameter (0=max diversity, 1=pure relevance)
            conversation_budget_ratio: Portion of budget for conversations
        """
        self.cvm = cvm
        self.token_counter = token_counter
        self.reranker = MemoryReranker(
            token_counter=token_counter,
            lambda_param=lambda_param,
            conversation_budget_ratio=conversation_budget_ratio,
        )

    def _query_by_buckets(
        self,
        queries: List[str],
        source_tag: str,
        seen_docs: set,
        top_n: int,
        doc_types: Optional[List[str]] = None,
        length_boost: float = 0.0,
    ) -> Tuple[List[TaggedResult], List[TaggedResult]]:
        """
        Execute dual queries: conversations at chunk_768, others at chunk_256.

        This mirrors the XMLMemoryTurnStrategy._query_by_buckets pattern for
        optimal retrieval across different document types.

        Args:
            queries: Query texts to search for
            source_tag: Tag for this source (paradigm_primary, paradigm_secondary, etc.)
            seen_docs: Doc IDs to filter out
            top_n: Max results per bucket
            doc_types: Optional filter for document types
            length_boost: Length boost factor

        Returns:
            (conversation_results, other_results) as lists of (source_tag, row)
        """
        conversation_results: List[TaggedResult] = []
        other_results: List[TaggedResult] = []

        # Query 1: Conversations at chunk_768 (larger context for dialog)
        conv_df = self.cvm.query(
            queries,
            filter_doc_ids=seen_docs,
            top_n=top_n,
            filter_metadocs=True,
            query_document_type=DOC_CONVERSATION,
            chunk_level=CHUNK_LEVEL_768,
            length_boost_factor=length_boost,
        )
        if not conv_df.empty:
            for _, row in conv_df.iterrows():
                conversation_results.append((source_tag, row))

        # Query 2: Other docs at chunk_256 (denser for analytical content)
        other_df = self.cvm.query(
            queries,
            filter_doc_ids=seen_docs,
            top_n=top_n,
            filter_metadocs=True,
            query_document_type=doc_types,
            chunk_level=CHUNK_LEVEL_256,
            length_boost_factor=length_boost,
        )
        if not other_df.empty:
            # Filter out conversations if they slipped through
            other_df = other_df[other_df['document_type'] != DOC_CONVERSATION]
            for _, row in other_df.iterrows():
                other_results.append((source_tag, row))

        return conversation_results, other_results

    def _get_paradigm_doc_types(self, paradigm: str) -> List[str]:
        """
        Get appropriate document types for a paradigm.

        Args:
            paradigm: One of "brainstorm", "daydream", "knowledge", "critique"

        Returns:
            List of document type constants
        """
        return get_paradigm_doc_types(paradigm)

    def _get_paradigm_queries(self, paradigm: str) -> List[dict]:
        """
        Get multiple varied query texts for each paradigm.

        Multiple queries help cast a wider net for discovery diversity.

        Args:
            paradigm: One of "brainstorm", "daydream", "knowledge", "critique"

        Returns:
            List of dicts with 'text' and 'weight' keys
        """
        return get_paradigm_queries(paradigm)

    async def broad_gather(
        self,
        paradigm: str,
        token_budget: int = 16000,
        top_n: int = 30,
    ) -> GatheredContext:
        """
        Randomly sample documents of paradigm-specific types for topic discovery.

        Uses random sampling rather than semantic search to ensure varied,
        unexpected context for exploration.

        Args:
            paradigm: The exploration paradigm ("brainstorm", "daydream", "knowledge", "critique")
            token_budget: Maximum tokens for all results combined
            top_n: Max documents to sample

        Returns:
            GatheredContext with randomly sampled documents within budget
        """
        logger.info(f"Broad gather: paradigm={paradigm}, budget={token_budget}")

        doc_types = self._get_paradigm_doc_types(paradigm)

        # Random sample by document type - no semantic search
        results = self.cvm.sample_by_type(doc_types=doc_types, top_n=top_n)

        if results.empty:
            logger.warning(f"Broad gather: No documents found for paradigm {paradigm}")
            return GatheredContext(paradigm=paradigm)

        # Convert to document dicts within token budget
        documents = []
        conv_count = 0
        other_count = 0
        tokens_used = 0

        for _, row in results.iterrows():
            doc = row_to_context_doc(row)
            content = doc.get('content', '')
            doc_tokens = self.token_counter(content) + 50  # +50 for XML overhead

            # Stop if we'd exceed budget
            if tokens_used + doc_tokens > token_budget:
                break

            documents.append(doc)
            tokens_used += doc_tokens

            if doc.get('document_type') == DOC_CONVERSATION:
                conv_count += 1
            else:
                other_count += 1

        logger.info(
            f"Broad gather complete: {len(documents)} docs "
            f"({conv_count} conv + {other_count} other), "
            f"{tokens_used}/{token_budget} tokens"
        )

        return GatheredContext(
            documents=documents,
            paradigm=paradigm,
            tokens_used=tokens_used,
            conversation_count=conv_count,
            other_count=other_count,
        )

    async def targeted_gather(
        self,
        topic: str,
        approach: str,
        token_budget: int = 16000,
        top_n: int = 20,
    ) -> GatheredContext:
        """
        Focused query using topic and approach-appropriate doc types.

        Unlike broad_gather which casts a wide net, this focuses on
        documents specifically related to the chosen topic.

        Args:
            topic: The topic to search for
            approach: The exploration approach ("philosopher", "journaler", "daydream")
            token_budget: Maximum tokens for results
            top_n: Maximum number of documents to retrieve

        Returns:
            GatheredContext with targeted documents
        """
        logger.info(f"Targeted gather: topic='{topic}', approach={approach}")

        doc_types = get_approach_doc_types(approach)
        seen_docs: set = set()

        # Single focused query with the topic
        conv_results, other_results = self._query_by_buckets(
            queries=[topic],
            source_tag=f"targeted_{approach}",
            seen_docs=seen_docs,
            top_n=top_n,
            doc_types=doc_types,
            length_boost=0.05,
        )

        # Apply MMR reranking
        if conv_results or other_results:
            reranked_results = self.reranker.rerank(
                conversation_results=conv_results,
                other_results=other_results,
                token_budget=token_budget,
                seen_parent_ids=seen_docs,
            )

            documents = []
            conv_count = 0
            other_count = 0
            tokens_used = 0

            for source_tag, row in reranked_results:
                doc = row_to_context_doc(row)
                documents.append(doc)

                content = doc.get('content', '')
                tokens_used += self.token_counter(content) + 50

                if doc.get('document_type') == DOC_CONVERSATION:
                    conv_count += 1
                else:
                    other_count += 1

            logger.info(
                f"Targeted gather complete: {len(documents)} docs for '{topic}', "
                f"{tokens_used}/{token_budget} tokens"
            )

            return GatheredContext(
                documents=documents,
                paradigm=approach,
                tokens_used=tokens_used,
                conversation_count=conv_count,
                other_count=other_count,
            )

        logger.warning(f"Targeted gather: No documents found for topic '{topic}'")
        return GatheredContext(paradigm=approach)

    def get_doc_types_for_approach(self, approach: str) -> List[str]:
        """
        Get appropriate document types for an exploration approach.

        Args:
            approach: The exploration approach ("philosopher", "journaler", "daydream", "critique")

        Returns:
            List of document type constants
        """
        return get_approach_doc_types(approach)
