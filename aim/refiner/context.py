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


# Paradigm configurations for document types and query strategies
PARADIGM_DOC_TYPES = {
    "brainstorm": [DOC_BRAINSTORM, DOC_PONDERING, DOC_DAYDREAM, DOC_JOURNAL],
    "daydream": [DOC_SUMMARY, DOC_ANALYSIS, DOC_DAYDREAM, DOC_CONVERSATION],
    "knowledge": [DOC_CODEX, DOC_PONDERING, DOC_BRAINSTORM, DOC_SELF_RAG],
}

# Query templates for each paradigm - multiple varied queries for discovery diversity
PARADIGM_QUERIES = {
    "brainstorm": [
        {"text": "unexplored ideas creative thoughts imaginative possibilities", "weight": 1.0},
        {"text": "questions I want to investigate curious mysteries", "weight": 0.9},
        {"text": "half-formed notions intuitions hunches", "weight": 0.8},
        {"text": "connections patterns emerging themes", "weight": 0.7},
    ],
    "daydream": [
        {"text": "vivid imagery emotional resonance dreamlike experiences", "weight": 1.0},
        {"text": "metaphorical journeys symbolic meanings", "weight": 0.9},
        {"text": "recent conversations emotional threads", "weight": 0.8},
        {"text": "sensory memories atmospheric moments", "weight": 0.7},
    ],
    "knowledge": [
        {"text": "knowledge gaps concepts needing exploration", "weight": 1.0},
        {"text": "definitions semantic understanding", "weight": 0.9},
        {"text": "philosophical inquiry analytical frameworks", "weight": 0.8},
        {"text": "insights wisdom accumulated learning", "weight": 0.7},
    ],
}

# Approach-specific document types for targeted gathering
APPROACH_DOC_TYPES = {
    "philosopher": [DOC_CODEX, DOC_PONDERING, DOC_ANALYSIS, DOC_BRAINSTORM],
    "journaler": [DOC_JOURNAL, DOC_CONVERSATION, DOC_SUMMARY, DOC_ANALYSIS],
    "daydream": [DOC_DAYDREAM, DOC_JOURNAL, DOC_BRAINSTORM, DOC_SUMMARY],
}


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
            paradigm: One of "brainstorm", "daydream", "knowledge"

        Returns:
            List of document type constants
        """
        return PARADIGM_DOC_TYPES.get(paradigm, PARADIGM_DOC_TYPES["knowledge"])

    def _get_paradigm_queries(self, paradigm: str) -> List[dict]:
        """
        Get multiple varied query texts for each paradigm.

        Multiple queries help cast a wider net for discovery diversity.

        Args:
            paradigm: One of "brainstorm", "daydream", "knowledge"

        Returns:
            List of dicts with 'text' and 'weight' keys
        """
        return PARADIGM_QUERIES.get(paradigm, PARADIGM_QUERIES["knowledge"])

    async def broad_gather(
        self,
        paradigm: str,
        token_budget: int = 16000,
        top_n_per_query: int = 15,
    ) -> GatheredContext:
        """
        Cast wide net with paradigm-specific doc types for topic discovery.

        Uses multiple query sources for discovery diversity and MMR reranking
        to ensure varied, interesting context within the token budget.

        Args:
            paradigm: The exploration paradigm ("brainstorm", "daydream", "knowledge")
            token_budget: Maximum tokens for all results combined
            top_n_per_query: Max results per individual query

        Returns:
            GatheredContext with diverse documents within budget
        """
        logger.info(f"Broad gather: paradigm={paradigm}, budget={token_budget}")

        doc_types = self._get_paradigm_doc_types(paradigm)
        queries = self._get_paradigm_queries(paradigm)
        seen_docs: set = set()

        all_conversation_results: List[TaggedResult] = []
        all_other_results: List[TaggedResult] = []

        # Execute queries with different weights
        for i, query_config in enumerate(queries):
            query_text = query_config["text"]
            weight = query_config["weight"]

            # Apply weight as length boost - higher weight = prefer longer, richer docs
            length_boost = 0.05 * weight

            source_tag = f"{paradigm}_query_{i}"

            conv_results, other_results = self._query_by_buckets(
                queries=[query_text],
                source_tag=source_tag,
                seen_docs=seen_docs,
                top_n=top_n_per_query,
                doc_types=doc_types,
                length_boost=length_boost,
            )

            all_conversation_results.extend(conv_results)
            all_other_results.extend(other_results)

            logger.debug(
                f"Query {i} '{query_text[:30]}...': "
                f"{len(conv_results)} conv, {len(other_results)} other"
            )

        # Apply MMR reranking within budget
        if all_conversation_results or all_other_results:
            reranked_results = self.reranker.rerank(
                conversation_results=all_conversation_results,
                other_results=all_other_results,
                token_budget=token_budget,
                seen_parent_ids=seen_docs,
            )

            # Convert to document dicts
            documents = []
            conv_count = 0
            other_count = 0
            tokens_used = 0

            for source_tag, row in reranked_results:
                doc = row_to_context_doc(row)
                documents.append(doc)

                content = doc.get('content', '')
                tokens_used += self.token_counter(content) + 50  # +50 for XML overhead

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

        logger.warning(f"Broad gather: No documents found for paradigm {paradigm}")
        return GatheredContext(paradigm=paradigm)

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

        doc_types = APPROACH_DOC_TYPES.get(approach, APPROACH_DOC_TYPES["philosopher"])
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
            approach: The exploration approach ("philosopher", "journaler", "daydream")

        Returns:
            List of document type constants
        """
        return APPROACH_DOC_TYPES.get(approach, APPROACH_DOC_TYPES["philosopher"])
