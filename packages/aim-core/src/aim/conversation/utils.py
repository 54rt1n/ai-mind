# aim/conversation/utils.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

"""Utilities for conversation management and indexing."""

import logging
from pathlib import Path
from typing import Optional

from aim.conversation.loader import ConversationLoader
from aim.conversation.index import SearchIndex
from aim.constants import CHUNK_LEVEL_FULL, CHUNK_LEVEL_768, CHUNK_LEVEL_256

logger = logging.getLogger(__name__)


def rebuild_agent_index(
    agent_id: str,
    embedding_model: str,
    device: str = "cpu",
    batch_size: int = 64,
    full: bool = False,
    memory_base: str = "memory"
) -> dict:
    """
    Rebuild or incrementally update index for a single agent.

    Args:
        agent_id: The agent ID to rebuild index for
        embedding_model: Model to use for embeddings
        device: Device to use for indexing (cpu/cuda)
        batch_size: Batch size for indexing operations
        full: Force full rebuild instead of incremental update
        memory_base: Base directory for memory storage

    Returns:
        dict with:
            - mode: "full" or "incremental"
            - added: int (if incremental)
            - updated: int (if incremental)
            - deleted: int (if incremental)
            - total_messages: int
            - total_entries: int
            - chunk_stats: dict with counts per chunk level
    """
    # Resolve paths from agent_id
    conversations_dir = Path(memory_base) / agent_id / "conversations"
    index_dir = Path(memory_base) / agent_id / "indices"

    # Validate conversations directory exists
    if not conversations_dir.exists():
        raise FileNotFoundError(f"Conversations directory not found: {conversations_dir}")

    # Load conversations
    loader = ConversationLoader(str(conversations_dir))
    messages = loader.load_all()

    if len(messages) == 0:
        raise ValueError(f"No messages found in {conversations_dir}")

    # Convert to index documents
    documents = [msg.to_dict() for msg in messages]

    # Check if index exists
    index_exists = index_dir.exists() and any(index_dir.iterdir()) if index_dir.exists() else False

    # Determine rebuild mode
    if full or not index_exists:
        # Full rebuild
        index = SearchIndex(index_path=index_dir, embedding_model=embedding_model, device=device)
        index.rebuild(documents, use_tqdm=False)

        # Get chunk stats
        chunk_stats = _get_chunk_stats(index)

        return {
            "mode": "full",
            "total_messages": len(messages),
            "total_entries": sum(chunk_stats.values()),
            "chunk_stats": chunk_stats
        }
    else:
        # Incremental update
        index = SearchIndex(index_path=index_dir, embedding_model=embedding_model, device=device)
        added_count, updated_count, deleted_count = index.incremental_update(
            documents, use_tqdm=False, batch_size=batch_size
        )

        # Get chunk stats
        chunk_stats = _get_chunk_stats(index)

        return {
            "mode": "incremental",
            "added": added_count,
            "updated": updated_count,
            "deleted": deleted_count,
            "total_messages": len(messages),
            "total_entries": sum(chunk_stats.values()),
            "chunk_stats": chunk_stats
        }


def _get_chunk_stats(index: SearchIndex) -> dict:
    """
    Get statistics about chunk levels in the index.

    Args:
        index: The SearchIndex instance

    Returns:
        dict with counts per chunk level:
            - full: number of parent documents
            - 768: number of 768-token chunks
            - 256: number of 256-token chunks
    """
    searcher = index.index.searcher()
    stats = {}

    for level in [CHUNK_LEVEL_FULL, CHUNK_LEVEL_768, CHUNK_LEVEL_256]:
        query = index.index.parse_query(query=level, default_field_names=["chunk_level"])
        results = searcher.search(query, limit=1)
        stats[level] = results.count

    return {
        "full": stats.get(CHUNK_LEVEL_FULL, 0),
        "768": stats.get(CHUNK_LEVEL_768, 0),
        "256": stats.get(CHUNK_LEVEL_256, 0)
    }
