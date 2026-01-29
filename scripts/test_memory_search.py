#!/usr/bin/env python3
"""
Test script for debugging conversation memory search.

This script loads Andi's conversation model from environment configuration
and allows testing search queries to diagnose why terms like "Turtle" might
not be matching.

Usage:
    python scripts/test_memory_search.py "Turtle"
    python scripts/test_memory_search.py "Turtle" --debug
    python scripts/test_memory_search.py "Turtle" --persona Andi --top-n 20
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add packages to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "packages" / "aim-core" / "src"))

from aim.conversation.model import ConversationModel
from aim.conversation.index import clean_query_text, boost_query_terms
from aim.conversation.blacklist import STOPWORDS, BLACKLIST_WORDS, CONTRACTIONS


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def analyze_query_processing(query_text: str) -> dict:
    """Analyze what happens to a query during processing."""

    print("\n" + "=" * 80)
    print(f"QUERY PROCESSING ANALYSIS: '{query_text}'")
    print("=" * 80)

    # Check if any terms are contractions, stopwords, or blacklisted
    words = query_text.lower().split()

    print("\n1. TERM FILTERING CHECK:")
    print("-" * 80)
    for word in words:
        clean_word = word.strip('.,!?;:"\'-')
        filters = []
        if clean_word in CONTRACTIONS:
            filters.append("CONTRACTION")
        if clean_word in STOPWORDS:
            filters.append("STOPWORD")
        if clean_word in BLACKLIST_WORDS:
            filters.append("BLACKLISTED")

        status = f"❌ FILTERED ({', '.join(filters)})" if filters else "✓ PASSES"
        print(f"  '{word}' → {status}")

    # Step 1: Clean query text
    print("\n2. CLEAN QUERY TEXT:")
    print("-" * 80)
    cleaned = clean_query_text(query_text)
    print(f"  Original: '{query_text}'")
    print(f"  Cleaned:  '{cleaned}'")

    # Step 2: Boost query terms
    print("\n3. BOOST QUERY TERMS:")
    print("-" * 80)
    boosted = boost_query_terms(cleaned, keyword_boost=2.0)
    print(f"  Input:   '{cleaned}'")
    print(f"  Boosted: '{boosted}'")

    if not boosted.strip():
        print("  ⚠️  WARNING: All terms were filtered! Query will be empty!")

    return {
        "original": query_text,
        "cleaned": cleaned,
        "boosted": boosted,
        "has_results": bool(boosted.strip())
    }


async def test_search(
    query_text: str,
    persona_id: Optional[str] = None,
    top_n: int = 10,
    debug: bool = False,
    env_file: Optional[str] = None
) -> None:
    """Test a search query against the conversation model."""

    # Set debug logging if requested
    if debug:
        logging.getLogger("aim").setLevel(logging.DEBUG)

    # Load ChatConfig from environment
    print("\nLoading configuration...")
    from aim.config import ChatConfig

    if env_file:
        config = ChatConfig.from_env(env_file)
    else:
        config = ChatConfig.from_env()  # Uses .env file

    # Override persona_id if specified
    if persona_id is not None:
        config.persona_id = persona_id

    print(f"Using persona: {config.persona_id}")
    print(f"Memory base path: {config.memory_path}")

    # Construct full memory path (memory_path + persona_id)
    full_memory_path = Path(config.memory_path) / config.persona_id
    print(f"Full memory path: {full_memory_path}")

    if not full_memory_path.exists():
        logger.error(f"Memory path does not exist: {full_memory_path}")
        print("\nExpected directory structure:")
        print(f"  {full_memory_path}/conversations/")
        print(f"  {full_memory_path}/indices/")
        return

    # Analyze query processing
    analysis = analyze_query_processing(query_text)

    if not analysis["has_results"]:
        print("\n⚠️  All query terms were filtered out. Search will return no results.")
        print("This is the problem! Your search term is being removed during preprocessing.")
        return

    # Load conversation model using canonical from_config() method
    print("\n" + "=" * 80)
    print("LOADING CONVERSATION MODEL")
    print("=" * 80)

    try:
        # Use from_config() - the canonical initialization method
        # This automatically constructs memory_path as {config.memory_path}/{config.persona_id}/
        cvm = ConversationModel.from_config(config, skip_vectorizer=True)

        print(f"\n✓ Model loaded successfully")
        print(f"  Memory path: {cvm.memory_path}")

        # Count conversation files
        jsonl_files = list(cvm.loader.conversations_dir.glob("*.jsonl"))
        print(f"  Conversation files: {len(jsonl_files)}")

        # Show index location (always, not just debug)
        print(f"  Conversations directory: {cvm.loader.conversations_dir}")
        print(f"  Index directory: {cvm.index.index_path}")

        # Check if index exists
        if cvm.index.index_path.exists():
            print(f"  ✓ Index directory exists")
            # Check for Tantivy index
            tantivy_meta = cvm.index.index_path / "meta.json"
            if tantivy_meta.exists():
                print(f"  ✓ Tantivy index found (meta.json exists)")
            else:
                print(f"  ⚠️  WARNING: Tantivy meta.json not found - index may not be built!")
        else:
            print(f"  ❌ WARNING: Index directory does not exist!")

        if debug:
            # Show more detailed index info
            import os
            if cvm.index.index_path.exists():
                index_files = list(cvm.index.index_path.iterdir())
                print(f"  Index files: {[f.name for f in index_files]}")

    except Exception as e:
        logger.error(f"Failed to load conversation model: {e}", exc_info=debug)
        return

    # Perform search
    print("\n" + "=" * 80)
    print("SEARCH RESULTS")
    print("=" * 80)

    try:
        results = cvm.query(
            query_texts=[query_text],
            top_n=top_n,
            keyword_boost=2.0
        )

        # Results is a DataFrame, check if empty
        num_results = len(results) if not results.empty else 0
        print(f"\nFound {num_results} results:")
        print("-" * 80)

        if results.empty:
            print("❌ No results found!")
            print("\nPossible reasons:")
            print("  1. The term doesn't exist in any messages")
            print("  2. The term was filtered during query processing")
            print("  3. Tantivy stemming is causing a mismatch")
            print("  4. Additional filters (persona_id, document_type, etc.) excluded all matches")

        # Iterate over DataFrame rows
        for i, (idx, row) in enumerate(results.iterrows(), 1):
            score = row.get('score', 0.0)
            print(f"\n{i}. Score: {score:.4f}")
            print(f"   Doc ID: {row['doc_id']}")
            print(f"   Persona: {row['persona_id']}")
            print(f"   Document Type: {row['document_type']}")
            print(f"   Conversation: {row['conversation_id']}")

            if 'date' in row:
                print(f"   Date: {row['date']}")

            # Show content snippet
            content = row['content']
            if len(content) > 200:
                # Try to find the query term in context
                query_lower = query_text.lower()
                content_lower = content.lower()

                if query_lower in content_lower:
                    # Show snippet around the match
                    pos = content_lower.find(query_lower)
                    start = max(0, pos - 100)
                    end = min(len(content), pos + len(query_text) + 100)
                    snippet = content[start:end]
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(content):
                        snippet = snippet + "..."
                    print(f"   Content: {snippet}")
                else:
                    # Just show beginning
                    print(f"   Content: {content[:200]}...")
            else:
                print(f"   Content: {content}")

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=debug)
        return

    # Additional diagnostics if no results
    if results.empty:
        print("\n" + "=" * 80)
        print("ADDITIONAL DIAGNOSTICS")
        print("=" * 80)

        print("\nLoading all messages for direct content search...")
        print("(This may take a moment for large conversation histories)")

        try:
            # Load all messages from JSONL files
            all_messages = cvm.loader.load_all(use_tqdm=False)

            # Try to find messages containing the raw term (case-insensitive)
            query_lower = query_text.lower()
            matching_messages = [
                msg for msg in all_messages
                if query_lower in msg.content.lower()
            ]

            print(f"\nDirect content search (case-insensitive):")
            print(f"  Total messages scanned: {len(all_messages)}")
            print(f"  Found {len(matching_messages)} messages containing '{query_text}'")

            if matching_messages:
                print("\n  Sample matching messages:")
                for msg in matching_messages[:3]:
                    print(f"    - {msg.doc_id}: {msg.persona_id} ({msg.document_type})")
                    snippet_start = msg.content.lower().find(query_lower)
                    snippet = msg.content[max(0, snippet_start - 50):snippet_start + len(query_text) + 50]
                    print(f"      ...{snippet}...")

                print("\n  ⚠️ Messages exist but search didn't find them!")
                print("  This suggests an issue with:")
                print("    - Tantivy indexing (messages not indexed)")
                print("    - Query parsing (term filtered or transformed incorrectly)")
                print("    - Filter configuration (results filtered out)")

        except Exception as e:
            logger.error(f"Failed to load all messages for diagnostics: {e}", exc_info=debug)


def main():
    parser = argparse.ArgumentParser(
        description="Test conversation memory search queries for Andi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Turtle"
  %(prog)s "Turtle" --debug
  %(prog)s "Turtle" --persona Andi --top-n 20
  %(prog)s "lighthouse" --env-file .env.andi

Environment variables (from .env file):
  MEMORY_PATH    Root directory for persona memories (default: "memory")
  PERSONA_ID     Which persona to load (default: "assistant")

Memory location will be: {MEMORY_PATH}/{PERSONA_ID}/
For Andi: memory/Andi/conversations/ and memory/Andi/indices/
        """
    )

    parser.add_argument(
        "query",
        help="Search query to test"
    )

    parser.add_argument(
        "--persona",
        default=None,
        help="Persona ID to search (overrides PERSONA_ID from .env)"
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of results to return (default: 10)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to .env file (default: .env in current directory)"
    )

    args = parser.parse_args()

    # Run async search
    asyncio.run(test_search(
        query_text=args.query,
        persona_id=args.persona,
        top_n=args.top_n,
        debug=args.debug,
        env_file=args.env_file
    ))


if __name__ == "__main__":
    main()
