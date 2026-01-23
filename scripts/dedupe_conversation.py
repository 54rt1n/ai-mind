#!/usr/bin/env python3
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Remove duplicate events from agent conversation history.

Walks through conversation entries in order, tracking event_ids.
- Entries where ALL event_ids are duplicates are removed entirely.
- Entries with SOME duplicate event_ids are rebuilt with only the new content.

For narratives, content is split by "[==" header pattern.
For other events, content is split by "\\n\\n".

Usage:
    # Dry run - show what would be changed
    python scripts/dedupe_conversation.py corroded

    # Actually apply changes
    python scripts/dedupe_conversation.py corroded --apply

    # Verbose output
    python scripts/dedupe_conversation.py corroded -v
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "aim-mud" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "aim-core" / "src"))

try:
    from redis.asyncio import Redis
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def get_conversation_key(agent_id: str) -> str:
    return f"mud:agent:{agent_id}:conversation"


def truncate(text: str, max_len: int = 60) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", "")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def split_content_parts(content: str, event_types: list[str]) -> list[str]:
    """Split content into parts matching event_ids.

    For narratives, split by '[==' header pattern.
    For other events, split by '\\n\\n'.
    """
    if not content or not event_types:
        return []

    # Check if this is narrative content (has [== headers)
    if "[==" in content:
        # Split by [== but keep the delimiter
        parts = re.split(r'(?=\[==)', content)
        # Filter empty parts and strip
        parts = [p.strip() for p in parts if p.strip()]
        return parts
    else:
        # Regular split by double newline
        parts = content.split("\n\n")
        return [p.strip() for p in parts if p.strip()]


def rebuild_entry(entry: dict, keep_indices: list[int]) -> dict:
    """Rebuild entry keeping only the specified event indices."""
    metadata = entry.get("metadata", {})
    event_ids = metadata.get("event_ids", [])
    event_types = metadata.get("event_types", [])
    event_metadatas = metadata.get("event_metadata", [])
    actors = metadata.get("actors", [])
    actor_ids = metadata.get("actor_ids", [])

    content = entry.get("content", "")
    parts = split_content_parts(content, event_types)

    # If we can't split properly, return None to skip rebuilding
    if len(parts) != len(event_ids):
        return None

    # Keep only the parts at keep_indices
    new_parts = [parts[i] for i in keep_indices if i < len(parts)]
    new_event_ids = [event_ids[i] for i in keep_indices if i < len(event_ids)]
    new_event_types = [event_types[i] for i in keep_indices if i < len(event_types)] if event_types else []

    # Handle event_metadata (could be single dict or list)
    if isinstance(event_metadatas, list):
        new_event_metadatas = [event_metadatas[i] for i in keep_indices if i < len(event_metadatas)]
    else:
        new_event_metadatas = event_metadatas  # Keep as-is if single dict

    # Rebuild content
    new_content = "\n\n".join(new_parts)

    # Create new entry
    new_entry = entry.copy()
    new_entry["content"] = new_content
    new_entry["metadata"] = metadata.copy()
    new_entry["metadata"]["event_ids"] = new_event_ids
    new_entry["metadata"]["event_types"] = new_event_types
    new_entry["metadata"]["event_count"] = len(new_event_ids)
    if isinstance(event_metadatas, list):
        new_entry["metadata"]["event_metadata"] = new_event_metadatas if len(new_event_metadatas) > 1 else (new_event_metadatas[0] if new_event_metadatas else {})

    # Recalculate tokens (rough estimate)
    new_entry["tokens"] = len(new_content) // 4

    return new_entry


async def dedupe_conversation(
    redis: Redis,
    agent_id: str,
    apply: bool = False,
    verbose: bool = False,
) -> None:
    """Remove duplicate event entries from conversation."""
    key = get_conversation_key(agent_id)

    raw_entries = await redis.lrange(key, 0, -1)
    if not raw_entries:
        print(f"No conversation found for agent '{agent_id}'")
        return

    print(f"Scanning {len(raw_entries)} entries for duplicates...")
    print()

    seen_event_ids: set[str] = set()
    entries_to_keep: list[bytes] = []
    entries_to_remove: list[tuple[int, dict]] = []
    entries_to_rebuild: list[tuple[int, dict, dict]] = []  # (index, old_entry, new_entry)
    rebuild_failures: list[tuple[int, dict, int, int]] = []  # (index, entry, dup_count, total)

    for i, raw in enumerate(raw_entries):
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            entries_to_keep.append(raw)
            continue

        metadata = entry.get("metadata", {})
        event_ids = metadata.get("event_ids", [])

        # Handle single event_id field (for code events)
        if not event_ids and "event_id" in metadata:
            event_ids = [metadata["event_id"]]

        # No event_ids = not an event-based entry (e.g., assistant response)
        if not event_ids:
            entries_to_keep.append(raw)
            continue

        # Check which event_ids are duplicates (by index)
        duplicate_indices = [idx for idx, eid in enumerate(event_ids) if eid in seen_event_ids]
        new_indices = [idx for idx, eid in enumerate(event_ids) if eid not in seen_event_ids]

        if len(duplicate_indices) == len(event_ids):
            # ALL event_ids are duplicates - remove this entry
            entries_to_remove.append((i, entry))
            if verbose:
                role = entry.get("role", "?")
                content = truncate(entry.get("content", ""), 50)
                print(f"  [{i}] REMOVE: {role} ({len(event_ids)} events) - {content}")
        elif duplicate_indices:
            # SOME event_ids are duplicates - try to rebuild
            new_entry = rebuild_entry(entry, new_indices)
            if new_entry:
                entries_to_rebuild.append((i, entry, new_entry))
                entries_to_keep.append(json.dumps(new_entry).encode())
                if verbose:
                    print(f"  [{i}] REBUILD: keeping {len(new_indices)}/{len(event_ids)} events")
            else:
                # Couldn't rebuild - keep original and flag
                rebuild_failures.append((i, entry, len(duplicate_indices), len(event_ids)))
                entries_to_keep.append(raw)
                if verbose:
                    print(f"  [{i}] SKIP: couldn't split content ({len(duplicate_indices)}/{len(event_ids)} dups)")
            # Mark new ones as seen
            for idx in new_indices:
                if idx < len(event_ids):
                    seen_event_ids.add(event_ids[idx])
        else:
            # No duplicates - keep and mark as seen
            entries_to_keep.append(raw)
            seen_event_ids.update(event_ids)

    print(f"Results:")
    print(f"  Total entries:      {len(raw_entries)}")
    print(f"  Unchanged:          {len(raw_entries) - len(entries_to_remove) - len(entries_to_rebuild) - len(rebuild_failures)}")
    print(f"  Full duplicates:    {len(entries_to_remove)} (will remove)")
    print(f"  Partial duplicates: {len(entries_to_rebuild)} (will rebuild)")
    print(f"  Rebuild failures:   {len(rebuild_failures)} (kept as-is)")
    print()

    if entries_to_remove:
        print("Entries to remove (all events are duplicates):")
        for i, entry in entries_to_remove[:15]:
            event_ids = entry.get("metadata", {}).get("event_ids", [])
            actor = entry.get("metadata", {}).get("actor", "?")
            event_type = entry.get("metadata", {}).get("event_type", "?")
            content = truncate(entry.get("content", ""), 50)
            print(f"  [{i}] actor={actor} type={event_type} events={len(event_ids)}")
            print(f"       {content}")
        if len(entries_to_remove) > 15:
            print(f"  ... and {len(entries_to_remove) - 15} more")
        print()

    if entries_to_rebuild:
        print("Entries to rebuild (removing duplicate events):")
        for i, old_entry, new_entry in entries_to_rebuild[:15]:
            old_count = len(old_entry.get("metadata", {}).get("event_ids", []))
            new_count = len(new_entry.get("metadata", {}).get("event_ids", []))
            actor = old_entry.get("metadata", {}).get("actor", "?")
            print(f"  [{i}] actor={actor}: {old_count} -> {new_count} events")
        if len(entries_to_rebuild) > 15:
            print(f"  ... and {len(entries_to_rebuild) - 15} more")
        print()

    if rebuild_failures:
        print("Rebuild failures (content couldn't be split, kept as-is):")
        for i, entry, dup_count, total in rebuild_failures[:10]:
            actor = entry.get("metadata", {}).get("actor", "?")
            event_types = entry.get("metadata", {}).get("event_types", [])
            print(f"  [{i}] actor={actor} types={event_types[:3]} ({dup_count}/{total} dups)")
        if len(rebuild_failures) > 10:
            print(f"  ... and {len(rebuild_failures) - 10} more")
        print()

    if not entries_to_remove and not entries_to_rebuild:
        print("No changes needed!")
        return

    if not apply:
        print("Dry run - no changes made. Use --apply to apply changes.")
        return

    # Apply changes
    total_changes = len(entries_to_remove) + len(entries_to_rebuild)
    print(f"Applying {total_changes} changes...")

    pipe = redis.pipeline()
    pipe.delete(key)
    if entries_to_keep:
        pipe.rpush(key, *entries_to_keep)
    await pipe.execute()

    print(f"Done! Removed {len(entries_to_remove)}, rebuilt {len(entries_to_rebuild)}. Total entries: {len(entries_to_keep)}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove duplicate events from conversation history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("agent_id", help="Agent ID (e.g., 'corroded', 'andi')")
    parser.add_argument(
        "--host", default="localhost", help="Redis host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=6379, help="Redis port (default: 6379)"
    )
    parser.add_argument(
        "--apply", action="store_true", help="Actually remove duplicates (default: dry run)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    args = parser.parse_args()

    try:
        redis = Redis(host=args.host, port=args.port, decode_responses=False)
        await redis.ping()
    except Exception as e:
        print(f"Cannot connect to Redis at {args.host}:{args.port}: {e}")
        sys.exit(1)

    try:
        await dedupe_conversation(
            redis,
            args.agent_id,
            apply=args.apply,
            verbose=args.verbose,
        )
    finally:
        await redis.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
