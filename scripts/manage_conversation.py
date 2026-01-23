#!/usr/bin/env python3
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Manage agent conversation history in Redis.

Usage:
    # List all entries for an agent
    python scripts/manage_conversation.py corroded list

    # Show details of a specific entry
    python scripts/manage_conversation.py corroded show 0

    # Delete a specific entry by index
    python scripts/manage_conversation.py corroded delete 5

    # Delete a range of entries (inclusive)
    python scripts/manage_conversation.py corroded delete-range 0 10

    # Delete all entries
    python scripts/manage_conversation.py corroded clear

    # Delete entries matching criteria
    python scripts/manage_conversation.py corroded prune --saved-only
    python scripts/manage_conversation.py corroded prune --role user
    python scripts/manage_conversation.py corroded prune --older-than 3600
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "aim-mud" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "aim-core" / "src"))

try:
    from redis.asyncio import Redis
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure Redis is installed: pip install redis")
    sys.exit(1)


def get_conversation_key(agent_id: str) -> str:
    """Get the Redis key for agent conversation."""
    return f"mud:agent:{agent_id}:conversation"


def format_timestamp(ts: int | float | None) -> str:
    """Format a Unix timestamp as readable datetime."""
    if ts is None:
        return "N/A"
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, OSError):
        return f"Invalid({ts})"


def truncate(text: str, max_len: int = 60) -> str:
    """Truncate text with ellipsis."""
    if not text:
        return ""
    # Replace newlines with spaces for display
    text = text.replace("\n", " ").replace("\r", "")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


async def list_entries(redis: Redis, agent_id: str, verbose: bool = False) -> None:
    """List all conversation entries."""
    key = get_conversation_key(agent_id)

    # Check if key exists
    exists = await redis.exists(key)
    if not exists:
        print(f"No conversation found for agent '{agent_id}'")
        print(f"Key: {key}")
        return

    # Get all entries
    raw_entries = await redis.lrange(key, 0, -1)

    if not raw_entries:
        print(f"Conversation for '{agent_id}' is empty")
        return

    print(f"=== Conversation History: {agent_id} ===")
    print(f"Key: {key}")
    print(f"Total entries: {len(raw_entries)}")
    print()

    total_tokens = 0
    saved_count = 0

    # Header
    if verbose:
        print(f"{'Idx':>4} {'Role':<10} {'Tokens':>6} {'Saved':>5} {'Timestamp':<24} Content")
        print("-" * 100)
    else:
        print(f"{'Idx':>4} {'Role':<10} {'Tokens':>6} {'Saved':>5} Content")
        print("-" * 80)

    for i, raw in enumerate(raw_entries):
        try:
            entry = json.loads(raw)
            role = entry.get("role", "?")
            content = entry.get("content", "")
            tokens = entry.get("tokens", 0)
            saved = entry.get("saved", False)
            timestamp = entry.get("timestamp")

            total_tokens += tokens
            if saved:
                saved_count += 1

            saved_str = "Yes" if saved else "No"

            if verbose:
                ts_str = format_timestamp(timestamp)
                print(f"{i:>4} {role:<10} {tokens:>6} {saved_str:>5} {ts_str:<24} {truncate(content, 40)}")
            else:
                print(f"{i:>4} {role:<10} {tokens:>6} {saved_str:>5} {truncate(content)}")

        except json.JSONDecodeError:
            print(f"{i:>4} [INVALID JSON]")

    print()
    print(f"Summary: {len(raw_entries)} entries, {total_tokens} tokens, {saved_count} saved")


async def show_entry(redis: Redis, agent_id: str, index: int) -> None:
    """Show detailed view of a specific entry."""
    key = get_conversation_key(agent_id)

    raw = await redis.lindex(key, index)
    if raw is None:
        length = await redis.llen(key)
        print(f"Entry {index} not found (list has {length} entries)")
        return

    try:
        entry = json.loads(raw)
    except json.JSONDecodeError:
        print(f"Entry {index} contains invalid JSON:")
        print(raw.decode() if isinstance(raw, bytes) else raw)
        return

    print(f"=== Entry {index} for {agent_id} ===")
    print()

    # Core fields
    print(f"Role:           {entry.get('role', 'N/A')}")
    print(f"Tokens:         {entry.get('tokens', 'N/A')}")
    print(f"Saved:          {entry.get('saved', False)}")
    print(f"Skip Save:      {entry.get('skip_save', False)}")
    print(f"Timestamp:      {format_timestamp(entry.get('timestamp'))}")
    print(f"Doc ID:         {entry.get('doc_id', 'N/A')}")
    print(f"Document Type:  {entry.get('document_type', 'N/A')}")
    print(f"Conversation:   {entry.get('conversation_id', 'N/A')}")
    print(f"Sequence No:    {entry.get('sequence_no', 'N/A')}")
    print(f"Speaker ID:     {entry.get('speaker_id', 'N/A')}")

    # Metadata
    metadata = entry.get("metadata", {})
    if metadata:
        print()
        print("Metadata:")
        for k, v in metadata.items():
            print(f"  {k}: {v}")

    # Think content
    think = entry.get("think")
    if think:
        print()
        print("Think:")
        print("-" * 40)
        print(think)
        print("-" * 40)

    # Main content
    print()
    print("Content:")
    print("-" * 40)
    print(entry.get("content", ""))
    print("-" * 40)


async def delete_entry(redis: Redis, agent_id: str, index: int, confirm: bool = True) -> None:
    """Delete a specific entry by index.

    Redis lists don't support direct index deletion, so we:
    1. Get all entries
    2. Remove the target entry
    3. Replace the entire list atomically
    """
    key = get_conversation_key(agent_id)

    # Get current length
    length = await redis.llen(key)
    if index < 0 or index >= length:
        print(f"Index {index} out of range (list has {length} entries)")
        return

    # Show what we're deleting
    raw = await redis.lindex(key, index)
    if raw:
        try:
            entry = json.loads(raw)
            print(f"Entry to delete (index {index}):")
            print(f"  Role: {entry.get('role')}")
            print(f"  Content: {truncate(entry.get('content', ''), 80)}")
            print(f"  Tokens: {entry.get('tokens')}")
            print(f"  Saved: {entry.get('saved')}")
        except json.JSONDecodeError:
            print(f"Entry to delete (index {index}): [invalid JSON]")

    if confirm:
        response = input("\nConfirm delete? [y/N]: ")
        if response.lower() != "y":
            print("Cancelled")
            return

    # Get all entries
    all_entries = await redis.lrange(key, 0, -1)

    # Remove the target
    new_entries = all_entries[:index] + all_entries[index + 1:]

    # Replace atomically using pipeline
    pipe = redis.pipeline()
    pipe.delete(key)
    if new_entries:
        pipe.rpush(key, *new_entries)
    await pipe.execute()

    print(f"Deleted entry {index}. Remaining entries: {len(new_entries)}")


async def delete_range(
    redis: Redis, agent_id: str, start: int, end: int, confirm: bool = True
) -> None:
    """Delete a range of entries (inclusive)."""
    key = get_conversation_key(agent_id)

    length = await redis.llen(key)
    if length == 0:
        print("Conversation is empty")
        return

    # Validate range
    if start < 0:
        start = 0
    if end >= length:
        end = length - 1
    if start > end:
        print(f"Invalid range: {start} to {end}")
        return

    count = end - start + 1
    print(f"Will delete {count} entries (index {start} to {end})")

    if confirm:
        response = input("Confirm delete? [y/N]: ")
        if response.lower() != "y":
            print("Cancelled")
            return

    # Get all entries and remove range
    all_entries = await redis.lrange(key, 0, -1)
    new_entries = all_entries[:start] + all_entries[end + 1:]

    # Replace atomically
    pipe = redis.pipeline()
    pipe.delete(key)
    if new_entries:
        pipe.rpush(key, *new_entries)
    await pipe.execute()

    print(f"Deleted {count} entries. Remaining: {len(new_entries)}")


async def clear_all(redis: Redis, agent_id: str, confirm: bool = True) -> None:
    """Delete all entries."""
    key = get_conversation_key(agent_id)

    length = await redis.llen(key)
    if length == 0:
        print("Conversation is already empty")
        return

    print(f"Will delete ALL {length} entries for agent '{agent_id}'")

    if confirm:
        response = input("Type 'DELETE' to confirm: ")
        if response != "DELETE":
            print("Cancelled")
            return

    await redis.delete(key)
    print(f"Deleted all {length} entries")


async def prune_entries(
    redis: Redis,
    agent_id: str,
    saved_only: bool = False,
    role: str | None = None,
    older_than: int | None = None,
    confirm: bool = True,
) -> None:
    """Delete entries matching criteria."""
    key = get_conversation_key(agent_id)

    all_entries = await redis.lrange(key, 0, -1)
    if not all_entries:
        print("Conversation is empty")
        return

    now = datetime.now(timezone.utc).timestamp()
    to_keep = []
    to_delete = []

    for i, raw in enumerate(all_entries):
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            to_keep.append(raw)  # Keep invalid entries
            continue

        should_delete = True

        # Check saved filter
        if saved_only and not entry.get("saved", False):
            should_delete = False

        # Check role filter
        if role and entry.get("role") != role:
            should_delete = False

        # Check age filter
        if older_than is not None:
            ts = entry.get("timestamp", now)
            age = now - ts
            if age < older_than:
                should_delete = False

        if should_delete:
            to_delete.append((i, entry))
        else:
            to_keep.append(raw)

    if not to_delete:
        print("No entries match the criteria")
        return

    print(f"Will delete {len(to_delete)} entries:")
    for i, entry in to_delete[:10]:  # Show first 10
        print(f"  [{i}] {entry.get('role')}: {truncate(entry.get('content', ''), 50)}")
    if len(to_delete) > 10:
        print(f"  ... and {len(to_delete) - 10} more")

    if confirm:
        response = input(f"\nDelete {len(to_delete)} entries? [y/N]: ")
        if response.lower() != "y":
            print("Cancelled")
            return

    # Replace with filtered list
    pipe = redis.pipeline()
    pipe.delete(key)
    if to_keep:
        pipe.rpush(key, *to_keep)
    await pipe.execute()

    print(f"Deleted {len(to_delete)} entries. Remaining: {len(to_keep)}")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Manage agent conversation history in Redis",
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
        "-y", "--yes", action="store_true", help="Skip confirmation prompts"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # List command
    list_parser = subparsers.add_parser("list", help="List all entries")
    list_parser.add_argument("-v", "--verbose", action="store_true", help="Show timestamps")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show details of an entry")
    show_parser.add_argument("index", type=int, help="Entry index")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete entry by index")
    delete_parser.add_argument("index", type=int, help="Entry index to delete")

    # Delete range command
    range_parser = subparsers.add_parser("delete-range", help="Delete range of entries")
    range_parser.add_argument("start", type=int, help="Start index (inclusive)")
    range_parser.add_argument("end", type=int, help="End index (inclusive)")

    # Clear command
    subparsers.add_parser("clear", help="Delete ALL entries")

    # Prune command
    prune_parser = subparsers.add_parser("prune", help="Delete entries matching criteria")
    prune_parser.add_argument(
        "--saved-only", action="store_true", help="Only delete entries marked as saved"
    )
    prune_parser.add_argument("--role", choices=["user", "assistant"], help="Only delete entries with this role")
    prune_parser.add_argument(
        "--older-than",
        type=int,
        metavar="SECONDS",
        help="Only delete entries older than N seconds",
    )

    args = parser.parse_args()

    # Connect to Redis
    try:
        redis = Redis(host=args.host, port=args.port, decode_responses=False)
        await redis.ping()
    except Exception as e:
        print(f"Cannot connect to Redis at {args.host}:{args.port}: {e}")
        sys.exit(1)

    try:
        confirm = not args.yes

        if args.command == "list":
            await list_entries(redis, args.agent_id, verbose=args.verbose)
        elif args.command == "show":
            await show_entry(redis, args.agent_id, args.index)
        elif args.command == "delete":
            await delete_entry(redis, args.agent_id, args.index, confirm=confirm)
        elif args.command == "delete-range":
            await delete_range(redis, args.agent_id, args.start, args.end, confirm=confirm)
        elif args.command == "clear":
            await clear_all(redis, args.agent_id, confirm=confirm)
        elif args.command == "prune":
            await prune_entries(
                redis,
                args.agent_id,
                saved_only=args.saved_only,
                role=args.role,
                older_than=args.older_than,
                confirm=confirm,
            )
    finally:
        await redis.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
