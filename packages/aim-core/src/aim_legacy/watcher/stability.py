# aim_legacy/watcher/stability.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Redis-backed stability tracking for conversation message counts.

Tracks when message counts stabilize to avoid triggering pipelines
while conversations are still being saved.

** LEGACY MODULE ** - Preserved from pre-migration codebase
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)


@dataclass
class ConversationSnapshot:
    """Tracks message and token count stability for a conversation."""

    conversation_id: str
    message_count: int
    token_count: int = 0  # Total tokens in conversation
    first_seen_at: float = 0.0  # Unix timestamp
    last_changed_at: float = 0.0  # Unix timestamp

    def is_stable(self, stability_seconds: int) -> bool:
        """Check if conversation has been stable for required duration."""
        elapsed = datetime.now().timestamp() - self.last_changed_at
        return elapsed >= stability_seconds

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str | bytes) -> ConversationSnapshot:
        """Deserialize from JSON string."""
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        parsed = json.loads(data)
        # Handle backwards compatibility - older snapshots may not have token_count
        if 'token_count' not in parsed:
            parsed['token_count'] = 0
        return cls(**parsed)


class StabilityTracker:
    """
    Redis-backed tracker for conversation message count stability.

    Tracks message counts over time to determine when a conversation
    has finished being written to disk. A conversation is considered
    stable when its message count hasn't changed for a configurable
    duration.

    Uses Redis with TTL for auto-cleanup of stale entries.
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        key_prefix: str = "watcher",
        stability_seconds: int = 120,
        ttl_seconds: int = 86400,  # 24 hours
    ):
        """
        Args:
            redis_client: Async Redis client
            key_prefix: Prefix for Redis keys
            stability_seconds: How long count must be stable before triggering
            ttl_seconds: TTL for Redis entries (auto-cleanup)
        """
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.stability_seconds = stability_seconds
        self.ttl_seconds = ttl_seconds

    def _snapshot_key(self, conversation_id: str) -> str:
        """Generate Redis key for conversation snapshot."""
        return f"{self.key_prefix}:stability:{conversation_id}"

    async def update_and_check(
        self,
        conversation_id: str,
        current_message_count: int,
        current_token_count: int = 0,
    ) -> tuple[bool, ConversationSnapshot]:
        """
        Update message/token count and check stability.

        Args:
            conversation_id: The conversation to track
            current_message_count: Current message count
            current_token_count: Current token count (optional)

        Returns:
            (is_stable, snapshot) - True if stable for required duration
        """
        key = self._snapshot_key(conversation_id)
        now = datetime.now().timestamp()

        # Load existing snapshot
        data = await self.redis.get(key)

        if data is None:
            # First time seeing this conversation
            snapshot = ConversationSnapshot(
                conversation_id=conversation_id,
                message_count=current_message_count,
                token_count=current_token_count,
                first_seen_at=now,
                last_changed_at=now,
            )
            await self._save_snapshot(key, snapshot)
            logger.debug(
                f"First observation of {conversation_id}: {current_message_count} messages, {current_token_count} tokens"
            )
            return False, snapshot

        snapshot = ConversationSnapshot.from_json(data)

        # Check if either count changed
        msg_changed = current_message_count != snapshot.message_count
        token_changed = current_token_count != snapshot.token_count and current_token_count > 0

        if msg_changed or token_changed:
            # Count changed - reset stability timer
            logger.debug(
                f"Counts changed for {conversation_id}: "
                f"msgs {snapshot.message_count} -> {current_message_count}, "
                f"tokens {snapshot.token_count} -> {current_token_count}"
            )
            snapshot.message_count = current_message_count
            snapshot.token_count = current_token_count
            snapshot.last_changed_at = now
            await self._save_snapshot(key, snapshot)
            return False, snapshot

        # Counts unchanged - check if stable long enough
        is_stable = snapshot.is_stable(self.stability_seconds)

        if is_stable:
            logger.info(
                f"Conversation {conversation_id} is stable "
                f"({current_message_count} messages, {current_token_count} tokens, stable for {self.stability_seconds}s)"
            )

        return is_stable, snapshot

    async def mark_processed(self, conversation_id: str) -> None:
        """Remove stability tracking after successful pipeline trigger."""
        key = self._snapshot_key(conversation_id)
        await self.redis.delete(key)
        logger.debug(f"Cleared stability tracking for {conversation_id}")

    async def _save_snapshot(self, key: str, snapshot: ConversationSnapshot) -> None:
        """Save snapshot with TTL for auto-cleanup."""
        await self.redis.set(key, snapshot.to_json(), ex=self.ttl_seconds)
