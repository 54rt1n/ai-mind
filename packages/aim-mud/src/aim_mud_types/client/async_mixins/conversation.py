# aim-mud-types/client/mixins/conversation.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Async helpers for agent conversation lists."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from ...redis_keys import RedisKeys

if TYPE_CHECKING:
    from ..base import BaseAsyncRedisMUDClient


class ConversationMixin:
    """Redis helpers for agent conversation lists."""

    def _conversation_key(self, agent_id: str) -> str:
        return RedisKeys.agent_conversation(agent_id)

    def _decode_list_entry(self, raw) -> str:
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)

    async def get_conversation_entries(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str,
        start: int = 0,
        end: int = -1,
    ) -> list[str]:
        """Fetch conversation entries as JSON strings."""
        key = self._conversation_key(agent_id)
        raw_entries = await self.redis.lrange(key, start, end) or []
        return [self._decode_list_entry(raw) for raw in raw_entries]

    async def get_conversation_entry(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str,
        index: int,
    ) -> Optional[str]:
        """Fetch a single conversation entry by index."""
        key = self._conversation_key(agent_id)
        raw = await self.redis.lindex(key, index)
        if raw is None:
            return None
        return self._decode_list_entry(raw)

    async def append_conversation_entry(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str,
        entry_json: str,
    ) -> int:
        """Append a conversation entry JSON to the list."""
        key = self._conversation_key(agent_id)
        return await self.redis.rpush(key, entry_json)

    async def set_conversation_entry(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str,
        index: int,
        entry_json: str,
    ) -> bool:
        """Set a conversation entry at index."""
        key = self._conversation_key(agent_id)
        await self.redis.lset(key, index, entry_json)
        return True

    async def pop_conversation_entry(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str,
    ) -> Optional[str]:
        """Pop the oldest conversation entry (left)."""
        key = self._conversation_key(agent_id)
        raw = await self.redis.lpop(key)
        if raw is None:
            return None
        return self._decode_list_entry(raw)

    async def pop_last_conversation_entry(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str,
    ) -> Optional[str]:
        """Pop the newest conversation entry (right)."""
        key = self._conversation_key(agent_id)
        raw = await self.redis.rpop(key)
        if raw is None:
            return None
        return self._decode_list_entry(raw)

    async def get_conversation_length(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str,
    ) -> int:
        """Return conversation list length."""
        key = self._conversation_key(agent_id)
        return await self.redis.llen(key)

    async def delete_conversation(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str,
    ) -> int:
        """Delete conversation list key."""
        key = self._conversation_key(agent_id)
        return await self.redis.delete(key)

    async def replace_conversation_entries(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str,
        entries_json: list[str],
    ) -> None:
        """Replace entire conversation list."""
        key = self._conversation_key(agent_id)
        pipe = self.redis.pipeline()
        pipe.delete(key)
        for entry_json in entries_json:
            pipe.rpush(key, entry_json)
        await pipe.execute()
