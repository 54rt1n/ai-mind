# aim-mud-types/client/mixins/agent_events.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Async helpers for agent:{id}:events streams."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from ...redis_keys import RedisKeys

if TYPE_CHECKING:
    from ..base import BaseAsyncRedisMUDClient


class AgentEventsStreamMixin:
    """Redis helpers for agent events streams."""

    def _decode_stream_id(self, msg_id: str | bytes) -> str:
        if isinstance(msg_id, bytes):
            return msg_id.decode("utf-8")
        return str(msg_id)

    def _agent_events_stream_key(
        self,
        agent_id: str,
        stream_key: Optional[str] = None,
    ) -> str:
        return stream_key or RedisKeys.agent_events(agent_id)

    async def get_agent_events_last_id(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str,
        *,
        stream_key: Optional[str] = None,
    ) -> Optional[str]:
        """Return last-generated id for agent events stream, or None if empty."""
        key = self._agent_events_stream_key(agent_id, stream_key)
        info = await self.redis.xinfo_stream(key)
        last_id = info.get("last-generated-id") or info.get(b"last-generated-id")
        if isinstance(last_id, bytes):
            last_id = last_id.decode("utf-8")
        if not last_id or last_id in ("0", "0-0"):
            return None
        return str(last_id)

    async def range_agent_events(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str,
        min_id: str,
        max_id: str,
        *,
        count: int = 100,
        stream_key: Optional[str] = None,
    ) -> list[tuple[str, dict]]:
        """Read a range of entries from agent events stream using XRANGE."""
        key = self._agent_events_stream_key(agent_id, stream_key)
        result = await self.redis.xrange(
            key,
            min=min_id,
            max=max_id,
            count=count,
        )
        return [(self._decode_stream_id(msg_id), data) for msg_id, data in result]

    async def range_agent_events_reverse(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str,
        *,
        count: int = 100,
        max_id: str = "+",
        min_id: str = "-",
        stream_key: Optional[str] = None,
    ) -> list[tuple[str, dict]]:
        """Read a reverse range of entries from agent events stream."""
        key = self._agent_events_stream_key(agent_id, stream_key)
        result = await self.redis.xrevrange(
            key,
            max=max_id,
            min=min_id,
            count=count,
        )
        return [(self._decode_stream_id(msg_id), data) for msg_id, data in result]

    async def append_agent_event(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str,
        payload: dict,
        *,
        maxlen: Optional[int] = None,
        approximate: bool = True,
        stream_key: Optional[str] = None,
    ) -> str:
        """Append an entry to agent events stream."""
        key = self._agent_events_stream_key(agent_id, stream_key)
        if maxlen is None:
            msg_id = await self.redis.xadd(key, payload)
        else:
            msg_id = await self.redis.xadd(
                key,
                payload,
                maxlen=maxlen,
                approximate=approximate,
            )
        return self._decode_stream_id(msg_id)

    async def get_agent_events_length(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str,
        *,
        stream_key: Optional[str] = None,
    ) -> int:
        """Return agent events stream length."""
        key = self._agent_events_stream_key(agent_id, stream_key)
        return await self.redis.xlen(key)

    async def delete_agent_events_stream(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str,
        *,
        stream_key: Optional[str] = None,
    ) -> bool:
        """Delete the agent events stream key entirely.

        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        key = self._agent_events_stream_key(agent_id, stream_key)
        result = await self.redis.delete(key)
        return result > 0

    async def delete_agent_events_if_drained(
        self: "BaseAsyncRedisMUDClient",
        agent_id: str,
        *,
        conversation_key: Optional[str] = None,
        stream_key: Optional[str] = None,
    ) -> bool:
        """Atomically delete agent events stream if fully drained.

        Uses a Lua script to atomically:
        1. Read the last conversation entry from the Redis list
        2. Parse JSON to extract last_event_id field
        3. Get stream's last-generated-id via XINFO STREAM
        4. Compare the two IDs
        5. Delete stream if they match

        This prevents race conditions where new events arrive between
        reading the conversation and checking the stream state.

        Args:
            agent_id: The agent ID.
            conversation_key: Optional override for conversation list key.
            stream_key: Optional override for stream key.

        Returns:
            True if the stream was deleted (was fully drained).
            False if stream has newer events, conversation is empty,
            last entry has no last_event_id, or stream doesn't exist.
        """
        conv_key = conversation_key or RedisKeys.agent_conversation(agent_id)
        event_key = self._agent_events_stream_key(agent_id, stream_key)

        # Lua script: atomically read conversation and stream, delete if IDs match
        lua_script = """
        -- KEYS[1]: conversation list key (mud:agent:{id}:conversation)
        -- KEYS[2]: stream key (agent:{id}:events)

        -- 1. Check if conversation list exists and has entries
        local list_len = redis.call('LLEN', KEYS[1])
        if list_len == 0 then
            return 0  -- No conversation entries, nothing to compare
        end

        -- 2. Get the LAST entry from the conversation list
        local last_entry_json = redis.call('LINDEX', KEYS[1], -1)
        if not last_entry_json then
            return 0  -- Should not happen if LLEN > 0, but guard anyway
        end

        -- 3. Parse JSON to extract last_event_id
        local cjson = require("cjson")
        local ok, entry = pcall(cjson.decode, last_entry_json)
        if not ok or not entry then
            return 0  -- JSON parse failed
        end

        local expected_id = entry.last_event_id
        if not expected_id or expected_id == "" then
            return 0  -- No last_event_id in conversation entry
        end

        -- 4. Check if stream exists
        local exists = redis.call('EXISTS', KEYS[2])
        if exists == 0 then
            return 0  -- Stream doesn't exist, nothing to delete
        end

        -- 5. Get stream's last-generated-id via XINFO STREAM
        local info = redis.call('XINFO', 'STREAM', KEYS[2])

        -- Parse the info array to find last-generated-id
        -- XINFO STREAM returns flat array: [field1, value1, field2, value2, ...]
        local last_id = nil
        for i = 1, #info, 2 do
            if info[i] == 'last-generated-id' then
                last_id = info[i + 1]
                break
            end
        end

        if not last_id or last_id == "" then
            return 0  -- Stream has no last-generated-id
        end

        -- 6. Compare IDs (both are strings like "1704096000000-42")
        if last_id == expected_id then
            redis.call('DEL', KEYS[2])
            return 1  -- Deleted
        end

        return 0  -- IDs don't match, stream has newer events
        """

        result = await self.redis.eval(lua_script, 2, conv_key, event_key)
        return result == 1
