# aim-mud-types/client/__init__.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Centralized Redis client for MUD types with Pydantic validation.

This module provides a type-safe Redis client for all ANDIMUD coordination
and state structures. All Redis hash operations go through Pydantic models
to ensure validation and prevent corruption.

Architecture:
    BaseRedisMUDClient: Core serialization and CRUD operations
    Type-specific mixins: Domain-specific methods for each MUD type
    RedisMUDClient: Composed client with all functionality

Usage:
    client = RedisMUDClient(redis_client)
    turn_request = await client.get_turn_request("andi")
    await client.update_turn_request("andi", updated_request, expected_turn_id)
"""

from typing import Any, Callable, Optional, Type, TypeVar
import logging
import json
from datetime import datetime
from enum import Enum

from pydantic import BaseModel

logger = logging.getLogger(__name__)
import redis.asyncio as redis

from ..redis_keys import RedisKeys
from ..coordination import MUDTurnRequest, TurnRequestStatus, TurnReason
from ..profile import AgentProfile, RoomProfile
from ..coordination import DreamerState

T = TypeVar("T", bound=BaseModel)


class BaseRedisMUDClient:
    """Base Redis client with generic hash operations.

    Provides serialization, deserialization, and CAS (Compare-And-Swap)
    operations for Pydantic models stored as Redis hashes.
    """

    def __init__(self, redis_client: redis.Redis):
        """Initialize with Redis client.

        Args:
            redis_client: Async Redis client instance
        """
        self.redis = redis_client

    def _serialize_value(self, value: Any) -> Optional[str]:
        """Convert Python value to Redis string.

        Serialization rules:
        - None → None (field will be skipped)
        - datetime → ISO format string
        - Enum → .value attribute
        - dict → JSON string
        - bool → "true" or "false" (lowercase)
        - int/float → string representation
        - str → unchanged

        Args:
            value: Python value to serialize

        Returns:
            Serialized string value, or None to skip field
        """
        if value is None:
            return None
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, dict):
            return json.dumps(value)
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        else:
            return str(value)

    def _serialize_object(self, obj: BaseModel) -> dict[str, str]:
        """Convert Pydantic model to Redis field dict.

        Uses model_dump() to extract all fields, then serializes each
        value using _serialize_value(). Skips None values.

        Args:
            obj: Pydantic model instance

        Returns:
            Dictionary of field_name -> serialized_value (no None values)
        """
        fields = {}
        for field_name, field_value in obj.model_dump().items():
            serialized = self._serialize_value(field_value)
            if serialized is not None:
                fields[field_name] = serialized
        return fields

    async def _get_hash(self, model_class: Type[T], key: str) -> Optional[T]:
        """Fetch and deserialize a Redis hash to Pydantic model.

        Args:
            model_class: Pydantic model class to deserialize into
            key: Redis key for the hash

        Returns:
            Deserialized model instance, or None if key doesn't exist
        """
        data = await self.redis.hgetall(key)

        if not data:
            return None

        # Decode bytes to strings
        decoded: dict[str, str] = {}
        for k, v in data.items():
            if isinstance(k, bytes):
                k = k.decode("utf-8")
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            decoded[str(k)] = str(v)

        # Deserialize to model
        try:
            return model_class.model_validate(decoded)
        except Exception as e:
            logger.warning(
                f"Failed to validate {model_class.__name__} from Redis key '{key}': {e}",
                exc_info=True
            )
            return None

    async def _create_hash(
        self,
        key: str,
        obj: BaseModel,
        exists_ok: bool = False
    ) -> bool:
        """Create a new Redis hash from Pydantic model.

        Uses Lua script for atomic EXISTS check followed by HSET.
        Prevents TOCTOU race conditions.

        Args:
            key: Redis key for the hash
            obj: Pydantic model to serialize
            exists_ok: If False, fail if key exists. If True, overwrite.

        Returns:
            True if created, False if key exists and exists_ok=False
        """
        # Lua script for atomic create
        if exists_ok:
            lua_script = """
                local key = KEYS[1]

                -- Create/overwrite with all fields
                for i = 1, #ARGV, 2 do
                    redis.call('HSET', key, ARGV[i], ARGV[i+1])
                end

                return 1
            """
        else:
            lua_script = """
                local key = KEYS[1]

                -- Fail if key already exists
                if redis.call('EXISTS', key) == 1 then
                    return 0
                end

                -- Create with all fields
                for i = 1, #ARGV, 2 do
                    redis.call('HSET', key, ARGV[i], ARGV[i+1])
                end

                return 1
            """

        # Serialize object to field list
        fields = []
        for field_name, field_value in self._serialize_object(obj).items():
            fields.extend([field_name, field_value])

        # Execute Lua script
        result = await self.redis.eval(lua_script, 1, key, *fields)
        return result == 1

    async def _update_hash(
        self,
        key: str,
        obj: BaseModel,
        cas_field: Optional[str] = None,
        cas_value: Optional[str] = None
    ) -> bool:
        """Update Redis hash with optional CAS check.

        Uses Lua script for atomic CAS check followed by HSET.

        Args:
            key: Redis key for the hash
            obj: Pydantic model with updated fields
            cas_field: Field name for CAS check (e.g., "turn_id")
            cas_value: Expected value for CAS field

        Returns:
            True if updated, False if CAS check failed
        """
        if cas_field and cas_value:
            # Lua script with CAS
            lua_script = """
                local key = KEYS[1]
                local cas_field = ARGV[1]
                local cas_value = ARGV[2]

                -- CAS check
                local current = redis.call('HGET', key, cas_field)
                if current ~= cas_value then
                    return 0
                end

                -- Update fields (start at ARGV[3])
                for i = 3, #ARGV, 2 do
                    redis.call('HSET', key, ARGV[i], ARGV[i+1])
                end

                return 1
            """

            # Serialize object to field list
            fields = []
            for field_name, field_value in self._serialize_object(obj).items():
                fields.extend([field_name, field_value])

            # Execute with CAS parameters
            result = await self.redis.eval(
                lua_script,
                1,
                key,
                cas_field,
                cas_value,
                *fields
            )
        else:
            # Simple update without CAS
            lua_script = """
                local key = KEYS[1]

                -- Update all fields
                for i = 1, #ARGV, 2 do
                    redis.call('HSET', key, ARGV[i], ARGV[i+1])
                end

                return 1
            """

            # Serialize object to field list
            fields = []
            for field_name, field_value in self._serialize_object(obj).items():
                fields.extend([field_name, field_value])

            result = await self.redis.eval(lua_script, 1, key, *fields)

        return result == 1

    async def _update_fields(
        self,
        key: str,
        fields: dict[str, Any],
        cas_field: Optional[str] = None,
        cas_value: Optional[str] = None
    ) -> bool:
        """Partial field update with optional CAS check.

        Similar to _update_hash but takes raw field dict instead of model.
        Useful for single-field updates like heartbeats.

        Args:
            key: Redis key for the hash
            fields: Dictionary of field_name -> value to update
            cas_field: Field name for CAS check
            cas_value: Expected value for CAS field

        Returns:
            True if updated, False if CAS check failed
        """
        # Serialize field values
        serialized_fields = []
        for field_name, field_value in fields.items():
            serialized_value = self._serialize_value(field_value)
            if serialized_value is not None:
                serialized_fields.extend([field_name, serialized_value])

        if cas_field and cas_value:
            # Lua script with CAS
            lua_script = """
                local key = KEYS[1]
                local cas_field = ARGV[1]
                local cas_value = ARGV[2]

                -- CAS check
                local current = redis.call('HGET', key, cas_field)
                if current ~= cas_value then
                    return 0
                end

                -- Update fields (start at ARGV[3])
                for i = 3, #ARGV, 2 do
                    redis.call('HSET', key, ARGV[i], ARGV[i+1])
                end

                return 1
            """

            result = await self.redis.eval(
                lua_script,
                1,
                key,
                cas_field,
                cas_value,
                *serialized_fields
            )
        else:
            # Simple update without CAS
            lua_script = """
                local key = KEYS[1]

                -- Update all fields
                for i = 1, #ARGV, 2 do
                    redis.call('HSET', key, ARGV[i], ARGV[i+1])
                end

                return 1
            """

            result = await self.redis.eval(lua_script, 1, key, *serialized_fields)

        return result == 1

    async def _get_field(
        self,
        key: str,
        field: str,
        convert: Optional[Callable] = None
    ) -> Any:
        """Fetch single field value from hash.

        Efficient single-field read using HGET.

        Args:
            key: Redis key for the hash
            field: Field name to fetch
            convert: Optional function to convert string value

        Returns:
            Field value (as string or converted), or None if missing
        """
        value = await self.redis.hget(key, field)

        if value is None:
            return None

        if isinstance(value, bytes):
            value = value.decode("utf-8")

        if convert:
            return convert(value)

        return value


# Import mixins
from .mixins.turn_request import TurnRequestMixin
from .mixins.agent_profile import AgentProfileMixin
from .mixins.room_profile import RoomProfileMixin
from .mixins.dreamer_state import DreamerStateMixin


class RedisMUDClient(
    BaseRedisMUDClient,
    TurnRequestMixin,
    AgentProfileMixin,
    RoomProfileMixin,
    DreamerStateMixin,
):
    """Complete Redis MUD client with all type-specific operations.

    Combines base serialization/deserialization with domain-specific
    methods for turn requests, profiles, and dreamer state.

    Usage:
        client = RedisMUDClient(redis_client)

        # Turn requests
        turn_req = await client.get_turn_request("andi")
        await client.create_turn_request("andi", new_request)
        await client.update_turn_request("andi", updated, expected_turn_id)
        await client.heartbeat_turn_request("andi")

        # Agent profiles
        profile = await client.get_agent_profile("andi")
        await client.create_agent_profile(profile)
        await client.update_agent_profile_fields("andi", conversation_id="conv2")

        # Room profiles
        room = await client.get_room_profile("room123")

        # Dreamer state
        state = await client.get_dreamer_state("andi")
        await client.update_dreamer_state_fields("andi", enabled=True)
    """
    pass


__all__ = [
    "BaseRedisMUDClient",
    "RedisMUDClient",
]
