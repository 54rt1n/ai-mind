# aim-mud-types/client/mixins/room_profile.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""RoomProfile-specific operations (read-only)."""

from typing import Optional, TYPE_CHECKING

from ...redis_keys import RedisKeys
from ...models.profile import RoomProfile

if TYPE_CHECKING:
    from ..base import BaseAsyncRedisMUDClient


class RoomProfileMixin:
    """RoomProfile-specific Redis operations.

    Read-only operations for room profiles.
    Room profiles are written by Evennia, workers only read them.
    """

    async def get_room_profile(
        self: "BaseAsyncRedisMUDClient",
        room_id: str
    ) -> Optional[RoomProfile]:
        """Fetch room profile.

        Args:
            room_id: Room identifier

        Returns:
            RoomProfile object, or None if not found
        """
        key = RedisKeys.room_profile(room_id)
        return await self._get_hash(RoomProfile, key)

    async def get_room_profile_raw(
        self: "BaseAsyncRedisMUDClient",
        room_id: str
    ) -> dict[str, str]:
        """Fetch room profile hash and decode to string values."""
        key = RedisKeys.room_profile(room_id)
        data = await self.redis.hgetall(key)
        decoded: dict[str, str] = {}
        for k, v in (data or {}).items():
            if isinstance(k, bytes):
                k = k.decode("utf-8")
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            decoded[str(k)] = str(v)
        return decoded

    async def update_room_profile_fields(
        self: "BaseAsyncRedisMUDClient",
        room_id: str,
        **fields,
    ) -> bool:
        """Partial update of room profile fields.

        Args:
            room_id: Room identifier
            **fields: Field names and values to update

        Returns:
            True if updated

        Example:
            await client.update_room_profile_fields(
                "room123",
                description="A cozy corner"
            )
        """
        key = RedisKeys.room_profile(room_id)
        return await self._update_fields(key, fields)

    async def set_room_profile_fields(
        self: "BaseAsyncRedisMUDClient",
        room_id: str,
        fields: dict[str, str],
    ) -> bool:
        """Set room profile fields from a mapping.

        Args:
            room_id: Room identifier
            fields: Dictionary of field names and string values

        Returns:
            True if fields were set, False if fields dict was empty
        """
        if not fields:
            return False
        key = RedisKeys.room_profile(room_id)
        await self.redis.hset(key, mapping=fields)
        return True
