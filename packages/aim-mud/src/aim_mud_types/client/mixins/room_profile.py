# aim-mud-types/client/mixins/room_profile.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""RoomProfile-specific operations (read-only)."""

from typing import Optional, TYPE_CHECKING

from ...redis_keys import RedisKeys
from ...profile import RoomProfile

if TYPE_CHECKING:
    from .. import BaseRedisMUDClient


class RoomProfileMixin:
    """RoomProfile-specific Redis operations.

    Read-only operations for room profiles.
    Room profiles are written by Evennia, workers only read them.
    """

    async def get_room_profile(
        self: "BaseRedisMUDClient",
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
        self: "BaseRedisMUDClient",
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
