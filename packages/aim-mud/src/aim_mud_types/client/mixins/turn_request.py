# aim-mud-types/client/mixins/turn_request.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""MUDTurnRequest-specific operations with CAS support."""

from typing import Optional, TYPE_CHECKING

from ...redis_keys import RedisKeys
from ...coordination import MUDTurnRequest
from ...helper import _utc_now

if TYPE_CHECKING:
    from .. import BaseRedisMUDClient


class TurnRequestMixin:
    """MUDTurnRequest-specific Redis operations.

    Provides CRUD operations for turn request coordination with
    CAS (Compare-And-Swap) support to prevent race conditions.
    """

    async def get_turn_request(self: "BaseRedisMUDClient", agent_id: str) -> Optional[MUDTurnRequest]:
        """Fetch turn request for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            MUDTurnRequest object, or None if not found
        """
        key = RedisKeys.agent_turn_request(agent_id)
        return await self._get_hash(MUDTurnRequest, key)

    async def create_turn_request(
        self: "BaseRedisMUDClient",
        agent_id: str,
        turn_request: MUDTurnRequest
    ) -> bool:
        """Create a new turn request.

        Uses atomic EXISTS check to prevent overwriting existing requests.

        Args:
            agent_id: Agent identifier
            turn_request: Complete MUDTurnRequest object

        Returns:
            True if created, False if turn request already exists
        """
        key = RedisKeys.agent_turn_request(agent_id)
        return await self._create_hash(key, turn_request, exists_ok=False)

    async def update_turn_request(
        self: "BaseRedisMUDClient",
        agent_id: str,
        turn_request: MUDTurnRequest,
        expected_turn_id: str
    ) -> bool:
        """Update turn request with CAS on turn_id.

        Atomically checks that turn_id matches before updating.
        Prevents race conditions between workers.

        Args:
            agent_id: Agent identifier
            turn_request: Updated MUDTurnRequest object
            expected_turn_id: Expected turn_id value for CAS check

        Returns:
            True if updated, False if CAS check failed
        """
        key = RedisKeys.agent_turn_request(agent_id)
        return await self._update_hash(
            key,
            turn_request,
            cas_field="turn_id",
            cas_value=expected_turn_id
        )

    async def heartbeat_turn_request(
        self: "BaseRedisMUDClient",
        agent_id: str
    ) -> bool:
        """Update heartbeat timestamp.

        Efficient single-field update for worker heartbeats.

        Args:
            agent_id: Agent identifier

        Returns:
            True if updated
        """
        key = RedisKeys.agent_turn_request(agent_id)
        return await self._update_fields(
            key,
            {"heartbeat_at": _utc_now()}
        )
