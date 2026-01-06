# aim/app/mud/worker/profile.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Agent profile management for the MUD worker.

Handles loading and updating agent profile state in Redis.
Extracted from worker.py lines 409-566
"""

import json
import logging
from typing import TYPE_CHECKING, Optional

import redis.asyncio as redis

from aim_mud_types import RedisKeys
from aim_mud_types import RoomState, EntityState, WorldState, InventoryItem
from aim_mud_types.helper import _utc_now

if TYPE_CHECKING:
    from ...worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class ProfileMixin:
    """Mixin for agent profile management methods.

    These methods are mixed into MUDAgentWorker in main.py.
    """

    def _agent_profile_key(self: "MUDAgentWorker") -> str:
        """Return the Redis key for this agent's profile hash.

        Originally from worker.py lines 664-669
        """
        return RedisKeys.agent_profile(self.config.agent_id)

    async def _save_agent_profile(self: "MUDAgentWorker") -> None:
        """Save agent profile state to Redis."""
        key = RedisKeys.agent_profile(self.config.agent_id)

        # Get current conversation_id from manager
        conversation_id = None
        if self.conversation_manager:
            conversation_id = self.conversation_manager.get_current_conversation_id()

        fields = {
            "last_event_id": self.session.last_event_id or "",
        }

        if conversation_id:
            fields["conversation_id"] = conversation_id

        await self.redis.hset(key, mapping=fields)

    async def _load_agent_profile(self: "MUDAgentWorker") -> None:
        """Load agent profile state from Redis.

        Originally from worker.py lines 409-432
        """
        if not self.session:
            return
        data = await self.redis.hgetall(self._agent_profile_key())
        if not data:
            # Initialize profile with persona metadata
            await self._update_agent_profile(
                persona_id=self.persona.persona_id,
                agent_id=self.config.agent_id,
            )
            return
        decoded: dict[str, str] = {}
        for k, v in data.items():
            if isinstance(k, bytes):
                k = k.decode("utf-8")
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            decoded[str(k)] = str(v)

        last_event_id = decoded.get("last_event_id")
        if last_event_id:
            self.session.last_event_id = last_event_id

        # Load conversation_id if available
        conversation_id = decoded.get("conversation_id")
        if conversation_id and self.conversation_manager:
            self.conversation_manager.set_conversation_id(conversation_id)
            logger.info(f"Loaded conversation_id: {conversation_id}")

    async def _load_agent_world_state(self: "MUDAgentWorker") -> tuple[Optional[str], Optional[str]]:
        """Load inventory and room pointer from agent profile.

        Originally from worker.py lines 433-486

        Returns:
            Tuple of (room_id, character_id) when available.
        """
        if not self.session:
            return None, None
        try:
            data = await self.redis.hgetall(self._agent_profile_key())
        except redis.RedisError as e:
            logger.error(f"Redis error loading agent profile: {e}")
            return None, None
        if not data:
            return None, None

        def _decode(value):
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return value

        room_id = _decode(data.get(b"room_id") or data.get("room_id"))
        character_id = _decode(data.get(b"character_id") or data.get("character_id"))

        inventory_raw = _decode(data.get(b"inventory") or data.get("inventory"))
        inventory_items: list = []
        if inventory_raw:
            try:
                inventory_items = json.loads(inventory_raw)
            except json.JSONDecodeError:
                logger.warning("Invalid inventory JSON in agent profile")

        home = _decode(data.get(b"home") or data.get("home"))
        time_val = _decode(data.get(b"time") or data.get("time"))

        inventory = [
            InventoryItem.from_dict(i)
            for i in inventory_items
            if isinstance(i, dict)
        ]

        if self.session.world_state is None:
            self.session.world_state = WorldState(
                inventory=inventory,
                home=home,
                time=time_val,
            )
        else:
            self.session.world_state.inventory = inventory
            self.session.world_state.home = home
            self.session.world_state.time = time_val

        return room_id, character_id

    async def _load_room_profile(self: "MUDAgentWorker", room_id: Optional[str], character_id: Optional[str]) -> None:
        """Load room profile snapshot from Redis and merge into world_state.

        Originally from worker.py lines 487-553
        """
        if not self.session or not room_id:
            return
        try:
            data = await self.redis.hgetall(RedisKeys.room_profile(room_id))
        except redis.RedisError as e:
            logger.error(f"Redis error loading room profile {room_id}: {e}")
            return
        if not data:
            return

        def _decode(value):
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return value

        room_state_raw = _decode(data.get(b"room_state") or data.get("room_state"))
        entities_raw = _decode(data.get(b"entities_present") or data.get("entities_present"))

        room_state = None
        entities_present: list[EntityState] = []

        if room_state_raw:
            try:
                room_state = RoomState.from_dict(json.loads(room_state_raw))
            except Exception:
                logger.warning("Invalid room_state JSON in room profile")

        if entities_raw:
            try:
                parsed = json.loads(entities_raw)
                if isinstance(parsed, list):
                    entities_present = [
                        EntityState.from_dict(e) for e in parsed if isinstance(e, dict)
                    ]
            except Exception:
                logger.warning("Invalid entities_present JSON in room profile")

        if character_id:
            for entity in entities_present:
                if entity.entity_id == character_id:
                    entity.is_self = True
                    break

        if self.session.world_state is None:
            self.session.world_state = WorldState(
                room_state=room_state,
                entities_present=entities_present,
            )
        else:
            if room_state is not None:
                self.session.world_state.room_state = room_state
            self.session.world_state.entities_present = entities_present

        # Update chat manager location XML for unified memory queries
        if self._chat_manager and self.session.world_state and self.session.world_state.room_state:
            room_state_only = WorldState(
                room_state=self.session.world_state.room_state,
                entities_present=self.session.world_state.entities_present,
            )
            self._chat_manager.current_location = room_state_only.to_xml(include_self=False)

        if room_state is not None:
            self.session.current_room = room_state
        self.session.entities_present = entities_present

    async def _update_agent_profile(self: "MUDAgentWorker", persona_id: str = None, **fields: str) -> None:
        """Update agent profile fields in Redis.

        Originally from worker.py lines 554-566
        IMPORTANT: Includes the persona_id fix from the plan!

        Args:
            persona_id: Persona identifier (e.g., "Andi").
            **fields: Additional profile fields to update.
        """
        payload = {"updated_at": _utc_now().isoformat()}
        if persona_id is not None:
            payload["persona_id"] = persona_id
        payload.update({k: v for k, v in fields.items() if v is not None})
        await self.redis.hset(self._agent_profile_key(), mapping=payload)
