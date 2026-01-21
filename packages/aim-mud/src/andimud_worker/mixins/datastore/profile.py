# aim/app/mud/worker/profile.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Agent profile management for the MUD worker.

Handles loading and updating agent profile state in Redis.
Extracted from worker.py lines 409-566
"""

import json
import logging
from typing import TYPE_CHECKING, Optional
import time

from aim_mud_types import RoomState, EntityState, WorldState, InventoryItem
from aim_mud_types.client import RedisMUDClient
from aim_mud_types import RedisKeys
from aim_mud_types.client import RedisMUDClient

if TYPE_CHECKING:
    from ...worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class ProfileMixin:
    """Mixin for agent profile management methods.

    These methods are mixed into MUDAgentWorker in main.py.
    """

    async def _save_agent_profile(self: "MUDAgentWorker") -> None:
        """Save agent profile state to Redis."""
        # Get current conversation_id from manager
        conversation_id = None
        if self.conversation_manager:
            conversation_id = self.conversation_manager.get_current_conversation_id()

        fields = {
            "last_event_id": self.session.last_event_id or "",
        }

        if conversation_id:
            fields["conversation_id"] = conversation_id

        client = RedisMUDClient(self.redis)
        await client.update_agent_profile_fields(self.config.agent_id, **fields)

    async def _load_agent_profile(self: "MUDAgentWorker") -> None:
        """Load agent profile state from Redis.

        Originally from worker.py lines 409-432
        """
        if not self.session:
            return
        client = RedisMUDClient(self.redis)
        decoded = await client.get_agent_profile_raw(self.config.agent_id)
        if not decoded:
            # Initialize profile with persona metadata
            await self._update_agent_profile(
                persona_id=self.persona.persona_id,
            )
            return

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
        client = RedisMUDClient(self.redis)
        data = await client.get_agent_profile_raw(self.config.agent_id)
        if not data:
            return None, None
        room_id = data.get("room_id")
        character_id = data.get("character_id")

        inventory_raw = data.get("inventory")
        inventory_items: list = []
        if inventory_raw:
            try:
                inventory_items = json.loads(inventory_raw)
            except json.JSONDecodeError:
                logger.warning("Invalid inventory JSON in agent profile")

        home = data.get("home")
        time_val = data.get("time")

        inventory = [
            InventoryItem.model_validate(i)
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
        client = RedisMUDClient(self.redis)
        data = await client.get_room_profile_raw(room_id)
        if not data:
            return
        room_state_raw = data.get("room_state")
        entities_raw = data.get("entities_present")

        room_state = None
        entities_present: list[EntityState] = []

        if room_state_raw:
            try:
                room_state = RoomState.model_validate(json.loads(room_state_raw))
            except Exception:
                logger.warning("Invalid room_state JSON in room profile")

        if entities_raw:
            try:
                parsed = json.loads(entities_raw)
                if isinstance(parsed, list):
                    entities_present = [
                        EntityState.model_validate(e) for e in parsed if isinstance(e, dict)
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

        # Update decision tools based on active room auras (if supported)
        if room_state and hasattr(self, "_decision_strategy") and self._decision_strategy:
            auras = []
            for aura in getattr(room_state, "auras", []) or []:
                if isinstance(aura, dict):
                    name = aura.get("name")
                else:
                    name = getattr(aura, "name", None)
                if name:
                    auras.append(name)
            if hasattr(self._decision_strategy, "update_aura_tools") and self.chat_config:
                self._decision_strategy.update_aura_tools(auras, self.chat_config.tools_path)

    async def _load_thought_content(self: "MUDAgentWorker") -> None:
        """Load stored thought content from Redis and set on strategies.

        Loads thought content from agent:{id}:thought key and sets it on
        both decision and response strategies if present and within TTL (2 hours).
        """
        thought_key = RedisKeys.agent_thought(self.config.agent_id)
        thought_raw = await self.redis.get(thought_key)
        if not thought_raw:
            return

        try:
            raw_str = thought_raw.decode("utf-8") if isinstance(thought_raw, bytes) else thought_raw
            thought_data = json.loads(raw_str)
            thought_content = thought_data.get("content", "")
            timestamp = thought_data.get("timestamp", 0)
            age_seconds = time.time() - timestamp

            if age_seconds < 7200 and thought_content:  # 2-hour TTL check
                if self._decision_strategy:
                    self._decision_strategy.thought_content = thought_content
                if self._response_strategy:
                    self._response_strategy.thought_content = thought_content
                logger.info("Loaded thought content (%d chars, %.0fs old)", len(thought_content), age_seconds)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse thought JSON: %s", e)

    async def _update_agent_profile(self: "MUDAgentWorker", persona_id: str = None, **fields: str) -> None:
        """Update agent profile fields in Redis.

        Originally from worker.py lines 554-566
        IMPORTANT: Includes the persona_id fix from the plan!

        Args:
            persona_id: Persona identifier (e.g., "Andi").
            **fields: Additional profile fields to update.
        """
        payload: dict[str, str] = {}
        if persona_id is not None:
            payload["persona_id"] = persona_id
        payload.update({k: v for k, v in fields.items() if v is not None})
        client = RedisMUDClient(self.redis)
        await client.update_agent_profile_fields(
            self.config.agent_id,
            touch_updated_at=True,
            **payload,
        )

    async def _check_agent_is_sleeping(self: "MUDAgentWorker") -> bool:
        """Check if agent is sleeping."""
        client = RedisMUDClient(self.redis)
        return await client.get_agent_is_sleeping(self.config.agent_id)