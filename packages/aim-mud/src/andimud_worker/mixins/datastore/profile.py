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

        # Recover position from conversation (source of truth)
        if self.conversation_manager:
            conv_last_event_id = await self.conversation_manager.get_last_event_id()
            if conv_last_event_id:
                self.session.last_event_id = conv_last_event_id
                logger.info(f"Recovered last_event_id from conversation: {conv_last_event_id}")

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

            # Sync sleep state for sleep/wake tool filtering
            if hasattr(self._decision_strategy, "set_is_sleeping"):
                is_sleeping = await self._check_agent_is_sleeping()
                self._decision_strategy.set_is_sleeping(is_sleeping)

    async def _load_thought_content(self: "MUDAgentWorker") -> None:
        """Load stored thought content from Redis and set on strategies.

        Loads ThoughtState from agent:{id}:thought and sets thought_content
        on both decision and response strategies if present and within TTL (2 hours).
        """
        client = RedisMUDClient(self.redis)
        thought = await client.get_thought_state(self.config.agent_id)

        if not thought or not thought.content:
            return

        # Check 2-hour TTL (unchanged from current behavior)
        age_seconds = time.time() - thought.created_at.timestamp()
        if age_seconds >= 7200:  # 2-hour TTL
            logger.debug("Thought content expired (%.0fs old)", age_seconds)
            return

        if self._decision_strategy:
            self._decision_strategy.thought_content = thought.content
        if self._response_strategy:
            self._response_strategy.thought_content = thought.content

        logger.info(
            "Loaded thought content (%d chars, %.0fs old, conversation_index=%d)",
            len(thought.content), age_seconds, thought.last_conversation_index
        )

    async def _clear_thought_content(self: "MUDAgentWorker") -> None:
        """Clear thought content from strategies only (preserve Redis state).

        Note: This no longer deletes from Redis. The thought remains for
        throttle tracking. Use client.delete_thought_state() to fully remove.
        """
        if self._decision_strategy:
            self._decision_strategy.thought_content = ""
        if self._response_strategy:
            self._response_strategy.thought_content = ""

    async def _should_generate_new_thought(self: "MUDAgentWorker") -> bool:
        """Check if a new thought should be generated.

        Returns True if:
        - No thought exists, OR
        - Index gap >= 5 (5+ new entries since last thought), OR
        - Time elapsed >= 5 minutes AND index gap > 0

        Returns False if:
        - Index gap == 0 (nothing new happened, even if time elapsed)
        """
        from datetime import datetime

        client = RedisMUDClient(self.redis)

        # Get existing thought state
        thought = await client.get_thought_state(self.config.agent_id)

        if not thought:
            return True  # No thought exists

        # Calculate index gap (NEW)
        current_length = await client.get_conversation_length(self.config.agent_id)
        index_gap = current_length - thought.last_conversation_index

        # CRITICAL: Don't regenerate if nothing new happened
        if index_gap == 0:
            return False

        # Check if enough new entries (5+ entries)
        if index_gap >= 5:
            return True

        # Fall back to time-based throttle (only when index_gap > 0)
        time_elapsed = (datetime.utcnow() - thought.created_at).total_seconds()
        return time_elapsed >= 300  # 5 minutes

    async def _increment_thought_action_counter(self: "MUDAgentWorker") -> int:
        """Increment the thought action counter after autonomous action.

        Called after idle/autonomous actions complete. Event-reactive turns
        do NOT increment this counter.

        Returns:
            New counter value
        """
        client = RedisMUDClient(self.redis)
        return await client.increment_thought_action_counter(self.config.agent_id)

    async def _is_idle_active(self: "MUDAgentWorker") -> bool:
        """Return True if idle active flag is set for this agent."""
        key = RedisKeys.agent_idle_active(self.config.agent_id)
        try:
            raw = await self.redis.get(key)
        except Exception as e:
            logger.warning("Failed to read idle active flag: %s", e)
            return False
        if raw is None:
            return False
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return str(raw).strip().lower() in ("1", "true", "yes", "on")

    async def _load_workspace_state(self: "MUDAgentWorker") -> None:
        """Load workspace state from Redis and set on strategies.

        Loads workspace state from agent:{id}:workspace key and sets it on
        the chat manager. Also enables/disables close_book tool based on
        whether workspace has content.
        """
        workspace_key = RedisKeys.agent_workspace(self.config.agent_id)
        workspace_raw = await self.redis.get(workspace_key)

        if not workspace_raw:
            # No workspace content - clear state and disable close_book
            if self._chat_manager:
                self._chat_manager.current_workspace = None
            if self._decision_strategy:
                self._decision_strategy.set_workspace_active(False)
            return

        try:
            workspace_data = workspace_raw.decode("utf-8") if isinstance(workspace_raw, bytes) else workspace_raw
        except (UnicodeDecodeError, AttributeError) as e:
            logger.warning("Failed to decode workspace data: %s", e)
            if self._decision_strategy:
                self._decision_strategy.set_workspace_active(False)
            return

        if workspace_data:
            if self._chat_manager:
                self._chat_manager.current_workspace = workspace_data
            if self._decision_strategy:
                self._decision_strategy.set_workspace_active(True)
            logger.info("Loaded workspace state (%d chars)", len(workspace_data))
        else:
            if self._chat_manager:
                self._chat_manager.current_workspace = None
            if self._decision_strategy:
                self._decision_strategy.set_workspace_active(False)
            logger.info("Workspace state empty")

    async def _load_focus_state(self: "MUDAgentWorker") -> None:
        """Load focus state from Redis and set on strategies.

        Loads focus state from agent profile and restores it on the
        decision and response strategies.

        Expected format (new per-file ranges):
            {
                "files": [
                    {"path": "model.py", "start": 10, "end": 50},
                    {"path": "utils.py"}
                ],
                "height": 2,
                "depth": 1
            }
        """
        client = RedisMUDClient(self.redis)
        data = await client.get_agent_profile_raw(self.config.agent_id)
        if not data:
            return

        focus_raw = data.get("focus")
        if not focus_raw:
            return

        try:
            if isinstance(focus_raw, bytes):
                focus_raw = focus_raw.decode("utf-8")
            if not focus_raw.strip():
                return

            focus_data = json.loads(focus_raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("Failed to parse focus data: %s", e)
            return

        files = focus_data.get("files", [])
        if not files:
            return

        # Validate files format - must be list of dicts with 'path' key
        if not isinstance(files, list):
            logger.warning("Invalid focus files format: expected list, got %s", type(files))
            return

        # Check if files are in new format (list of dicts) or old format (list of strings)
        if files and isinstance(files[0], str):
            # Old format detected - cannot migrate without global start/end
            # Log warning and skip loading
            logger.warning(
                "Focus data in old format (list of strings). "
                "Clear focus and re-set with new per-file format."
            )
            return

        # Import FocusRequest here to avoid circular imports
        from aim_code.strategy.base import FocusRequest

        focus_request = FocusRequest(
            files=files,  # list[dict] with path, start, end
            height=focus_data.get("height", 1),
            depth=focus_data.get("depth", 1),
        )

        # Set focus on decision strategy
        if self._decision_strategy is not None and hasattr(self._decision_strategy, "set_focus"):
            self._decision_strategy.set_focus(focus_request)

        # Set focus on response strategy
        if self._response_strategy is not None and hasattr(self._response_strategy, "set_focus"):
            self._response_strategy.set_focus(focus_request)

        logger.info(
            "Loaded focus state (%d files)",
            len(files),
        )

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
