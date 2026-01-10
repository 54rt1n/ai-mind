# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""MUD event model for world events."""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_serializer

from .enums import EventType, ActorType
from .world_state import WorldState
from .helper import _utc_now

logger = logging.getLogger(__name__)

class MUDEvent(BaseModel):
    """A world event from the MUD.

    Events are published by Evennia to the mud:events stream and delivered
    to agents via their per-agent streams after enrichment by the mediator.
    Each event represents something that happened in the world that agents
    can perceive.

    The mediator adds room_state and entities_present before distributing
    to agents, so agents receive full context without needing to query.

    Attributes:
        event_id: Redis stream message ID.
        event_type: Type of event (speech, emote, movement, etc.).
        actor: Name/key of the entity that caused the event.
        actor_id: Stable id (dbref) of the actor, if available.
        actor_type: Type of actor (player, ai, npc, system).
        room_id: Room where the event occurred.
        room_name: Human-readable room name.
        content: Event content (e.g., speech text, emote description).
        target: Optional target of the event.
        target_id: Optional stable id (dbref) of the target, if available.
        timestamp: When the event occurred (UTC).
        metadata: Additional event-specific data.
        room_state: Optional room state payload (legacy).
        entities_present: Optional entity list payload (legacy).
        world_state: Optional world snapshot payload (legacy).
    """

    event_id: str = ""
    sequence_id: Optional[int] = None
    event_type: EventType
    actor: str
    actor_id: str = ""
    actor_type: ActorType = ActorType.PLAYER
    room_id: str
    room_name: str = ""
    content: str = ""
    target: Optional[str] = None
    target_id: str = ""
    timestamp: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Enrichment fields (added by mediator)
    room_state: Optional[dict[str, Any]] = None
    entities_present: list[dict[str, Any]] = Field(default_factory=list)
    world_state: Optional[WorldState] = None

    @field_serializer("timestamp")
    def serialize_timestamp(self, dt: datetime, _info: Any) -> str:
        """Serialize datetime to ISO format string."""
        return dt.isoformat()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MUDEvent":
        """Create MUDEvent from a dictionary.

        Handles both raw events (from Evennia) and enriched events
        (from mediator). Supports multiple key naming conventions.

        Args:
            data: Dictionary with event data (e.g., from Redis JSON).

        Returns:
            MUDEvent instance.
        """
        # Parse timestamp if string
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            if timestamp:  # Non-empty string
                # Handle both Z suffix and explicit offset
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            else:  # Empty string - log warning since this is always a data error
                logger.warning("Event data contains empty timestamp string, defaulting to current time")
                timestamp = _utc_now()
        elif timestamp is None:
            timestamp = _utc_now()

        world_state_data = data.get("world_state")
        world_state = (
            WorldState.from_dict(world_state_data)
            if world_state_data is not None
            else None
        )

        return cls(
            # Support both 'id' (Redis) and 'event_id' (normalized)
            event_id=data.get("id", data.get("event_id", "")),
            sequence_id=data.get("sequence_id"),
            # Support both 'type' (compact) and 'event_type' (explicit)
            event_type=EventType(data.get("type", data.get("event_type", "system"))),
            actor=data.get("actor", ""),
            actor_id=data.get("actor_id", ""),
            actor_type=ActorType(data.get("actor_type", "player")),
            room_id=data.get("room_id", ""),
            room_name=data.get("room_name", ""),
            content=data.get("content", ""),
            target=data.get("target"),
            target_id=data.get("target_id", ""),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
            # Enrichment fields
            room_state=data.get("room_state"),
            entities_present=data.get("entities_present", []),
            world_state=world_state,
        )

    def to_redis_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Redis stream publishing.

        Uses compact key names for efficiency.

        Returns:
            Dictionary ready for JSON serialization to Redis.
        """
        result = {
            "type": self.event_type.value,
            "actor": self.actor,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type.value,
            "room_id": self.room_id,
            "room_name": self.room_name,
            "content": self.content,
            "target": self.target,
            "target_id": self.target_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

        # Include sequence_id if set (for self-actions written directly to agent streams)
        if self.sequence_id is not None:
            result["sequence_id"] = self.sequence_id

        return result

    def is_self_speech_echo(self) -> bool:
        """Check if this is a self-speech echo event.

        These are events where the agent's own speech action was echoed
        back from Evennia. They should not be added to conversation because
        the agent already has their speech in the assistant turn.

        Returns:
            True if this is a self-speech echo that should be filtered.
        """
        from .enums import EventType
        return (
            self.metadata.get("is_self_action", False) and
            self.event_type == EventType.SPEECH
        )
