# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""MUD event model for world events."""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_serializer, field_validator, AliasChoices

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

    event_id: str = Field(
        default="",
        validation_alias=AliasChoices("id", "event_id")
    )
    sequence_id: Optional[int] = None
    event_type: EventType = Field(
        serialization_alias="type",
        validation_alias=AliasChoices("type", "event_type")
    )
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

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v):
        """Parse timestamp from various formats."""
        if isinstance(v, str):
            if not v:
                logger.warning("Event data contains empty timestamp string, defaulting to current time")
                return _utc_now()
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        if v is None:
            return _utc_now()
        return v

    @field_serializer("timestamp")
    def serialize_timestamp(self, dt: datetime, _info: Any) -> str:
        """Serialize datetime to ISO format string."""
        return dt.isoformat()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MUDEvent":
        """Create MUDEvent from dictionary.

        Deprecated: Prefer using model_validate() directly.
        This wrapper is kept for backward compatibility.
        """
        return cls.model_validate(data)

    def to_redis_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Redis stream publishing.

        Uses compact key names via aliases and excludes enrichment fields.

        Architectural note on sequence_id:
        - Evennia events: Created without sequence_id (mediator assigns it externally)
        - Self-actions: Created with sequence_id from turn_request (required field)

        This method handles both cases by conditionally excluding sequence_id when None.
        """
        result = self.model_dump(
            by_alias=True,
            mode="json",
            exclude={"event_id", "room_state", "entities_present", "world_state"}
        )

        # Exclude sequence_id if None (Evennia events before mediator assignment)
        # Self-actions always have sequence_id from turn_request, so this is a no-op
        # Note: Can't use exclude_none=True globally because target=None must be included
        if self.sequence_id is None:
            result.pop("sequence_id", None)

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
