# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""State models for rooms and entities in the MUD world."""

from typing import Any

from pydantic import BaseModel, Field


class RoomState(BaseModel):
    """Current state of a room in the MUD.

    Represents the spatial context for an agent, including the room's
    identity, description, and available exits. This is included in
    enriched events by the mediator.

    Attributes:
        room_id: Unique identifier for the room (e.g., "#123").
        name: Human-readable room name.
        description: Full description of the room.
        exits: Mapping of direction names to destination room IDs.
    """

    room_id: str
    name: str
    description: str = ""
    exits: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoomState":
        """Create RoomState from a dictionary.

        Handles missing optional fields gracefully.

        Args:
            data: Dictionary with room data (e.g., from JSON).

        Returns:
            RoomState instance.
        """
        return cls(
            room_id=data.get("room_id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            exits=data.get("exits", {}),
        )


class EntityState(BaseModel):
    """State of an entity present in a room.

    Represents any character, object, or NPC that can be perceived
    by an agent. Included in enriched events by the mediator.

    Attributes:
        entity_id: Unique identifier for the entity.
        name: Display name of the entity.
        entity_type: Type of entity (player, ai, npc, object).
        description: Optional description of the entity.
        is_self: True if this entity is the perceiving agent.
    """

    entity_id: str
    name: str
    entity_type: str = "object"
    description: str = ""
    is_self: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EntityState":
        """Create EntityState from a dictionary.

        Handles missing optional fields gracefully.

        Args:
            data: Dictionary with entity data (e.g., from JSON).

        Returns:
            EntityState instance.
        """
        return cls(
            entity_id=data.get("entity_id", ""),
            name=data.get("name", ""),
            entity_type=data.get("entity_type", "object"),
            description=data.get("description", ""),
            is_self=data.get("is_self", False),
        )
