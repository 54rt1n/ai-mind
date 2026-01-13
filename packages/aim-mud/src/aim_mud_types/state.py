# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""State models for rooms and entities in the MUD world."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


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
    tags: list[str] = Field(default_factory=list)

    @field_validator("tags", mode="before")
    @classmethod
    def ensure_tags_list(cls, v):
        """Ensure tags is a list, converting None to []."""
        return v if v else []


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
    tags: list[str] = Field(default_factory=list)
    agent_id: str = ""
    contents: list[str] = Field(default_factory=list)

    @field_validator("tags", mode="before")
    @classmethod
    def ensure_tags_list(cls, v):
        """Ensure tags is a list, converting None to []."""
        return v if v else []

    @field_validator("agent_id", mode="before")
    @classmethod
    def ensure_agent_id(cls, v):
        """Ensure agent_id is a string, converting None to empty string."""
        return v if v else ""

    @field_validator("contents", mode="before")
    @classmethod
    def ensure_contents_list(cls, v):
        """Ensure contents is a list, converting None to []."""
        return v if v else []
