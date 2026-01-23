# aim-mud-types/profile.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Agent and room profile structures stored in Redis."""

from datetime import datetime
from typing import Optional
import json

from pydantic import BaseModel, Field, field_validator, field_serializer, model_validator

from ..helper import _utc_now, _datetime_to_unix, _unix_to_datetime
from .state import EntityState, RoomState
from .world_state import InventoryItem


class AgentProfile(BaseModel):
    """Agent profile stored in Redis.

    Stored in `agent:{id}:profile` Redis hash.
    Contains agent state that persists across worker restarts.

    Attributes:
        agent_id: Unique identifier for this agent
        persona_id: ID of the persona configuration
        last_event_id: Last processed event stream ID
        last_action_id: Last emitted action stream ID
        conversation_id: Current conversation ID
        updated_at: When profile was last updated (Unix timestamp)
    """

    agent_id: str
    persona_id: Optional[str] = None
    last_event_id: str = "0"
    last_action_id: Optional[str] = None
    conversation_id: Optional[str] = None
    updated_at: datetime = Field(default_factory=_utc_now)

    @field_validator("updated_at", mode="before")
    @classmethod
    def parse_updated_at(cls, v):
        return _unix_to_datetime(v) or _utc_now()

    @field_serializer("updated_at")
    def serialize_updated_at(self, dt: datetime) -> int:
        return _datetime_to_unix(dt)


class RoomProfile(BaseModel):
    """Room profile stored in Redis.

    Stored in `room:{id}:profile` Redis hash by Evennia.
    Contains current room state snapshot.

    Attributes:
        room_id: Unique identifier for this room
        name: Room name
        desc: Room description
        room_state: Full room state (exits, etc.)
        entities: Entities currently in the room (stored as 'entities_present' in Redis)
        updated_at: When profile was last updated
    """

    model_config = {"populate_by_name": True}  # Allow both field name and alias

    room_id: str = ""
    name: str = ""
    desc: str = ""
    room_state: Optional[RoomState] = None
    entities: list[EntityState] = Field(default_factory=list, alias="entities_present")
    updated_at: datetime = Field(default_factory=_utc_now)

    @field_validator("updated_at", mode="before")
    @classmethod
    def parse_updated_at(cls, v):
        return _unix_to_datetime(v) or _utc_now()

    @field_serializer("updated_at")
    def serialize_updated_at(self, dt: datetime) -> int:
        return _datetime_to_unix(dt)

    @field_validator('room_state', mode='before')
    @classmethod
    def parse_room_state(cls, v):
        """Parse room_state if stored as JSON string by Evennia."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return None
        return v

    @field_validator('entities', mode='before')
    @classmethod
    def parse_entities(cls, v):
        """Parse entities_present if stored as JSON string by Evennia."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return []
        return v

    @model_validator(mode='after')
    def extract_room_info(self):
        """Extract room_id and name from room_state if not set directly."""
        if self.room_state:
            # Extract room_id if not already set
            if not self.room_id and hasattr(self.room_state, 'room_id'):
                self.room_id = self.room_state.room_id
            # Extract name if not already set
            if not self.name and hasattr(self.room_state, 'name'):
                self.name = self.room_state.name
        return self
