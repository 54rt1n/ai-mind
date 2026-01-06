# aim-mud-types/profile.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Agent and room profile structures stored in Redis."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from .helper import _utc_now
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
        updated_at: When profile was last updated
    """

    agent_id: str
    persona_id: str
    last_event_id: str = "0"
    last_action_id: Optional[str] = None
    conversation_id: Optional[str] = None
    updated_at: datetime = Field(default_factory=_utc_now)


class RoomProfile(BaseModel):
    """Room profile stored in Redis.

    Stored in `room:{id}:profile` Redis hash by Evennia.
    Contains current room state snapshot.

    Attributes:
        room_id: Unique identifier for this room
        name: Room name
        desc: Room description
        room_state: Full room state (exits, etc.)
        entities: Entities currently in the room
        updated_at: When profile was last updated
    """

    room_id: str
    name: str = ""
    desc: str = ""
    room_state: Optional[RoomState] = None
    entities: list[EntityState] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=_utc_now)
