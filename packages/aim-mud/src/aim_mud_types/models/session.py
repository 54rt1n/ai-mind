# aim/app/mud/session.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Session and event models for MUD agent integration.

This module provides AIM-specific session models (MUDTurn, MUDSession)
and re-exports shared types from aim_mud_types for convenience.

The shared types are defined in packages/aim-mud-types and used by
both AIM and Evennia for type-safe communication.
"""

from datetime import datetime
from typing import Optional, Any

from pydantic import BaseModel, Field, field_serializer

# Import shared types using relative imports
from .state import RoomState, EntityState
from .world_state import WorldState
from .events import MUDEvent
from .actions import MUDAction

from ..helper import _utc_now, _datetime_to_unix, _unix_to_datetime

class MUDTurn(BaseModel):
    """One complete agent turn: perception -> reasoning -> action.

    Represents the full context and outcome of a single agent decision
    cycle. Used for session history and memory persistence.

    This is an AIM-specific model that references the shared types
    from aim_mud_types.

    Attributes:
        timestamp: When the turn was processed.
        events_received: Events that triggered this turn.
        room_context: Room state at time of decision.
        entities_context: Entities present at time of decision.
        memories_retrieved: Memories retrieved for context.
        thinking: Agent's internal reasoning.
        actions_taken: Actions the agent decided to take.
        doc_id: Document ID if persisted to memory.
    """

    timestamp: datetime = Field(default_factory=_utc_now)
    events_received: list[MUDEvent] = Field(default_factory=list)
    room_context: Optional[RoomState] = None
    entities_context: list[EntityState] = Field(default_factory=list)
    memories_retrieved: list[dict[str, Any]] = Field(default_factory=list)
    thinking: str = ""
    actions_taken: list[MUDAction] = Field(default_factory=list)
    doc_id: Optional[str] = None

    @field_serializer("timestamp")
    def serialize_timestamp(self, dt: datetime, _info: Any) -> int:
        """Serialize datetime to Unix timestamp."""
        return _datetime_to_unix(dt)


class MUDSession(BaseModel):
    """Session state for a MUD agent.

    Tracks the agent's current context, recent history, and stream
    position. This is the primary state object for the worker loop.

    This is an AIM-specific model that maintains agent session state
    across turns.

    Attributes:
        agent_id: Unique identifier for this agent.
        persona_id: ID of the persona configuration.
        current_room: Current room state (rebuilt each turn).
        entities_present: Entities in current room (rebuilt each turn).
        pending_events: Events waiting to be processed.
        world_state: Latest enriched world snapshot.
        recent_turns: Rolling history of recent turns.
        last_event_id: Redis stream ID for resumption.
        last_action_time: Timestamp of last action taken.
        created_at: When session was created.
        updated_at: When session was last updated.
    """

    agent_id: str
    persona_id: str

    # Immediate context (rebuilt each turn)
    current_room: Optional[RoomState] = None
    entities_present: list[EntityState] = Field(default_factory=list)
    pending_events: list[MUDEvent] = Field(default_factory=list)
    world_state: Optional[WorldState] = None

    # Rolling history (persists, compressed over time)
    recent_turns: list[MUDTurn] = Field(default_factory=list)
    max_recent_turns: int = 20

    # Stream tracking
    last_event_id: str = "0"

    # Timing
    last_action_time: Optional[datetime] = None
    last_event_time: Optional[datetime] = None

    # Timestamps
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    @field_serializer("created_at", "updated_at", "last_action_time", "last_event_time")
    def serialize_datetime(
        self, dt: Optional[datetime], _info: Any
    ) -> Optional[int]:
        """Serialize datetime to Unix timestamp."""
        return _datetime_to_unix(dt)

    def add_turn(self, turn: MUDTurn) -> None:
        """Add a turn to the session history.

        Updates the session timestamp and appends the turn.
        """
        self.recent_turns.append(turn)
        if self.max_recent_turns > 0 and len(self.recent_turns) > self.max_recent_turns:
            self.recent_turns = self.recent_turns[-self.max_recent_turns :]
        self.last_action_time = turn.timestamp
        self.updated_at = _utc_now()

    def get_last_turn(self) -> Optional[MUDTurn]:
        """Get the most recent turn, if any."""
        return self.recent_turns[-1] if self.recent_turns else None

    def clear_pending_events(self) -> None:
        """Clear pending events after processing."""
        self.pending_events = []
        self.updated_at = _utc_now()
