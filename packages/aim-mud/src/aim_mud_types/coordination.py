# aim-mud-types/coordination.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Coordination and control structures stored in Redis."""

from datetime import datetime
from typing import Optional, Literal

from pydantic import BaseModel, Field

from .helper import _utc_now


class TurnRequest(BaseModel):
    """Turn request coordination structure.

    Stored in `agent:{id}:turn_request` Redis hash.
    Mediator assigns turns, worker processes and updates status.

    Attributes:
        turn_id: Unique identifier for this turn
        status: Current turn status
        reason: Why the turn was assigned
        message: Optional status message
        heartbeat_at: Last heartbeat from worker
        assigned_at: When turn was assigned by mediator
        scenario: For dream turns, the scenario to run
        query: For dream turns, optional query text
        guidance: Optional guidance text
        conversation_id: For @new command, the new conversation ID
    """

    turn_id: str
    status: Literal["assigned", "in_progress", "done", "fail", "ready"] = "assigned"
    reason: Literal["events", "idle", "dream", "agent", "choose", "flush", "clear", "new"] = "events"
    message: Optional[str] = None
    heartbeat_at: datetime = Field(default_factory=_utc_now)
    assigned_at: datetime = Field(default_factory=_utc_now)

    # Optional fields for specific turn types
    scenario: Optional[str] = None
    query: Optional[str] = None
    guidance: Optional[str] = None
    conversation_id: Optional[str] = None


class DreamerState(BaseModel):
    """Automatic dreaming configuration.

    Stored in `agent:{id}:dreamer` Redis hash.

    Attributes:
        enabled: Whether automatic dreaming is enabled
        idle_threshold_seconds: Seconds of idle before triggering dream
        token_threshold: Minimum tokens accumulated before dream
        last_dream_time: When the last automatic dream occurred
    """

    enabled: bool = False
    idle_threshold_seconds: int = 600
    token_threshold: int = 5000
    last_dream_time: Optional[datetime] = None
