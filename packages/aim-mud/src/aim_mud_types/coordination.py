# aim-mud-types/coordination.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Coordination and control structures stored in Redis."""

from datetime import datetime
from enum import Enum
from typing import Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from .helper import _utc_now
from .redis_keys import RedisKeys

if TYPE_CHECKING:
    import redis.asyncio as redis


class TurnRequestStatus(str, Enum):
    """Turn request status values."""
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAIL = "fail"
    READY = "ready"
    CRASHED = "crashed"
    ABORTED = "aborted"
    ABORT_REQUESTED = "abort_requested"


class TurnReason(str, Enum):
    """Reason for turn assignment."""
    EVENTS = "events"
    IDLE = "idle"
    DREAM = "dream"
    AGENT = "agent"
    CHOOSE = "choose"
    FLUSH = "flush"
    CLEAR = "clear"
    NEW = "new"
    RETRY = "retry"


class MUDTurnRequest(BaseModel):
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
        sequence_id: Event sequence ID for chronological ordering
        attempt_count: Number of retry attempts for this turn
        scenario: For dream turns, the scenario to run
        query: For dream turns, optional query text
        guidance: Optional guidance text
        conversation_id: For @new command, the new conversation ID
    """

    turn_id: str
    status: TurnRequestStatus = TurnRequestStatus.ASSIGNED
    reason: TurnReason = TurnReason.EVENTS
    message: Optional[str] = None
    heartbeat_at: datetime = Field(default_factory=_utc_now)
    assigned_at: datetime = Field(default_factory=_utc_now)
    sequence_id: Optional[int] = None
    attempt_count: int = 0

    # Retry coordination
    next_attempt_at: Optional[str] = None  # ISO format datetime string for retry timing
    status_reason: Optional[str] = None  # Human-readable reason for current status
    deadline_ms: Optional[str] = None  # Deadline in milliseconds (used by mediator)

    # Optional fields for specific turn types
    scenario: Optional[str] = None
    query: Optional[str] = None
    guidance: Optional[str] = None
    conversation_id: Optional[str] = None

    @classmethod
    async def from_redis(cls, redis_client: "redis.Redis", agent_id: str) -> Optional["MUDTurnRequest"]:
        """Fetch and deserialize turn request from Redis.

        Args:
            redis_client: Async Redis client
            agent_id: Agent identifier

        Returns:
            MUDTurnRequest object, or None if not found or invalid
        """
        key = RedisKeys.agent_turn_request(agent_id)
        data = await redis_client.hgetall(key)

        if not data:
            return None

        # Decode bytes to strings
        decoded: dict[str, str] = {}
        for k, v in data.items():
            if isinstance(k, bytes):
                k = k.decode("utf-8")
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            decoded[str(k)] = str(v)

        # Deserialize to MUDTurnRequest object
        try:
            return cls.model_validate(decoded)
        except Exception:
            return None


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
