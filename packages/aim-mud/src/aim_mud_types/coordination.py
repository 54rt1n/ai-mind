# aim-mud-types/coordination.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Coordination and control structures stored in Redis."""

import json
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from .helper import _utc_now
from .redis_keys import RedisKeys


class TurnRequestStatus(str, Enum):
    """Turn request status values."""
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    RETRY = "retry"            # Temporary failure, will retry after backoff
    FAIL = "fail"              # Permanent failure after max attempts
    READY = "ready"
    CRASHED = "crashed"
    ABORTED = "aborted"
    ABORT_REQUESTED = "abort_requested"
    EXECUTE = "execute"        # Command ready to execute (non-blocking)
    EXECUTING = "executing"    # Command is executing (non-blocking)


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

    def is_immediate_command(self) -> bool:
        """Return True if this is an immediate command that uses EXECUTE status.

        Immediate commands:
        - Execute with interrupt semantics (don't wait for earlier events)
        - Don't block other workers
        - Skip event draining and turn guard
        """
        return self in {
            TurnReason.FLUSH,
            TurnReason.CLEAR,
            TurnReason.NEW,
        }


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
        completed_at: When turn execution finished (set on terminal states)
        sequence_id: Event sequence ID for chronological ordering (REQUIRED)
        attempt_count: Number of retry attempts for this turn
        next_attempt_at: ISO format datetime string for retry timing
        status_reason: Human-readable reason for current status
        deadline_ms: Deadline in milliseconds (used by mediator)
        metadata: Turn-specific context (scenario, query, guidance, conversation_id, etc.)
    """

    turn_id: str
    status: TurnRequestStatus = TurnRequestStatus.ASSIGNED
    reason: TurnReason = TurnReason.EVENTS
    message: Optional[str] = None
    heartbeat_at: datetime = Field(default_factory=_utc_now)
    assigned_at: datetime = Field(default_factory=_utc_now)
    completed_at: Optional[datetime] = None  # When turn execution finished
    sequence_id: int  # REQUIRED: Every turn must have a sequence_id
    attempt_count: int = 0

    # Retry coordination
    next_attempt_at: Optional[str] = None  # ISO format datetime string for retry timing
    status_reason: Optional[str] = None  # Human-readable reason for current status
    deadline_ms: Optional[str] = None  # Deadline in milliseconds (used by mediator)

    # Metadata for turn-specific context
    metadata: Optional[dict] = None

    @field_validator("metadata", mode="before")
    @classmethod
    def parse_metadata(cls, v):
        """Parse metadata from JSON string (Redis) or dict (Python).

        When metadata is retrieved from Redis, it comes as a JSON string.
        This validator converts it back to a dict for Pydantic validation.

        Args:
            v: Either a JSON string (from Redis) or dict (from Python code)

        Returns:
            Dict or None

        Raises:
            ValueError: If v is a string but not valid JSON
        """
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in metadata field: {e}")
        return v

    def is_available_for_assignment(self, now: Optional[datetime] = None) -> bool:
        """Return True if a new turn can be assigned to this agent.

        Mirrors mediator availability rules:
        - Busy: ASSIGNED, IN_PROGRESS, ABORT_REQUESTED, EXECUTING, EXECUTE
        - Crashed: CRASHED
        - Retry/Fail: available only after next_attempt_at (if set)
        """
        status = self.status
        if status == TurnRequestStatus.CRASHED:
            return False

        if status in (
            TurnRequestStatus.ASSIGNED,
            TurnRequestStatus.IN_PROGRESS,
            TurnRequestStatus.ABORT_REQUESTED,
            TurnRequestStatus.EXECUTING,
            TurnRequestStatus.EXECUTE,
        ):
            return False

        if status in (TurnRequestStatus.RETRY, TurnRequestStatus.FAIL):
            if self.next_attempt_at:
                try:
                    next_attempt_at = datetime.fromisoformat(self.next_attempt_at)
                except (TypeError, ValueError):
                    # If malformed, be conservative and treat as unavailable
                    return False
                current = now or datetime.now(next_attempt_at.tzinfo)
                if current < next_attempt_at:
                    return False

        return True


class DreamerState(BaseModel):
    """Automatic dreaming configuration.

    Stored in `agent:{id}:dreamer` Redis hash.

    Attributes:
        enabled: Whether automatic dreaming is enabled
        idle_threshold_seconds: Seconds of idle before triggering dream
        token_threshold: Minimum tokens accumulated before dream
        last_dream_at: When the last automatic dream occurred (ISO format)
        last_dream_scenario: Scenario that was last executed
        pending_pipeline_id: Pipeline ID if dream is currently running
    """

    enabled: bool = False
    idle_threshold_seconds: int = 600
    token_threshold: int = 5000
    last_dream_at: Optional[str] = None  # ISO datetime string
    last_dream_scenario: Optional[str] = None
    pending_pipeline_id: Optional[str] = None
