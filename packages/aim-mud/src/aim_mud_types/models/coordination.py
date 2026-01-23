# aim-mud-types/coordination.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Coordination and control structures stored in Redis."""

import json
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, field_serializer

from ..helper import _utc_now, _datetime_to_unix, _unix_to_datetime
from ..redis_keys import RedisKeys


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
    DREAM = "dream"          # Legacy - inline dream execution (DEPRECATED)
    AGENT = "agent"
    CHOOSE = "choose"
    FLUSH = "flush"
    CLEAR = "clear"
    NEW = "new"
    RETRY = "retry"
    THINK = "think"          # Process with injected thought content
    SLEEP = "sleep"          # Agent is falling asleep
    WAKE = "wake"            # Agent is waking up

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
    All datetime fields are serialized as Unix timestamps (integers).

    Attributes:
        turn_id: Unique identifier for this turn
        status: Current turn status
        reason: Why the turn was assigned
        message: Optional status message
        heartbeat_at: Last heartbeat from worker (Unix timestamp)
        assigned_at: When turn was assigned by mediator (Unix timestamp)
        completed_at: When turn execution finished (Unix timestamp)
        sequence_id: Event sequence ID for chronological ordering (REQUIRED)
        attempt_count: Number of retry attempts for this turn
        next_attempt_at: Unix timestamp for retry timing
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
    next_attempt_at: Optional[datetime] = None  # Unix timestamp for retry timing
    status_reason: Optional[str] = None  # Human-readable reason for current status
    deadline_ms: Optional[str] = None  # Deadline in milliseconds (used by mediator)

    # Metadata for turn-specific context
    metadata: Optional[dict] = None

    # Validators to parse Unix timestamps from Redis
    @field_validator("heartbeat_at", "assigned_at", mode="before")
    @classmethod
    def parse_required_datetime(cls, v):
        return _unix_to_datetime(v) or _utc_now()

    @field_validator("completed_at", "next_attempt_at", mode="before")
    @classmethod
    def parse_optional_datetime(cls, v):
        return _unix_to_datetime(v)

    # Serializers to output Unix timestamps
    @field_serializer("heartbeat_at", "assigned_at")
    def serialize_required_datetime(self, dt: datetime) -> int:
        return _datetime_to_unix(dt)

    @field_serializer("completed_at", "next_attempt_at")
    def serialize_optional_datetime(self, dt: Optional[datetime]) -> Optional[int]:
        return _datetime_to_unix(dt)

    @field_validator("metadata", mode="before")
    @classmethod
    def parse_metadata(cls, v):
        """Parse metadata from JSON string (Redis) or dict (Python)."""
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in metadata field: {e}")
        return v

    @field_serializer("metadata")
    def serialize_metadata(self, v: Optional[dict]) -> Optional[str]:
        """Serialize metadata dict to JSON string for Redis."""
        if v is None:
            return None
        return json.dumps(v)

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
                current = now or _utc_now()
                if current < self.next_attempt_at:
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


class DreamStatus(str, Enum):
    """Dream pipeline execution status."""
    PENDING = "pending"      # Initialized, not started
    RUNNING = "running"      # Steps in progress
    COMPLETE = "complete"    # All steps finished
    FAILED = "failed"        # Pipeline failed, retry exhausted
    ABORTED = "aborted"      # Pipeline aborted by step returning "abort"


class DreamingState(BaseModel):
    """Serialized state for step-by-step dream execution.

    Stored in `agent:{id}:dreaming` Redis hash.
    All datetime fields are serialized as Unix timestamps (integers).
    """

    # Identity
    pipeline_id: str                    # UUID for this dream session
    agent_id: str                       # Owner agent

    # Status
    status: DreamStatus                 # Current execution state
    created_at: datetime                # Initialization timestamp
    updated_at: datetime                # Last step completion
    completed_at: Optional[datetime] = None

    # Pipeline Configuration (frozen at init)
    scenario_name: str                  # e.g., "analysis_dialogue"
    execution_order: list[str] = Field(default_factory=list)  # All step IDs in order
    query: Optional[str] = None         # Optional query text
    guidance: Optional[str] = None      # User guidance (manual dreams)
    conversation_id: str = ""           # Target conversation
    base_model: str = ""                # Model used for execution

    # Step Execution State
    step_index: int = 0                 # Current position (0-based)
    completed_steps: list[str] = Field(default_factory=list)  # Finished step IDs
    step_doc_ids: dict[str, str] = Field(default_factory=dict)  # step_id → doc_id
    context_doc_ids: list[str] = Field(default_factory=list)   # Accumulated context

    # Retry/Error Handling
    current_step_attempts: int = 0      # Retry count for current step
    max_step_retries: int = 3           # Configurable retry limit
    next_retry_at: Optional[datetime] = None  # Backoff timestamp
    last_error: Optional[str] = None    # Most recent error message

    # Heartbeat
    heartbeat_at: Optional[datetime] = None  # Worker liveness signal
    heartbeat_timeout_seconds: int = 300     # 5 minutes

    # Scenario Context (for dialogue flow)
    scenario_config: dict = Field(default_factory=dict)  # Frozen scenario YAML
    persona_config: dict = Field(default_factory=dict)   # Frozen persona config

    # Strategy-based scenario execution (Phase C: ScenarioState & Persistence)
    # These fields support the new step type strategy pattern for complex workflows
    framework: Optional[dict] = None  # ScenarioFramework (immutable definition)
    state: Optional[dict] = None      # ScenarioState (mutable runtime state)
    metadata: Optional[dict] = None   # Error info, custom data

    # Validators to parse Unix timestamps from Redis
    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def parse_required_datetime(cls, v):
        return _unix_to_datetime(v) or _utc_now()

    @field_validator("completed_at", "next_retry_at", "heartbeat_at", mode="before")
    @classmethod
    def parse_optional_datetime(cls, v):
        return _unix_to_datetime(v)

    # Validators to parse JSON strings for list/dict fields from Redis
    @field_validator(
        "execution_order", "completed_steps", "context_doc_ids",
        "step_doc_ids", "scenario_config", "persona_config",
        "framework", "state", "metadata",
        mode="before"
    )
    @classmethod
    def parse_json_field(cls, v):
        """Parse JSON string from Redis or passthrough list/dict."""
        if v is None:
            return v
        if isinstance(v, str):
            if not v:
                return None
            return json.loads(v)
        return v

    # Serializers to output Unix timestamps
    @field_serializer("created_at", "updated_at")
    def serialize_required_datetime(self, dt: datetime) -> int:
        return _datetime_to_unix(dt)

    @field_serializer("completed_at", "next_retry_at", "heartbeat_at")
    def serialize_optional_datetime(self, dt: Optional[datetime]) -> Optional[int]:
        return _datetime_to_unix(dt)
