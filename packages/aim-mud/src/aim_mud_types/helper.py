# aim-mud-types/helper.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Helper functions for MUD types."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from .models.coordination import MUDTurnRequest, TurnRequestStatus


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


def _datetime_to_unix(dt: Any) -> Optional[int]:
    """Convert datetime to Unix timestamp (seconds since epoch).

    Handles datetime objects and strings (ISO format or Unix timestamp string).
    """
    if dt is None:
        return None
    if isinstance(dt, str):
        # Empty string means None
        if not dt:
            return None
        # If it's already a Unix timestamp string, convert directly
        try:
            return int(dt)
        except ValueError:
            # Otherwise parse as ISO format
            dt = datetime.fromisoformat(dt)
    return int(dt.timestamp())


def _unix_to_datetime(ts: Any) -> Optional[datetime]:
    """Convert Unix timestamp to datetime.

    Handles int, float, string, or datetime passthrough.
    Returns None for None, empty string, or "0".
    """
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        if not ts or ts == "0":
            return None
        # Try Unix timestamp first, then ISO format for backwards compatibility
        try:
            ts = int(ts)
        except ValueError:
            return datetime.fromisoformat(ts)
    if ts == 0:
        return None
    return datetime.fromtimestamp(int(ts), tz=timezone.utc)


def model_to_redis_hash(model: BaseModel) -> dict[str, str]:
    """Convert a Pydantic model to a Redis hash mapping.

    Uses model_dump(mode="json") to invoke field_serializers, then converts
    all values to strings suitable for Redis HSET.

    - None → ""
    - list/dict → JSON string
    - Other → str()

    Args:
        model: Pydantic model instance with proper field_serializers defined

    Returns:
        Dictionary with string keys and string values for Redis
    """
    dumped = model.model_dump(mode="json")
    data: dict[str, str] = {}
    for k, v in dumped.items():
        if v is None:
            data[k] = ""
        elif isinstance(v, (list, dict)):
            data[k] = json.dumps(v)
        else:
            data[k] = str(v)
    return data


def get_hash_field(data: dict, key: str) -> str:
    """Get a field from a Redis hash, handling bytes keys/values.

    Redis returns bytes by default. This helper checks both bytes and string
    keys and decodes the value if needed.

    Args:
        data: Redis hash data (from hgetall)
        key: Field name to retrieve

    Returns:
        String value or empty string if not found
    """
    val = data.get(key.encode()) or data.get(key)
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return val or ""


def format_time_ago(timestamp_str: str) -> str:
    """Format timestamp as human-readable 'Xs ago' string.

    Handles Unix timestamps and ISO 8601 format strings.

    Args:
        timestamp_str: Unix timestamp or ISO 8601 timestamp string

    Returns:
        Human-readable string like "5s ago", "2m 30s ago", "1h 15m ago", "3d 2h ago"
    """
    try:
        heartbeat = _unix_to_datetime(timestamp_str)
        if not heartbeat:
            return "unknown"

        now = datetime.now(heartbeat.tzinfo)
        delta = now - heartbeat

        # Handle negative deltas (future timestamps)
        if delta.total_seconds() < 0:
            return "in the future"

        total_seconds = int(delta.total_seconds())
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if days > 0:
            if hours > 0:
                return f"{days}d {hours}h ago"
            return f"{days}d ago"
        elif hours > 0:
            if minutes > 0:
                return f"{hours}h {minutes}m ago"
            return f"{hours}h ago"
        elif minutes > 0:
            if seconds > 0:
                return f"{minutes}m {seconds}s ago"
            return f"{minutes}m ago"
        else:
            return f"{seconds}s ago"
    except (ValueError, AttributeError):
        return "unknown"


def parse_stream_timestamp(stream_id: str) -> datetime:
    """Extract datetime from Redis stream ID.

    Redis stream IDs are formatted as "{timestamp_ms}-{sequence}".

    Args:
        stream_id: Redis stream ID (e.g., "1767834269949-0")

    Returns:
        Datetime object parsed from stream ID, or current time on error
    """
    try:
        timestamp_ms = int(stream_id.split("-")[0])
        return datetime.fromtimestamp(timestamp_ms / 1000)
    except (ValueError, IndexError, AttributeError):
        return datetime.now()


def compute_next_attempt_at(backoff_seconds: int) -> str:
    """Compute next_attempt_at as Unix timestamp string from backoff seconds.

    Used for retry scheduling in turn request state machine.

    Args:
        backoff_seconds: Number of seconds to delay next attempt

    Returns:
        Unix timestamp as string
    """
    return str(_datetime_to_unix(_utc_now() + timedelta(seconds=backoff_seconds)))


def is_agent_online(heartbeat_at: str, threshold_minutes: int = 5) -> bool:
    """Check if agent heartbeat is recent enough to be considered online.

    Args:
        heartbeat_at: Unix timestamp or ISO 8601 timestamp string
        threshold_minutes: Minutes before considering agent offline (default: 5)

    Returns:
        True if heartbeat is within threshold, False otherwise
    """
    try:
        heartbeat = _unix_to_datetime(heartbeat_at)
        if not heartbeat:
            return False
        # Use datetime.now with same tzinfo to handle both naive and aware datetimes
        delta = datetime.now(heartbeat.tzinfo) - heartbeat
        return delta < timedelta(minutes=threshold_minutes)
    except (ValueError, AttributeError):
        return False


def normalize_agent_id(name: str) -> str:
    """Normalize a display name to agent_id format.

    Converts "Lin Yu" -> "linyu", "Nova" -> "nova", etc.
    Removes spaces and converts to lowercase for Redis key lookups.

    Args:
        name: Display name or persona_id

    Returns:
        Normalized agent_id for Redis lookups
    """
    return name.strip().replace(" ", "").lower()


# -----------------------------------------------------------------------------
# Turn Request Transition Utilities
# -----------------------------------------------------------------------------
# These are pure domain model utilities that mutate MUDTurnRequest objects
# in-place without any Redis operations. They are used by the worker to
# prepare state transitions before persisting to Redis.

def touch_turn_request_heartbeat(turn_request: "MUDTurnRequest") -> None:
    """Update heartbeat_at to current UTC time.

    Mutates turn_request in-place. No Redis operations.

    Args:
        turn_request: MUDTurnRequest object to update
    """
    turn_request.heartbeat_at = _utc_now()


def touch_turn_request_completed(
    turn_request: "MUDTurnRequest",
    *,
    update_heartbeat: bool = True,
) -> None:
    """Update completed_at (and optionally heartbeat_at) to current UTC time.

    Mutates turn_request in-place. No Redis operations.

    Args:
        turn_request: MUDTurnRequest object to update
        update_heartbeat: Also update heartbeat_at (default: True)
    """
    now = _utc_now()
    turn_request.completed_at = now
    if update_heartbeat:
        turn_request.heartbeat_at = now


def transition_turn_request(
    turn_request: "MUDTurnRequest",
    *,
    status: Optional["TurnRequestStatus"] = None,
    message: Optional[str] = None,
    status_reason: Optional[str] = None,
    attempt_count: Optional[int] = None,
    next_attempt_at: Optional[str] = None,
    turn_id: Optional[str] = None,
    new_turn_id: bool = False,
    set_completed: bool = False,
    update_heartbeat: bool = False,
) -> "MUDTurnRequest":
    """Apply common turn_request field updates in one place.

    This is a pure utility function that mutates the turn_request in-place
    without any Redis operations. Use this to prepare state transitions
    before calling update_turn_request() to persist changes.

    Args:
        turn_request: MUDTurnRequest object to modify
        status: New status (optional)
        message: New message (optional)
        status_reason: New status reason (optional)
        attempt_count: New attempt count (optional)
        next_attempt_at: Next attempt timestamp (optional)
        turn_id: New turn_id (optional)
        new_turn_id: Generate new turn_id via uuid4 (default: False)
        set_completed: Set completed_at timestamp (default: False)
        update_heartbeat: Update heartbeat_at timestamp (default: False)

    Returns:
        The modified turn_request (same object, mutated in-place)
    """
    import uuid

    if new_turn_id:
        turn_request.turn_id = str(uuid.uuid4())
    if turn_id is not None:
        turn_request.turn_id = turn_id
    if status is not None:
        turn_request.status = status
    if message is not None:
        turn_request.message = message
    if status_reason is not None:
        turn_request.status_reason = status_reason
    if attempt_count is not None:
        turn_request.attempt_count = attempt_count
    if next_attempt_at is not None:
        turn_request.next_attempt_at = next_attempt_at

    if set_completed:
        touch_turn_request_completed(turn_request, update_heartbeat=update_heartbeat)
    elif update_heartbeat:
        touch_turn_request_heartbeat(turn_request)

    return turn_request


# -----------------------------------------------------------------------------
# Dream Command Mapping
# -----------------------------------------------------------------------------

# Map command names to scenario names for dream pipeline execution
COMMAND_TO_SCENARIO = {
    "analyze": "analysis_dialogue",
    "summary": "summarizer",
    "journal": "journaler_dialogue",
    "ponder": "philosopher_dialogue",
    "daydream": "daydream_dialogue",
    "critique": "critique_dialogue",
    "research": "researcher_dialogue",
}


def create_pending_dream_stub(
    redis_client,
    agent_id: str,
    scenario_name: str,
    conversation_id: Optional[str] = None,
    query: Optional[str] = None,
    guidance: Optional[str] = None,
) -> str:
    """Create PENDING dream stub for manual command.

    This function creates a DreamingState stub in Redis with PENDING status,
    to be picked up and executed by the worker's dream pipeline.

    Args:
        redis_client: Redis client (sync or async)
        agent_id: Agent ID
        scenario_name: Scenario to execute (e.g., "analysis_dialogue", "summarizer")
        conversation_id: Optional conversation to analyze
        query: Optional query for creative commands
        guidance: Optional guidance for creative commands

    Returns:
        pipeline_id of created stub
    """
    import uuid

    from andimud_worker.conversation.storage import generate_conversation_id

    from .models.coordination import DreamingState, DreamStatus
    from .redis_keys import RedisKeys

    # Generate conversation_id for creative scenarios if not provided
    if not conversation_id:
        prefix = scenario_name.split("_")[0]  # "journaler" from "journaler_dialogue"
        conversation_id = generate_conversation_id(prefix)

    stub = DreamingState(
        pipeline_id=str(uuid.uuid4()),
        agent_id=agent_id,
        status=DreamStatus.PENDING,
        scenario_name=scenario_name,
        conversation_id=conversation_id,
        query=query,
        guidance=guidance,
        created_at=_utc_now(),
        updated_at=_utc_now(),
        # Worker will populate: execution_order, base_model, etc.
        execution_order=[],
        base_model="",
        step_index=0,
        completed_steps=[],
        step_doc_ids={},
        context_doc_ids=[],
        current_step_attempts=0,
        scenario_config={},
        persona_config={},
    )

    # Serialize and save
    stub_data = stub.model_dump(mode='json')

    # JSON-encode complex types
    for field in ['execution_order', 'completed_steps', 'context_doc_ids',
                  'step_doc_ids', 'scenario_config', 'persona_config']:
        if field in stub_data:
            stub_data[field] = json.dumps(stub_data[field])

    # Filter None values
    stub_data = {k: v for k, v in stub_data.items() if v is not None}

    # Save to Redis (works with both sync and async)
    key = RedisKeys.agent_dreaming_state(agent_id)
    redis_client.hset(key, mapping=stub_data)

    return stub.pipeline_id
