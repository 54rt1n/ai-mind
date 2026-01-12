# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Helpers for turn request assignment and updates."""

from __future__ import annotations

import json
import uuid
from datetime import timedelta
from typing import Optional, Tuple

from .coordination import MUDTurnRequest, TurnReason, TurnRequestStatus
from .helper import _utc_now
from .redis_keys import RedisKeys
from .client import RedisMUDClient


def _decode_hash(raw: dict) -> dict[str, str]:
    decoded: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, bytes):
            k = k.decode("utf-8")
        if isinstance(v, bytes):
            v = v.decode("utf-8")
        decoded[str(k)] = str(v)
    return decoded


def touch_turn_request_heartbeat(turn_request: MUDTurnRequest) -> None:
    """Update heartbeat_at to current UTC time."""
    turn_request.heartbeat_at = _utc_now()


def touch_turn_request_completed(
    turn_request: MUDTurnRequest,
    *,
    update_heartbeat: bool = True,
) -> None:
    """Update completed_at (and optionally heartbeat_at) to current UTC time."""
    now = _utc_now()
    turn_request.completed_at = now
    if update_heartbeat:
        turn_request.heartbeat_at = now


def compute_next_attempt_at(backoff_seconds: int) -> str:
    """Compute next_attempt_at ISO timestamp from backoff seconds."""
    return (_utc_now() + timedelta(seconds=backoff_seconds)).isoformat()


def transition_turn_request(
    turn_request: MUDTurnRequest,
    *,
    status: Optional[TurnRequestStatus] = None,
    message: Optional[str] = None,
    status_reason: Optional[str] = None,
    attempt_count: Optional[int] = None,
    next_attempt_at: Optional[str] = None,
    turn_id: Optional[str] = None,
    new_turn_id: bool = False,
    set_completed: bool = False,
    update_heartbeat: bool = False,
) -> MUDTurnRequest:
    """Apply common turn_request field updates in one place."""
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


def transition_turn_request_and_update(
    redis_client,
    agent_id: str,
    turn_request: MUDTurnRequest,
    expected_turn_id: str,
    **kwargs,
) -> bool:
    """Apply a transition and persist with CAS (sync)."""
    transition_turn_request(turn_request, **kwargs)
    return update_turn_request(
        redis_client,
        agent_id,
        turn_request,
        expected_turn_id=expected_turn_id,
    )


async def transition_turn_request_and_update_async(
    redis_client,
    agent_id: str,
    turn_request: MUDTurnRequest,
    expected_turn_id: str,
    **kwargs,
) -> bool:
    """Apply a transition and persist with CAS (async)."""
    transition_turn_request(turn_request, **kwargs)
    return await update_turn_request_async(
        redis_client,
        agent_id,
        turn_request,
        expected_turn_id=expected_turn_id,
    )


def assign_turn_request(
    redis_client,
    agent_id: str,
    reason: str | TurnReason,
    *,
    deadline_ms: int = 120_000,
    turn_id_prefix: str = "manual",
    status: Optional[TurnRequestStatus] = None,
    **metadata_kwargs,
) -> Tuple[bool, str]:
    """Assign a new turn_request if the agent is available.

    Returns:
        (True, turn_id) on success
        (False, reason) on failure
    """
    key = RedisKeys.agent_turn_request(agent_id)

    pause_key = RedisKeys.agent_pause(agent_id)
    paused = redis_client.get(pause_key)
    if paused in (b"1", "1"):
        return False, "paused"

    current = redis_client.hgetall(key) or {}
    if not current:
        return False, "offline"

    decoded = _decode_hash(current)
    try:
        turn_request = MUDTurnRequest.model_validate(decoded)
    except Exception:
        return False, "invalid"

    if not turn_request.is_available_for_assignment():
        return False, f"busy:{turn_request.status.value}"

    now = _utc_now()
    turn_reason = reason if isinstance(reason, TurnReason) else TurnReason(reason)
    if status is None:
        status = TurnRequestStatus.EXECUTE if turn_reason.is_immediate_command() else TurnRequestStatus.ASSIGNED
    turn_id = f"{turn_id_prefix}_{int(now.timestamp() * 1000)}"
    sequence_id = redis_client.incr(RedisKeys.SEQUENCE_COUNTER)

    payload = {
        "turn_id": turn_id,
        "status": status.value,
        "reason": turn_reason.value,
        "assigned_at": now.isoformat(),
        "heartbeat_at": now.isoformat(),
        "deadline_ms": str(deadline_ms),
        "sequence_id": str(sequence_id),
        "attempt_count": "0",
    }

    metadata = {k: v for k, v in metadata_kwargs.items() if v is not None}
    if metadata:
        payload["metadata"] = json.dumps(metadata)

    redis_client.hset(key, mapping=payload)

    return True, turn_id


def initialize_turn_request(
    redis_client,
    agent_id: str,
    *,
    status: TurnRequestStatus = TurnRequestStatus.READY,
    reason: TurnReason = TurnReason.EVENTS,
    status_reason: Optional[str] = None,
    attempt_count: int = 0,
    deadline_ms: int = 0,
    sequence_id: Optional[int] = None,
    **metadata_kwargs,
) -> Tuple[bool, Optional[MUDTurnRequest], str]:
    """Initialize agent turn_request if missing (sync).

    Returns:
        (True, turn_request, "created") on success
        (False, existing_turn_request, "exists") if already present
        (False, None, "invalid") if existing hash cannot be parsed
    """
    key = RedisKeys.agent_turn_request(agent_id)
    current = redis_client.hgetall(key) or {}
    if current:
        decoded = _decode_hash(current)
        try:
            turn_request = MUDTurnRequest.model_validate(decoded)
        except Exception:
            return False, None, "invalid"
        return False, turn_request, "exists"

    if sequence_id is None:
        sequence_id = redis_client.incr(RedisKeys.SEQUENCE_COUNTER)

    now = _utc_now()
    turn_request = MUDTurnRequest(
        turn_id=str(uuid.uuid4()),
        sequence_id=sequence_id,
        status=status,
        reason=reason,
        assigned_at=now,
        heartbeat_at=now,
        completed_at=None,
        deadline_ms=str(deadline_ms),
        attempt_count=attempt_count,
        status_reason=status_reason,
        metadata={k: v for k, v in metadata_kwargs.items() if v is not None} or None,
    )

    fields = {}
    for field_name, field_value in turn_request.model_dump().items():
        if field_value is None:
            continue
        if hasattr(field_value, "isoformat"):
            fields[field_name] = field_value.isoformat()
        elif isinstance(field_value, (TurnRequestStatus, TurnReason)):
            fields[field_name] = field_value.value
        elif isinstance(field_value, dict):
            fields[field_name] = json.dumps(field_value)
        else:
            fields[field_name] = str(field_value)

    lua_script = """
        local key = KEYS[1]

        if redis.call('EXISTS', key) == 1 then
            return 0
        end

        for i = 1, #ARGV, 2 do
            redis.call('HSET', key, ARGV[i], ARGV[i+1])
        end

        return 1
    """

    args = []
    for field_name, value in fields.items():
        args.extend([field_name, value])

    result = redis_client.eval(lua_script, 1, key, *args)
    if result == 1:
        return True, turn_request, "created"

    # Race: someone else created it
    current = redis_client.hgetall(key) or {}
    if current:
        decoded = _decode_hash(current)
        try:
            turn_request = MUDTurnRequest.model_validate(decoded)
        except Exception:
            return False, None, "invalid"
        return False, turn_request, "exists"
    return False, None, "exists"


def update_turn_request(
    redis_client,
    agent_id: str,
    turn_request: MUDTurnRequest,
    expected_turn_id: str,
) -> bool:
    """Update turn_request with CAS (sync)."""
    key = RedisKeys.agent_turn_request(agent_id)

    def _serialize_value(value) -> Optional[str]:
        if value is None:
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if isinstance(value, (TurnRequestStatus, TurnReason)):
            return value.value
        if isinstance(value, dict):
            return json.dumps(value)
        return str(value)

    fields = []
    for field_name, field_value in turn_request.model_dump().items():
        serialized = _serialize_value(field_value)
        if serialized is not None:
            fields.extend([field_name, serialized])

    lua_script = """
        local key = KEYS[1]
        local expected_turn_id = ARGV[1]

        local current = redis.call('HGET', key, 'turn_id')
        if current ~= expected_turn_id then
            return 0
        end

        for i = 2, #ARGV, 2 do
            redis.call('HSET', key, ARGV[i], ARGV[i+1])
        end

        return 1
    """

    result = redis_client.eval(lua_script, 1, key, expected_turn_id, *fields)
    return result == 1


async def update_turn_request_async(
    redis_client,
    agent_id: str,
    turn_request: MUDTurnRequest,
    expected_turn_id: str,
) -> bool:
    """Update turn_request with CAS (async)."""
    client = RedisMUDClient(redis_client)
    success = await client.update_turn_request(agent_id, turn_request, expected_turn_id=expected_turn_id)
    return success


async def assign_turn_request_async(
    redis_client,
    agent_id: str,
    reason: str | TurnReason,
    *,
    attempt_count: int = 0,
    deadline_ms: int = 0,
    status: Optional[TurnRequestStatus] = None,
    expected_turn_id: Optional[str] = None,
    skip_availability_check: bool = False,
    **metadata_kwargs,
) -> Tuple[bool, Optional[MUDTurnRequest], str]:
    """Assign a new turn_request with CAS (async).

    Returns:
        (True, turn_request, "ok") on success
        (False, None, reason) on failure
    """
    pause_key = RedisKeys.agent_pause(agent_id)
    paused = await redis_client.get(pause_key)
    if paused in (b"1", "1"):
        return False, None, "paused"

    current = None
    if not skip_availability_check or expected_turn_id is None:
        client = RedisMUDClient(redis_client)
        current = await client.get_turn_request(agent_id)
        if not current:
            return False, None, "offline"
        if not skip_availability_check and not current.is_available_for_assignment():
            return False, None, f"busy:{current.status.value}"
        if expected_turn_id is None:
            expected_turn_id = current.turn_id

    if expected_turn_id is None:
        return False, None, "missing_expected_turn_id"

    reason_enum = reason if isinstance(reason, TurnReason) else TurnReason(reason)
    if status is None:
        status = TurnRequestStatus.EXECUTE if reason_enum.is_immediate_command() else TurnRequestStatus.ASSIGNED

    sequence_id = await redis_client.incr(RedisKeys.SEQUENCE_COUNTER)
    now = _utc_now()
    turn_request = MUDTurnRequest(
        turn_id=str(uuid.uuid4()),
        sequence_id=sequence_id,
        status=status,
        reason=reason_enum,
        assigned_at=now,
        heartbeat_at=now,
        completed_at=None,
        deadline_ms=str(deadline_ms),
        attempt_count=attempt_count,
        metadata={k: v for k, v in metadata_kwargs.items() if v is not None} or None,
    )

    success = await update_turn_request_async(
        redis_client,
        agent_id,
        turn_request,
        expected_turn_id=expected_turn_id,
    )
    if not success:
        return False, None, "cas_failed"

    return True, turn_request, "ok"


def atomic_heartbeat_update(
    redis_client,
    agent_id: str,
) -> int:
    """Atomically update heartbeat with validation (sync).

    Returns:
        1: Success
        0: Key doesn't exist
        -1: Corrupted hash (missing required fields)
    """
    key = RedisKeys.agent_turn_request(agent_id)
    lua_script = """
        local key = KEYS[1]
        local heartbeat_at = ARGV[1]

        if redis.call('EXISTS', key) == 0 then
            return 0
        end

        local status = redis.call('HGET', key, 'status')
        local turn_id = redis.call('HGET', key, 'turn_id')

        if not status or not turn_id then
            return -1
        end

        redis.call('HSET', key, 'heartbeat_at', heartbeat_at)

        return 1
    """

    result = redis_client.eval(
        lua_script,
        1,
        key,
        _utc_now().isoformat(),
    )
    return int(result)


async def atomic_heartbeat_update_async(
    redis_client,
    agent_id: str,
) -> int:
    """Atomically update heartbeat with validation (async).

    Returns:
        1: Success
        0: Key doesn't exist
        -1: Corrupted hash (missing required fields)
    """
    key = RedisKeys.agent_turn_request(agent_id)
    lua_script = """
        local key = KEYS[1]
        local heartbeat_at = ARGV[1]

        if redis.call('EXISTS', key) == 0 then
            return 0
        end

        local status = redis.call('HGET', key, 'status')
        local turn_id = redis.call('HGET', key, 'turn_id')

        if not status or not turn_id then
            return -1
        end

        redis.call('HSET', key, 'heartbeat_at', heartbeat_at)

        return 1
    """

    result = await redis_client.eval(
        lua_script,
        1,
        key,
        _utc_now().isoformat(),
    )
    return int(result)


async def initialize_turn_request_async(
    redis_client,
    agent_id: str,
    *,
    status: TurnRequestStatus = TurnRequestStatus.READY,
    reason: TurnReason = TurnReason.EVENTS,
    status_reason: Optional[str] = None,
    attempt_count: int = 0,
    deadline_ms: int = 0,
    sequence_id: Optional[int] = None,
    **metadata_kwargs,
) -> Tuple[bool, Optional[MUDTurnRequest], str]:
    """Initialize agent turn_request if missing (async).

    Returns:
        (True, turn_request, "created") on success
        (False, existing_turn_request, "exists") if already present
        (False, None, "invalid") if existing hash cannot be parsed
    """
    key = RedisKeys.agent_turn_request(agent_id)
    current = await redis_client.hgetall(key) or {}
    if current:
        decoded = _decode_hash(current)
        try:
            turn_request = MUDTurnRequest.model_validate(decoded)
        except Exception:
            return False, None, "invalid"
        return False, turn_request, "exists"

    if sequence_id is None:
        sequence_id = await redis_client.incr(RedisKeys.SEQUENCE_COUNTER)

    now = _utc_now()
    turn_request = MUDTurnRequest(
        turn_id=str(uuid.uuid4()),
        sequence_id=sequence_id,
        status=status,
        reason=reason,
        assigned_at=now,
        heartbeat_at=now,
        completed_at=None,
        deadline_ms=str(deadline_ms),
        attempt_count=attempt_count,
        status_reason=status_reason,
        metadata={k: v for k, v in metadata_kwargs.items() if v is not None} or None,
    )

    client = RedisMUDClient(redis_client)
    created = await client.create_turn_request(agent_id, turn_request)
    if created:
        return True, turn_request, "created"

    current = await redis_client.hgetall(key) or {}
    if current:
        decoded = _decode_hash(current)
        try:
            turn_request = MUDTurnRequest.model_validate(decoded)
        except Exception:
            return False, None, "invalid"
        return False, turn_request, "exists"
    return False, None, "exists"


def delete_turn_request(redis_client, agent_id: str) -> bool:
    """Delete agent turn_request (sync)."""
    key = RedisKeys.agent_turn_request(agent_id)
    deleted = redis_client.delete(key)
    return bool(deleted)


async def delete_turn_request_async(redis_client, agent_id: str) -> bool:
    """Delete agent turn_request (async)."""
    key = RedisKeys.agent_turn_request(agent_id)
    deleted = await redis_client.delete(key)
    return bool(deleted)
