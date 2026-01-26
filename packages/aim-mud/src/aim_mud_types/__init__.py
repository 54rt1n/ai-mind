# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Shared types for AI-Mind and Evennia MUD integration.

This package provides the common data types and utilities used by both
the AI-Mind agent workers and the Evennia MUD server for event-driven
communication via Redis streams.

Architecture:
    Evennia -> mud:events -> Mediator -> agent:{id}:events -> AIM Worker
    AIM Worker -> mud:actions -> Mediator -> Evennia

All events and actions flow through Redis streams using these shared types.

Usage:
    # For Redis operations, use the typed clients:
    from aim_mud_types.client import AsyncRedisMUDClient, SyncRedisMUDClient

    # Async context (workers, mediator)
    client = AsyncRedisMUDClient(redis_client)
    turn_request = await client.get_turn_request("andi")
    await client.update_turn_request("andi", turn_request, expected_turn_id)

    # Sync context (Evennia commands)
    client = SyncRedisMUDClient(redis_client)
    profile = client.get_agent_profile_raw("andi")
    client.set_room_profile_fields("room123", {"field": "value"})

    # Pure utilities (no Redis)
    from aim_mud_types.helper import normalize_agent_id, transition_turn_request
"""

# Pydantic models and enums - imported from models subpackage
from .models import (
    # Enums
    EventType,
    ActorType,
    TurnRequestStatus,
    TurnReason,
    DreamStatus,
    PlanStatus,
    TaskStatus,
    DecisionType,
    # Models
    AuraState,
    RoomState,
    EntityState,
    WorldState,
    InventoryItem,
    WhoEntry,
    MUDAction,
    MUDConversationEntry,
    AgentProfile,
    RoomProfile,
    MUDTurnRequest,
    DreamerState,
    DreamingState,
    ThoughtState,
    THOUGHT_THROTTLE_SECONDS,
    THOUGHT_THROTTLE_ACTIONS,
    MUDEvent,
    AgentPlan,
    PlanTask,
    MUDTurn,
    MUDSession,
    DecisionResult,
)

# Constants
from .constants import (
    AURA_RINGABLE,
    AURA_WEB_ACCESS,
    AURA_CODE_ACCESS,
    AURA_MARKET_ACCESS,
    AURA_NEWS_ACCESS,
    AURA_RESEARCH_ACCESS,
    AURA_LIST_ACCESS,
    # Paper system auras
    AURA_PRINT_ACCESS,
    AURA_SCAN_ACCESS,
    AURA_BIND_ACCESS,
    AURA_BOOK_ACCESS,
    AURA_COPY_ACCESS,
    AURA_PAPER_WRITE,
    AURA_CODE_ROOM,
    AURA_SLEEPABLE,
)

# Redis Keys
from .redis_keys import RedisKeys

# Redis Clients (primary API for Redis operations)
from .client import (
    AsyncRedisMUDClient,
    SyncRedisMUDClient,
    # Backwards compatibility aliases
    RedisMUDClient,
    BaseRedisMUDClient,
)

# Pure utility functions (no Redis dependency)
from .helper import (
    # Time utilities
    _utc_now,
    _datetime_to_unix,
    _unix_to_datetime,
    # Redis hash utilities
    model_to_redis_hash,
    get_hash_field,
    # Formatting utilities
    format_time_ago,
    parse_stream_timestamp,
    # Turn request utilities (pure domain functions)
    compute_next_attempt_at,
    is_agent_online,
    normalize_agent_id,
    transition_turn_request,
    touch_turn_request_heartbeat,
    touch_turn_request_completed,
    # Dream command mapping
    COMMAND_TO_SCENARIO,
    create_pending_dream_stub,
)


__all__ = [
    # Pydantic Models
    "MUDAction",
    "MUDConversationEntry",
    "MUDTurnRequest",
    "DreamerState",
    "DreamingState",
    "ThoughtState",
    "THOUGHT_THROTTLE_SECONDS",
    "THOUGHT_THROTTLE_ACTIONS",
    "DreamStatus",
    "MUDEvent",
    "AgentPlan",
    "PlanTask",
    "AgentProfile",
    "RoomProfile",
    "MUDTurn",
    "MUDSession",
    "AuraState",
    "RoomState",
    "EntityState",
    "WorldState",
    "InventoryItem",
    "WhoEntry",
    "DecisionType",
    "DecisionResult",
    # Enums
    "EventType",
    "ActorType",
    "TurnRequestStatus",
    "TurnReason",
    "PlanStatus",
    "TaskStatus",
    # Constants
    "AURA_RINGABLE",
    "AURA_WEB_ACCESS",
    "AURA_CODE_ACCESS",
    "AURA_MARKET_ACCESS",
    "AURA_NEWS_ACCESS",
    "AURA_RESEARCH_ACCESS",
    "AURA_LIST_ACCESS",
    "AURA_PRINT_ACCESS",
    "AURA_SCAN_ACCESS",
    "AURA_BIND_ACCESS",
    "AURA_BOOK_ACCESS",
    "AURA_COPY_ACCESS",
    "AURA_PAPER_WRITE",
    "AURA_CODE_ROOM",
    "AURA_SLEEPABLE",
    # Redis Keys
    "RedisKeys",
    # Redis Clients
    "AsyncRedisMUDClient",
    "SyncRedisMUDClient",
    "RedisMUDClient",  # Backwards compatibility alias for AsyncRedisMUDClient
    "BaseRedisMUDClient",  # Backwards compatibility alias
    # Pure Utilities
    "_utc_now",
    "_datetime_to_unix",
    "_unix_to_datetime",
    "model_to_redis_hash",
    "get_hash_field",
    "format_time_ago",
    "parse_stream_timestamp",
    "compute_next_attempt_at",
    "is_agent_online",
    "normalize_agent_id",
    "transition_turn_request",
    "touch_turn_request_heartbeat",
    "touch_turn_request_completed",
    "COMMAND_TO_SCENARIO",
    "create_pending_dream_stub",
]

__version__ = "0.1.0"
