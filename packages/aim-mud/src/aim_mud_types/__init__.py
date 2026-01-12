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
"""

from .actions import MUDAction
from .conversation import MUDConversationEntry
from .coordination import MUDTurnRequest, DreamerState, TurnRequestStatus, TurnReason
from .enums import EventType, ActorType
from .events import MUDEvent
from .plan import AgentPlan, PlanTask, PlanStatus, TaskStatus
from .profile import AgentProfile, RoomProfile
from .redis_keys import RedisKeys
from .session import MUDTurn, MUDSession
from .state import RoomState, EntityState
from .world_state import WorldState, InventoryItem, WhoEntry
from .client import RedisMUDClient
from .turn_request_helpers import (
    assign_turn_request,
    assign_turn_request_async,
    update_turn_request,
    update_turn_request_async,
    initialize_turn_request,
    initialize_turn_request_async,
    atomic_heartbeat_update,
    atomic_heartbeat_update_async,
    delete_turn_request,
    delete_turn_request_async,
    touch_turn_request_heartbeat,
    touch_turn_request_completed,
    compute_next_attempt_at,
    transition_turn_request,
    transition_turn_request_and_update,
    transition_turn_request_and_update_async,
)
from .agent_profile_helpers import (
    get_agent_profile_raw,
    update_agent_profile_fields,
    set_agent_profile_fields,
)
from .room_profile_helpers import (
    get_room_profile_raw,
    update_room_profile_fields,
    set_room_profile_fields,
)
from .conversation_helpers import (
    get_conversation_entries,
    get_conversation_entry,
    append_conversation_entry,
    set_conversation_entry,
    pop_conversation_entry,
    get_conversation_length,
    delete_conversation,
    replace_conversation_entries,
)
from .conversation_report_helpers import (
    get_conversation_report,
    set_conversation_report,
    get_conversation_report_async,
    set_conversation_report_async,
)
from .pause_helpers import (
    is_paused,
    is_paused_async,
    is_agent_paused,
    is_agent_paused_async,
    is_mediator_paused,
    is_mediator_paused_async,
)
from .sequence_helpers import (
    next_sequence_id,
    next_sequence_id_async,
)
from .dreamer_state_helpers import (
    get_dreamer_state,
    update_dreamer_state_fields,
)
from .mud_events_helpers import (
    get_mud_events_last_id,
    read_mud_events,
    append_mud_event,
    trim_mud_events_minid,
    is_mud_event_processed,
    mark_mud_event_processed,
    get_mud_event_processed_ids,
    get_min_processed_mud_event_id,
    get_max_processed_mud_event_id,
    trim_processed_mud_event_ids,
)
from .agent_events_helpers import (
    get_agent_events_last_id,
    range_agent_events,
    range_agent_events_reverse,
    append_agent_event,
    get_agent_events_length,
)
from .mud_actions_helpers import (
    read_mud_actions,
    append_mud_action,
    range_mud_actions_reverse,
    get_mud_actions_length,
    trim_mud_actions_maxlen,
    is_mud_action_processed,
    mark_mud_action_processed,
    get_mud_action_processed_ids,
    get_max_processed_mud_action_id,
    trim_processed_mud_action_ids,
)
__all__ = [
    # Enums
    "EventType",
    "ActorType",
    "TurnRequestStatus",
    "TurnReason",
    "PlanStatus",
    "TaskStatus",
    # State
    "RoomState",
    "EntityState",
    "WorldState",
    "InventoryItem",
    "WhoEntry",
    # Conversation
    "MUDConversationEntry",
    # Session
    "MUDTurn",
    "MUDSession",
    # Events and Actions
    "MUDEvent",
    "MUDAction",
    # Coordination
    "MUDTurnRequest",
    "DreamerState",
    # Plans
    "AgentPlan",
    "PlanTask",
    # Profiles
    "AgentProfile",
    "RoomProfile",
    # Redis Keys
    "RedisKeys",
    # Redis Client
    "RedisMUDClient",
    # Turn request helpers
    "assign_turn_request",
    "assign_turn_request_async",
    "update_turn_request",
    "update_turn_request_async",
    "initialize_turn_request",
    "initialize_turn_request_async",
    "atomic_heartbeat_update",
    "atomic_heartbeat_update_async",
    "delete_turn_request",
    "delete_turn_request_async",
    "touch_turn_request_heartbeat",
    "touch_turn_request_completed",
    "compute_next_attempt_at",
    "transition_turn_request",
    "transition_turn_request_and_update",
    "transition_turn_request_and_update_async",
    # Agent/room profile helpers (sync)
    "get_agent_profile_raw",
    "update_agent_profile_fields",
    "set_agent_profile_fields",
    "get_room_profile_raw",
    "update_room_profile_fields",
    "set_room_profile_fields",
    # Conversation helpers (sync)
    "get_conversation_entries",
    "get_conversation_entry",
    "append_conversation_entry",
    "set_conversation_entry",
    "pop_conversation_entry",
    "get_conversation_length",
    "delete_conversation",
    "replace_conversation_entries",
    # Conversation report helpers (sync/async)
    "get_conversation_report",
    "set_conversation_report",
    "get_conversation_report_async",
    "set_conversation_report_async",
    # Pause helpers (sync/async)
    "is_paused",
    "is_paused_async",
    "is_agent_paused",
    "is_agent_paused_async",
    "is_mediator_paused",
    "is_mediator_paused_async",
    # Sequence helpers (sync/async)
    "next_sequence_id",
    "next_sequence_id_async",
    # Dreamer state helpers (sync)
    "get_dreamer_state",
    "update_dreamer_state_fields",
    # Stream helpers: mud events
    "get_mud_events_last_id",
    "read_mud_events",
    "append_mud_event",
    "trim_mud_events_minid",
    "is_mud_event_processed",
    "mark_mud_event_processed",
    "get_mud_event_processed_ids",
    "get_min_processed_mud_event_id",
    "get_max_processed_mud_event_id",
    "trim_processed_mud_event_ids",
    # Stream helpers: agent events
    "get_agent_events_last_id",
    "range_agent_events",
    "range_agent_events_reverse",
    "append_agent_event",
    "get_agent_events_length",
    # Stream helpers: mud actions
    "read_mud_actions",
    "append_mud_action",
    "range_mud_actions_reverse",
    "get_mud_actions_length",
    "trim_mud_actions_maxlen",
    "is_mud_action_processed",
    "mark_mud_action_processed",
    "get_mud_action_processed_ids",
    "get_max_processed_mud_action_id",
    "trim_processed_mud_action_ids",
]

__version__ = "0.1.0"
