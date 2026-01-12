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
]

__version__ = "0.1.0"
