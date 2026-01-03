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

from .enums import EventType, ActorType
from .state import RoomState, EntityState
from .world_state import WorldState, InventoryItem, WhoEntry
from .events import MUDEvent
from .actions import MUDAction
from .redis_keys import RedisKeys

__all__ = [
    # Enums
    "EventType",
    "ActorType",
    # State
    "RoomState",
    "EntityState",
    "WorldState",
    "InventoryItem",
    "WhoEntry",
    # Events and Actions
    "MUDEvent",
    "MUDAction",
    # Utilities
    "RedisKeys",
]

__version__ = "0.1.0"
