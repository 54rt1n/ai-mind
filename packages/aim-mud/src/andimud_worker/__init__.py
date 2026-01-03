# aim/app/mud/__init__.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""MUD integration module for AI-Mind agents.

This module provides the infrastructure for AI agents to exist in a
persistent multi-user dungeon (MUD) environment. Agents receive world
events, reason about them, and emit actions back to the world.

Architecture follows the Dreamer worker pattern - agents consume events
from per-agent Redis streams (push model) rather than polling.
"""

from .adapter import (
    MAX_RECENT_TURNS,
    build_system_prompt,
    build_current_context,
    format_event,
    format_turn_events,
    format_turn_response,
)
from .config import MUDConfig
from .session import (
    EventType,
    ActorType,
    MUDEvent,
    MUDAction,
    RoomState,
    EntityState,
    MUDTurn,
    MUDSession,
)
from .worker import MUDAgentWorker, run_worker

# Mediator is in a separate package now
try:
    from andimud_mediator.mediator import MediatorService, MediatorConfig, run_mediator
except ImportError:
    # Optional import - mediator may not be installed
    MediatorService = None
    MediatorConfig = None
    run_mediator = None

__all__ = [
    # Adapter
    "MAX_RECENT_TURNS",
    "build_system_prompt",
    "build_current_context",
    "format_event",
    "format_turn_events",
    "format_turn_response",
    # Config
    "MUDConfig",
    "MediatorConfig",
    # Session types
    "EventType",
    "ActorType",
    "MUDEvent",
    "MUDAction",
    "RoomState",
    "EntityState",
    "MUDTurn",
    "MUDSession",
    # Worker
    "MUDAgentWorker",
    "run_worker",
    # Mediator
    "MediatorService",
    "run_mediator",
]
