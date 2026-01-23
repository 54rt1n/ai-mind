# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Enumeration types for MUD events and actors."""

from enum import Enum


class EventType(str, Enum):
    """Type of event from the MUD world.

    These correspond to the different kinds of actions and occurrences
    that can happen in the MUD environment.

    Attributes:
        SPEECH: A character speaking (say command).
        EMOTE: A character performing an emote/action.
        NARRATIVE: Multi-paragraph prose narrative (act command).
        MOVEMENT: A character entering or leaving a room.
        OBJECT: An object being manipulated (get, drop, give).
        AMBIENT: Environmental or atmospheric events.
        NOTIFICATION: Interactive/environmental notifications (e.g., doorbells).
        TERMINAL: Terminal tool execution events (code, web, research, etc.).
        CODE_ACTION: Code command output or action (targeted to caller).
        CODE_FILE: Code file content output (targeted to caller).
        SYSTEM: System messages (login, logout, errors).
        NON_REACTIVE: Non-reactive events (e.g., internal state changes).
        NON_PUBLISHED: Non-published events (e.g., internal state changes).
        SLEEP_AWARE: Events that bypass sleep filtering (delivered even when sleeping).
    """

    SPEECH = "speech"
    EMOTE = "emote"
    NARRATIVE = "narrative"
    MOVEMENT = "movement"
    OBJECT = "object"
    AMBIENT = "ambient"
    NOTIFICATION = "notification"
    TERMINAL = "terminal"
    CODE_ACTION = "code-action"
    CODE_FILE = "code-file"
    SYSTEM = "system"
    NON_REACTIVE = "non_reactive"
    NON_PUBLISHED = "non_published"
    SLEEP_AWARE = "sleep_aware"


class ActorType(str, Enum):
    """Type of actor that caused an event.

    Distinguishes between different types of entities that can
    generate events in the MUD world.

    Attributes:
        PLAYER: A human player.
        AI: An AI-controlled character (via AIM).
        NPC: A non-player character (Evennia scripted).
        SYSTEM: The MUD system itself.
    """

    PLAYER = "player"
    AI = "ai"
    NPC = "npc"
    SYSTEM = "system"
