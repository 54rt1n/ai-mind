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
        SYSTEM: System messages (login, logout, errors).
    """

    SPEECH = "speech"
    EMOTE = "emote"
    NARRATIVE = "narrative"
    MOVEMENT = "movement"
    OBJECT = "object"
    AMBIENT = "ambient"
    NOTIFICATION = "notification"
    SYSTEM = "system"


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
