# aim/app/mud/worker/turns/validation.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Validation helpers for MUD worker turn processing.

Pure functions for validating actions and resolving game state references.
"""

from typing import Optional
from aim_mud_types import MUDSession


def resolve_target_name(session: Optional[MUDSession], target_id: str) -> str:
    """Resolve a target ID to a display name for emotes.

    Args:
        session: Current MUD session, may be None
        target_id: The ID of the target entity

    Returns:
        Display name of the target, or fallback to target_id
    """
    if not target_id or not session:
        return target_id or "object"

    world_state = session.world_state
    if world_state and world_state.room_state:
        room = world_state.room_state
        if room.room_id == target_id:
            return room.name or "room"

    # Check entities present
    if world_state:
        for entity in world_state.entities_present:
            if entity.entity_id == target_id:
                return entity.name or target_id

        for item in world_state.inventory:
            if getattr(item, "item_id", None) == target_id:
                return item.name or target_id

    return target_id


def resolve_move_location(session: Optional[MUDSession], location: Optional[str]) -> Optional[str]:
    """Validate and normalize a move location against current room exits.

    Performs case-insensitive matching against the current room's exit names.

    Args:
        session: Current MUD session, may be None
        location: The location/direction to move to

    Returns:
        Canonical exit name if valid, None if invalid or no session
    """
    if not location:
        return None

    room = session.current_room if session else None
    exits = room.exits if room else None
    if not exits:
        return location

    if location in exits:
        return location

    lowered = location.lower()
    for exit_name in exits.keys():
        if exit_name.lower() == lowered:
            return exit_name

    return None


def is_superuser_persona(persona) -> bool:
    """Check if persona has builder/superuser permissions.

    Args:
        persona: The persona object to check

    Returns:
        True if persona has superuser or builder role/permission
    """
    if not persona:
        return False

    attrs = persona.attributes or {}
    role = str(attrs.get("mud_role", "")).lower()
    perms = attrs.get("mud_permissions")

    if role in ("superuser", "builder"):
        return True

    if isinstance(perms, list):
        return any(str(p).lower() in ("superuser", "builder") for p in perms)

    if isinstance(perms, str):
        return perms.lower() in ("superuser", "builder")

    return False


def get_room_objects(session: Optional[MUDSession]) -> list[str]:
    """Get names of objects available to take in the current room.

    Returns entities that are not players, AIs, or NPCs (i.e., takeable objects).

    Args:
        session: Current MUD session, may be None

    Returns:
        List of object names in the room
    """
    objects: list[str] = []
    world_state = session.world_state if session else None
    if world_state:
        for entity in world_state.entities_present:
            if entity.is_self:
                continue
            # Objects are entities that aren't players/AIs/NPCs
            if entity.entity_type not in ("player", "ai", "npc"):
                if entity.name:
                    objects.append(entity.name)
    else:
        # Fall back to session entities
        if session:
            for entity in session.entities_present:
                if entity.is_self:
                    continue
                if entity.entity_type not in ("player", "ai", "npc"):
                    if entity.name:
                        objects.append(entity.name)
    return objects


def get_inventory_items(session: Optional[MUDSession]) -> list[str]:
    """Get names of items in the agent's inventory.

    Args:
        session: Current MUD session, may be None

    Returns:
        List of item names in inventory
    """
    items: list[str] = []
    world_state = session.world_state if session else None
    if world_state:
        for item in world_state.inventory:
            if item.name:
                items.append(item.name)
    return items


def get_valid_give_targets(session: Optional[MUDSession]) -> list[str]:
    """Get names of valid targets for giving items.

    Valid targets are players, AIs, NPCs, and objects (anything we can hand something to).

    Args:
        session: Current MUD session, may be None

    Returns:
        List of entity names that can receive items
    """
    targets: list[str] = []
    world_state = session.world_state if session else None
    if world_state:
        for entity in world_state.entities_present:
            if entity.is_self:
                continue
            if entity.entity_type in ("player", "ai", "npc", "object"):
                if entity.name:
                    targets.append(entity.name)
    else:
        # Fall back to session entities
        if session:
            for entity in session.entities_present:
                if entity.is_self:
                    continue
                if entity.entity_type in ("player", "ai", "npc", "object"):
                    if entity.name:
                        targets.append(entity.name)
    return targets
