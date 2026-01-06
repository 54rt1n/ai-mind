# aim/app/mud/worker/turns/decision.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Decision phase validation for MUD worker turn processing.

Functions for validating tool decisions from the LLM during phase 1.
"""

from typing import Optional
from aim_mud_types import MUDSession
from .validation import (
    resolve_move_location,
    get_room_objects,
    get_inventory_items,
    get_valid_give_targets,
)


class DecisionResult:
    """Result of validating a tool decision."""

    def __init__(
        self,
        is_valid: bool,
        args: Optional[dict] = None,
        guidance: Optional[str] = None
    ):
        """Initialize decision result.

        Args:
            is_valid: Whether the decision was valid
            args: Arguments to pass to the tool if valid
            guidance: Error guidance if invalid
        """
        self.is_valid = is_valid
        self.args = args or {}
        self.guidance = guidance


def validate_move(session: Optional[MUDSession], args: dict) -> DecisionResult:
    """Validate a move/navigation decision.

    Args:
        session: Current MUD session
        args: Arguments from the tool call (location/direction)

    Returns:
        DecisionResult with validation outcome
    """
    location = args.get("location") or args.get("direction")
    resolved = resolve_move_location(session, location)

    if not resolved:
        # Get valid exits for guidance
        valid_exits = []
        if session and session.current_room and session.current_room.exits:
            valid_exits = list(session.current_room.exits.keys())
        exits_str = ", ".join(valid_exits) if valid_exits else "none available"
        guidance = (
            f"Invalid move location '{location}'. "
            f"Valid exits are: {exits_str}. "
            f"Please try again with a valid exit, or use {{\"speak\": {{}}}} to respond instead."
        )
        return DecisionResult(is_valid=False, guidance=guidance)

    return DecisionResult(is_valid=True, args={"location": resolved})


def validate_take(session: Optional[MUDSession], args: dict) -> DecisionResult:
    """Validate a take/grab decision.

    Args:
        session: Current MUD session
        args: Arguments from the tool call (object)

    Returns:
        DecisionResult with validation outcome
    """
    obj = args.get("object")
    room_objects = get_room_objects(session)

    if obj and obj.lower() in [o.lower() for o in room_objects]:
        return DecisionResult(is_valid=True, args=args)

    # Invalid - give guidance
    objects_str = ", ".join(room_objects) if room_objects else "nothing here to take"
    guidance = (
        f"Cannot take '{obj}'. "
        f"Available items: {objects_str}. "
        f"Please try again with a valid item, or use {{\"speak\": {{}}}} to respond instead."
    )
    return DecisionResult(is_valid=False, guidance=guidance)


def validate_drop(session: Optional[MUDSession], args: dict) -> DecisionResult:
    """Validate a drop/discard decision.

    Args:
        session: Current MUD session
        args: Arguments from the tool call (object)

    Returns:
        DecisionResult with validation outcome
    """
    obj = args.get("object")
    inventory = get_inventory_items(session)

    if obj and obj.lower() in [i.lower() for i in inventory]:
        return DecisionResult(is_valid=True, args=args)

    # Invalid - give guidance
    inventory_str = ", ".join(inventory) if inventory else "nothing in inventory"
    guidance = (
        f"Cannot drop '{obj}'. "
        f"Your inventory: {inventory_str}. "
        f"Please try again with an item you're carrying, or use {{\"speak\": {{}}}} to respond instead."
    )
    return DecisionResult(is_valid=False, guidance=guidance)


def validate_give(session: Optional[MUDSession], args: dict) -> DecisionResult:
    """Validate a give/transfer decision.

    Args:
        session: Current MUD session
        args: Arguments from the tool call (object, target)

    Returns:
        DecisionResult with validation outcome
    """
    obj = args.get("object")
    target = args.get("target")

    # Get inventory and valid targets
    inventory = get_inventory_items(session)
    valid_targets = get_valid_give_targets(session)

    obj_valid = obj and obj.lower() in [i.lower() for i in inventory]
    target_valid = target and target.lower() in [t.lower() for t in valid_targets]

    if obj_valid and target_valid:
        return DecisionResult(is_valid=True, args=args)

    # Build specific guidance
    errors = []
    if not obj_valid:
        inventory_str = ", ".join(inventory) if inventory else "nothing"
        errors.append(f"Cannot give '{obj}'. Your inventory: {inventory_str}.")
    if not target_valid:
        targets_str = ", ".join(valid_targets) if valid_targets else "no one here"
        errors.append(f"Cannot give to '{target}'. People present: {targets_str}.")

    guidance = (
        " ".join(errors) + " "
        f"Please try again with valid item and target, or use {{\"speak\": {{}}}} to respond instead."
    )
    return DecisionResult(is_valid=False, guidance=guidance)
