# tests/unit/mud/test_inventory_validation.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Regression tests for give/drop inventory validation with wearable items."""

from aim_mud_types import EntityState, InventoryItem, MUDSession, WorldState
from andimud_worker.turns.decision import validate_drop, validate_give
from andimud_worker.turns.validation import get_inventory_items


def _build_session(
    inventory: list[InventoryItem] | None = None,
    worn: list[InventoryItem] | None = None,
    entities: list[EntityState] | None = None,
) -> MUDSession:
    """Build a minimal session for turn validation tests."""
    entities_present = entities or []
    world_state = WorldState(
        entities_present=entities_present,
        inventory=inventory or [],
        worn=worn or [],
    )
    return MUDSession(
        agent_id="test_agent",
        persona_id="test_persona",
        world_state=world_state,
        entities_present=entities_present,
    )


def test_get_inventory_items_includes_worn_items() -> None:
    """Worn clothing should count as carried for give/drop validation."""
    session = _build_session(
        inventory=[],
        worn=[InventoryItem(item_id="#42", name="Red Scarf", is_worn=True)],
    )

    assert get_inventory_items(session) == ["Red Scarf"]


def test_validate_drop_accepts_worn_item() -> None:
    """Drop validation should allow a wearable item name."""
    session = _build_session(
        worn=[InventoryItem(item_id="#42", name="Red Scarf", is_worn=True)],
    )

    result = validate_drop(session, {"object": "red scarf"})

    assert result.is_valid is True
    assert result.args == {"object": "red scarf"}


def test_validate_give_accepts_worn_item() -> None:
    """Give validation should allow wearable items as valid objects."""
    target = EntityState(entity_id="#7", name="Prax", entity_type="npc", is_self=False)
    session = _build_session(
        worn=[InventoryItem(item_id="#42", name="Red Scarf", is_worn=True)],
        entities=[target],
    )

    result = validate_give(session, {"object": "Red Scarf", "target": "Prax"})

    assert result.is_valid is True
    assert result.args == {"object": "Red Scarf", "target": "Prax"}


def test_get_inventory_items_dedupes_inventory_and_worn_names() -> None:
    """Same item name present in multiple carried lists should be deduped."""
    session = _build_session(
        inventory=[InventoryItem(item_id="#42", name="Red Scarf")],
        worn=[InventoryItem(item_id="#42", name="Red Scarf", is_worn=True)],
    )

    assert get_inventory_items(session) == ["Red Scarf"]
