# tests/unit/mud/test_world_state.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for WorldState XML serialization."""

from aim_mud_types import RoomState, EntityState, WorldState
from aim_mud_types.world_state import InventoryItem, WhoEntry


def test_world_state_to_xml_with_entities_and_objects():
    room = RoomState(room_id="#1", name="Garden", description="A quiet garden.")
    entities = [
        EntityState(
            entity_id="c1",
            name="Prax",
            entity_type="player",
            description="The creator.",
            is_self=False,
        ),
        EntityState(
            entity_id="o1",
            name="Lantern",
            entity_type="object",
            description="A brass lantern.",
            is_self=False,
        ),
    ]
    state = WorldState(room_state=room, entities_present=entities)
    xml = state.to_xml(include_self=False)

    assert "<world_state>" in xml
    assert '<location name="Garden" id="#1">' in xml
    assert "A quiet garden." in xml
    assert "<present>" in xml
    assert '<entity name="Prax" type="player" id="c1">The creator.</entity>' in xml
    assert "<objects>" in xml
    assert '<object name="Lantern" type="object" id="o1">A brass lantern.</object>' in xml


def test_world_state_to_xml_includes_inventory_who_time_home():
    state = WorldState(
        inventory=[InventoryItem(item_id="i1", name="Key", description="A silver key.")],
        who=[WhoEntry(name="Andi", status="online", location="Garden", is_self=True)],
        time="2026-01-01T12:00:00+00:00",
        home="#2",
    )
    xml = state.to_xml()

    assert "<inventory>" in xml
    assert '<item name="Key" id="i1">A silver key.</item>' in xml
    assert "<who>" in xml
    assert 'name="Andi"' in xml
    assert 'status="online"' in xml
    assert 'location="Garden"' in xml
    assert "<time>2026-01-01T12:00:00+00:00</time>" in xml
    assert "<home>#2</home>" in xml
