# packages/aim-mud/tests/mud_tests/unit/test_profile.py
"""Tests for RoomProfile and AgentProfile Pydantic models."""

import json
import pytest
from datetime import datetime, timezone

from aim_mud_types.profile import RoomProfile, AgentProfile
from aim_mud_types.state import EntityState, RoomState


class TestRoomProfile:
    """Test RoomProfile validation and JSON string parsing."""

    def test_room_profile_with_parsed_objects(self):
        """Test RoomProfile validation with already-parsed objects."""
        room_state = RoomState(
            room_id="#2",
            name="Limbo",
            description="A void between worlds",
            exits={},
        )
        entities = [
            EntityState(
                entity_id="#3",
                name="Andi",
                agent_id="andi",
                entity_type="ai",
            ),
            EntityState(
                entity_id="#43",
                name="Val",
                agent_id="val",
                entity_type="ai",
            ),
        ]

        profile = RoomProfile(
            room_id="#2",
            name="Limbo",
            desc="A void between worlds",
            room_state=room_state,
            entities=entities,
        )

        assert profile.room_id == "#2"
        assert profile.name == "Limbo"
        assert len(profile.entities) == 2
        assert profile.entities[0].agent_id == "andi"
        assert profile.entities[1].agent_id == "val"

    def test_room_profile_with_json_strings(self):
        """Test RoomProfile validation with JSON strings (Evennia format)."""
        # This is how Evennia stores data in Redis
        room_state_json = json.dumps({
            "room_id": "#2",
            "name": "Limbo",
            "description": "A void between worlds",
            "exits": {},
        })
        entities_json = json.dumps([
            {
                "entity_id": "#3",
                "name": "Andi",
                "agent_id": "andi",
                "entity_type": "ai",
            },
            {
                "entity_id": "#43",
                "name": "Val",
                "agent_id": "val",
                "entity_type": "ai",
            },
        ])

        # Simulate Redis data structure
        data = {
            "room_state": room_state_json,
            "entities_present": entities_json,  # Uses alias
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        profile = RoomProfile.model_validate(data)

        assert profile.room_id == "#2"
        assert profile.name == "Limbo"
        assert len(profile.entities) == 2
        assert profile.entities[0].agent_id == "andi"
        assert profile.entities[1].agent_id == "val"

    def test_room_profile_extracts_room_id_from_room_state(self):
        """Test that room_id is extracted from room_state if not provided."""
        room_state_json = json.dumps({
            "room_id": "#2",
            "name": "Limbo",
            "description": "A void between worlds",
            "exits": {},
        })

        # Don't provide room_id at top level
        data = {
            "room_state": room_state_json,
            "entities_present": "[]",
        }

        profile = RoomProfile.model_validate(data)

        # Should extract from room_state
        assert profile.room_id == "#2"
        assert profile.name == "Limbo"

    def test_room_profile_with_empty_entities(self):
        """Test RoomProfile with no entities present."""
        room_state_json = json.dumps({
            "room_id": "#2",
            "name": "Limbo",
            "description": "Empty room",
            "exits": {},
        })

        data = {
            "room_state": room_state_json,
            "entities_present": "[]",
        }

        profile = RoomProfile.model_validate(data)

        assert profile.room_id == "#2"
        assert len(profile.entities) == 0

    def test_room_profile_handles_invalid_json(self):
        """Test RoomProfile gracefully handles invalid JSON strings."""
        data = {
            "room_id": "#2",
            "name": "Test Room",
            "room_state": "NOT VALID JSON{",  # Invalid JSON
            "entities_present": "[NOT VALID",  # Invalid JSON
        }

        profile = RoomProfile.model_validate(data)

        # Should use defaults for invalid JSON
        assert profile.room_id == "#2"
        assert profile.room_state is None
        assert profile.entities == []

    def test_room_profile_with_alias(self):
        """Test that entities_present alias works correctly."""
        entities_json = json.dumps([
            {
                "entity_id": "#3",
                "name": "Andi",
                "agent_id": "andi",
                "entity_type": "ai",
            },
        ])

        # Use both the field name and alias
        data1 = {
            "room_id": "#2",
            "name": "Test",
            "entities": entities_json,  # Field name
        }
        data2 = {
            "room_id": "#2",
            "name": "Test",
            "entities_present": entities_json,  # Alias
        }

        profile1 = RoomProfile.model_validate(data1)
        profile2 = RoomProfile.model_validate(data2)

        assert len(profile1.entities) == 1
        assert len(profile2.entities) == 1
        assert profile1.entities[0].agent_id == "andi"
        assert profile2.entities[0].agent_id == "andi"


class TestRoomProfileIntegration:
    """Integration tests for RoomProfile with Redis client."""

    async def test_agents_from_room_profile_with_real_data(self):
        """Test extracting agent_ids from RoomProfile with EntityState objects."""
        # Create a RoomProfile with EntityState objects (not dicts)
        room_state_json = json.dumps({
            "room_id": "#2",
            "name": "Limbo",
            "description": "Test room",
            "exits": {},
        })
        entities_json = json.dumps([
            {
                "entity_id": "#3",
                "name": "Andi",
                "agent_id": "andi",
                "entity_type": "ai",
            },
            {
                "entity_id": "#43",
                "name": "Val",
                "agent_id": "val",
                "entity_type": "ai",
            },
            {
                "entity_id": "#44",
                "name": "Rock",
                "entity_type": "object",  # Not an AI
            },
        ])

        data = {
            "room_state": room_state_json,
            "entities_present": entities_json,
        }

        profile = RoomProfile.model_validate(data)

        # Now extract agent_ids like the mediator does
        agent_ids = []
        for entity in profile.entities:
            if entity.entity_type != "ai":
                continue
            if entity.agent_id:
                agent_ids.append(entity.agent_id)

        # Should find both AI agents, but not the rock
        assert len(agent_ids) == 2
        assert "andi" in agent_ids
        assert "val" in agent_ids


class TestAgentProfile:
    """Test AgentProfile validation."""

    def test_agent_profile_creation(self):
        """Test basic AgentProfile creation."""
        profile = AgentProfile(
            agent_id="andi",
            persona_id="andi",
            last_event_id="1768112909557-0",
            conversation_id="conv123",
        )

        assert profile.agent_id == "andi"
        assert profile.persona_id == "andi"
        assert profile.last_event_id == "1768112909557-0"
        assert profile.conversation_id == "conv123"

    def test_agent_profile_defaults(self):
        """Test AgentProfile default values."""
        profile = AgentProfile(agent_id="test")

        assert profile.agent_id == "test"
        assert profile.persona_id is None
        assert profile.last_event_id == "0"
        assert profile.last_action_id is None
        assert profile.conversation_id is None
        assert isinstance(profile.updated_at, datetime)
