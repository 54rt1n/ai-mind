# tests/unit/mud/test_auras.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Comprehensive unit tests for the aura system.

Tests cover:
- AuraState model validation and serialization
- get_ringable_objects() extraction logic
- validate_ring() action validation
- update_aura_tools() tool loading
- RoomState aura field normalization
- End-to-end integration flows

Testing philosophy: Mock external services only (Redis, LLM).
Use real implementations for all internal logic (ToolLoader, validation functions).
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

from aim_mud_types import (
    RoomState,
    EntityState,
    MUDSession,
    WorldState,
    AURA_RINGABLE,
)
from aim_mud_types.state import AuraState
from andimud_worker.turns.validation import get_ringable_objects
from andimud_worker.turns.decision import validate_ring, DecisionResult
from andimud_worker.conversation.memory.decision import MUDDecisionStrategy
from andimud_worker.tools.helper import ToolHelper
from aim.tool.dto import Tool, ToolFunction, ToolFunctionParameters
from aim.tool.formatting import ToolUser


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_tool():
    """Create a sample tool for testing."""
    return Tool(
        type="test",
        function=ToolFunction(
            name="test_function",
            description="A test function",
            parameters=ToolFunctionParameters(
                type="object",
                properties={"arg1": {"type": "string"}},
                required=["arg1"],
            ),
        ),
    )


@pytest.fixture
def tool_user(sample_tool):
    """Create a ToolUser with a sample tool."""
    return ToolUser([sample_tool])


@pytest.fixture
def test_aura_ringable() -> AuraState:
    """Valid RINGABLE aura with source."""
    return AuraState(
        name="RINGABLE",
        source="Front Door Doorbell",
        source_id="bell_001"
    )


@pytest.fixture
def test_aura_ringable_no_source() -> AuraState:
    """RINGABLE aura without source (should be filtered out)."""
    return AuraState(name="RINGABLE", source="", source_id="")


@pytest.fixture
def test_aura_custom() -> AuraState:
    """Non-ringable aura (different capability)."""
    return AuraState(name="TELEPATHY", source="", source_id="")


@pytest.fixture
def test_room_with_ringable_aura() -> RoomState:
    """Room with single RINGABLE aura."""
    return RoomState(
        room_id="#123",
        name="Entrance Hall",
        description="A grand entrance.",
        exits={"out": "#1"},
        auras=[AuraState(name="RINGABLE", source="Front Door Doorbell", source_id="bell_001")]
    )


@pytest.fixture
def test_room_with_multiple_auras() -> RoomState:
    """Room with multiple ringable and non-ringable auras."""
    return RoomState(
        room_id="#456",
        name="Ritual Chamber",
        description="A mystical space.",
        exits={"back": "#2"},
        auras=[
            AuraState(name="RINGABLE", source="Ritual Bell", source_id="bell_002"),
            AuraState(name="RINGABLE", source="Silver Chime", source_id="chime_001"),
            AuraState(name="TELEPATHY", source="", source_id=""),
        ]
    )


@pytest.fixture
def test_session_with_ringable_room(test_room_with_ringable_aura) -> MUDSession:
    """MUD session with a ringable room."""
    return MUDSession(
        agent_id="test_agent",
        persona_id="test_persona",
        current_room=test_room_with_ringable_aura,
        world_state=WorldState(
            room_state=test_room_with_ringable_aura,
            entities_present=[],
            inventory=[]
        ),
        entities_present=[]
    )


@pytest.fixture
def tools_path() -> str:
    """Path to config/tools for real YAML loading."""
    # Navigate from packages/aim-mud/tests/mud_tests/unit/mud/test_auras.py
    # to the project root, then to config/tools
    test_file = Path(__file__)
    # Go up 6 levels: mud -> unit -> mud_tests -> tests -> aim-mud -> packages
    project_root = test_file.parent.parent.parent.parent.parent.parent.parent
    return str(project_root / "config" / "tools")


# ============================================================================
# TestAuraState - Pydantic model validation (6 tests)
# ============================================================================

class TestAuraState:
    """Tests for AuraState Pydantic model."""

    def test_aura_state_minimal(self):
        """Test AuraState with name only."""
        aura = AuraState(name="RINGABLE")
        assert aura.name == "RINGABLE"
        assert aura.source == ""
        assert aura.source_id == ""

    def test_aura_state_complete(self):
        """Test AuraState with all fields."""
        aura = AuraState(
            name="RINGABLE",
            source="Front Door Doorbell",
            source_id="bell_001"
        )
        assert aura.name == "RINGABLE"
        assert aura.source == "Front Door Doorbell"
        assert aura.source_id == "bell_001"

    def test_aura_state_serialization(self):
        """Test model_dump() produces correct dict structure."""
        aura = AuraState(
            name="RINGABLE",
            source="Front Door Doorbell",
            source_id="bell_001"
        )
        dumped = aura.model_dump()
        assert dumped == {
            "name": "RINGABLE",
            "source": "Front Door Doorbell",
            "source_id": "bell_001"
        }

    def test_aura_state_deserialization(self):
        """Test model_validate() from dict creates AuraState."""
        data = {
            "name": "RINGABLE",
            "source": "Front Door Doorbell",
            "source_id": "bell_001"
        }
        aura = AuraState.model_validate(data)
        assert isinstance(aura, AuraState)
        assert aura.name == "RINGABLE"
        assert aura.source == "Front Door Doorbell"
        assert aura.source_id == "bell_001"

    def test_aura_state_empty_strings_default(self):
        """Test that None values are handled appropriately."""
        # Pydantic defaults handle empty strings, not None conversion
        aura = AuraState(name="RINGABLE")
        assert aura.source == ""
        assert aura.source_id == ""

    def test_aura_ringable_normalized_name(self):
        """Test that name is stored exactly as provided."""
        aura_upper = AuraState(name="RINGABLE")
        aura_lower = AuraState(name="ringable")
        aura_mixed = AuraState(name="Ringable")

        assert aura_upper.name == "RINGABLE"
        assert aura_lower.name == "ringable"
        assert aura_mixed.name == "Ringable"


# ============================================================================
# TestGetRingableObjects - Extraction logic (10 tests)
# ============================================================================

class TestGetRingableObjects:
    """Tests for get_ringable_objects() extraction logic."""

    def test_get_ringable_objects_single(self, test_session_with_ringable_room):
        """Test extraction with single ringable object."""
        ringables = get_ringable_objects(test_session_with_ringable_room)
        assert ringables == ["Front Door Doorbell"]

    def test_get_ringable_objects_multiple(self):
        """Test extraction with multiple ringable objects."""
        room = RoomState(
            room_id="#456",
            name="Ritual Chamber",
            auras=[
                AuraState(name="RINGABLE", source="Ritual Bell", source_id="bell_002"),
                AuraState(name="RINGABLE", source="Silver Chime", source_id="chime_001"),
            ]
        )
        session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            current_room=room,
            world_state=WorldState(room_state=room, entities_present=[], inventory=[]),
            entities_present=[]
        )
        ringables = get_ringable_objects(session)
        assert len(ringables) == 2
        assert "Ritual Bell" in ringables
        assert "Silver Chime" in ringables

    def test_get_ringable_objects_duplicates_deduped(self):
        """Test that duplicate sources are deduplicated."""
        room = RoomState(
            room_id="#789",
            name="Echo Chamber",
            auras=[
                AuraState(name="RINGABLE", source="Ancient Bell", source_id="bell_003"),
                AuraState(name="RINGABLE", source="Ancient Bell", source_id="bell_004"),
            ]
        )
        session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            current_room=room,
            world_state=WorldState(room_state=room, entities_present=[], inventory=[]),
            entities_present=[]
        )
        ringables = get_ringable_objects(session)
        # STRICT ASSERTION: Deduplication MUST happen
        assert "Ancient Bell" in ringables
        assert len(ringables) == 1, f"Expected 1 unique ringable, got {len(ringables)}: {ringables}"

    def test_get_ringable_objects_empty_session(self):
        """Test with None session."""
        ringables = get_ringable_objects(None)
        assert ringables == []

    def test_get_ringable_objects_no_room(self):
        """Test session without current_room."""
        session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            current_room=None,
            world_state=None,
            entities_present=[]
        )
        ringables = get_ringable_objects(session)
        assert ringables == []

    def test_get_ringable_objects_no_auras(self):
        """Test room with empty auras list."""
        room = RoomState(
            room_id="#111",
            name="Empty Room",
            auras=[]
        )
        session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            current_room=room,
            world_state=WorldState(room_state=room, entities_present=[], inventory=[]),
            entities_present=[]
        )
        ringables = get_ringable_objects(session)
        assert ringables == []

    def test_get_ringable_objects_empty_aura_source(self):
        """Test RINGABLE aura with empty source is excluded."""
        room = RoomState(
            room_id="#222",
            name="Silent Room",
            auras=[
                AuraState(name="RINGABLE", source="", source_id=""),
            ]
        )
        session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            current_room=room,
            world_state=WorldState(room_state=room, entities_present=[], inventory=[]),
            entities_present=[]
        )
        ringables = get_ringable_objects(session)
        assert ringables == []

    def test_get_ringable_objects_non_ringable_aura(self):
        """Test that non-ringable auras don't appear in results."""
        room = RoomState(
            room_id="#333",
            name="Mystic Chamber",
            auras=[
                AuraState(name="TELEPATHY", source="Mind Link", source_id="tele_001"),
            ]
        )
        session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            current_room=room,
            world_state=WorldState(room_state=room, entities_present=[], inventory=[]),
            entities_present=[]
        )
        ringables = get_ringable_objects(session)
        assert ringables == []

    def test_get_ringable_objects_mixed_auras(self):
        """Test mix of valid/invalid ringables and other auras."""
        room = RoomState(
            room_id="#444",
            name="Mixed Room",
            auras=[
                AuraState(name="RINGABLE", source="Valid Bell", source_id="bell_005"),
                AuraState(name="RINGABLE", source="", source_id=""),  # Invalid: empty source
                AuraState(name="TELEPATHY", source="Mind Link", source_id="tele_001"),
            ]
        )
        session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            current_room=room,
            world_state=WorldState(room_state=room, entities_present=[], inventory=[]),
            entities_present=[]
        )
        ringables = get_ringable_objects(session)
        assert ringables == ["Valid Bell"]

    def test_get_ringable_objects_case_insensitive(self):
        """Test that both 'ringable' and 'RINGABLE' are matched."""
        room = RoomState(
            room_id="#555",
            name="Case Test Room",
            auras=[
                AuraState(name="ringable", source="Lower Bell", source_id="bell_006"),
                AuraState(name="RINGABLE", source="Upper Bell", source_id="bell_007"),
                AuraState(name="Ringable", source="Mixed Bell", source_id="bell_008"),
            ]
        )
        session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            current_room=room,
            world_state=WorldState(room_state=room, entities_present=[], inventory=[]),
            entities_present=[]
        )
        ringables = get_ringable_objects(session)
        assert len(ringables) == 3
        assert "Lower Bell" in ringables
        assert "Upper Bell" in ringables
        assert "Mixed Bell" in ringables


# ============================================================================
# TestValidateRing - Ring action validation (8 tests)
# ============================================================================

class TestValidateRing:
    """Tests for validate_ring() validation logic."""

    def test_validate_ring_valid_with_object(self, test_session_with_ringable_room):
        """Test validation with valid object specified."""
        result = validate_ring(test_session_with_ringable_room, {"object": "Front Door Doorbell"})
        assert result.is_valid is True
        assert result.args == {"object": "Front Door Doorbell"}

    def test_validate_ring_valid_case_insensitive(self, test_session_with_ringable_room):
        """Test case-insensitive object matching."""
        result = validate_ring(test_session_with_ringable_room, {"object": "front door doorbell"})
        assert result.is_valid is True

    def test_validate_ring_valid_empty_object_single_ringable(self, test_session_with_ringable_room):
        """Test empty object with single ringable auto-populates."""
        result = validate_ring(test_session_with_ringable_room, {"object": ""})
        assert result.is_valid is True
        assert result.args == {"object": "Front Door Doorbell"}

    def test_validate_ring_invalid_object(self, test_session_with_ringable_room):
        """Test validation with invalid object name."""
        result = validate_ring(test_session_with_ringable_room, {"object": "Magic Mirror"})
        assert result.is_valid is False
        assert "Cannot ring 'Magic Mirror'" in result.guidance
        assert "Front Door Doorbell" in result.guidance

    def test_validate_ring_invalid_empty_object_multiple(self):
        """Test empty object with multiple ringables requires clarification."""
        room = RoomState(
            room_id="#456",
            name="Ritual Chamber",
            auras=[
                AuraState(name="RINGABLE", source="Ritual Bell", source_id="bell_002"),
                AuraState(name="RINGABLE", source="Silver Chime", source_id="chime_001"),
            ]
        )
        session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            current_room=room,
            world_state=WorldState(room_state=room, entities_present=[], inventory=[]),
            entities_present=[]
        )
        result = validate_ring(session, {"object": ""})
        assert result.is_valid is False
        assert "Ring which object?" in result.guidance
        assert "Ritual Bell" in result.guidance
        assert "Silver Chime" in result.guidance

    def test_validate_ring_invalid_no_ringables(self):
        """Test validation when room has no ringable auras."""
        room = RoomState(
            room_id="#666",
            name="Empty Room",
            auras=[]
        )
        session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            current_room=room,
            world_state=WorldState(room_state=room, entities_present=[], inventory=[]),
            entities_present=[]
        )
        result = validate_ring(session, {"object": "anything"})
        assert result.is_valid is False
        assert "No ringable objects are available" in result.guidance

    def test_validate_ring_empty_session(self):
        """Test validation with None session."""
        result = validate_ring(None, {"object": "anything"})
        assert result.is_valid is False

    def test_validate_ring_guidance_includes_options(self):
        """Test that guidance includes list of valid ringables."""
        room = RoomState(
            room_id="#777",
            name="Test Room",
            auras=[
                AuraState(name="RINGABLE", source="Option A", source_id="opt_a"),
                AuraState(name="RINGABLE", source="Option B", source_id="opt_b"),
            ]
        )
        session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            current_room=room,
            world_state=WorldState(room_state=room, entities_present=[], inventory=[]),
            entities_present=[]
        )
        result = validate_ring(session, {"object": "Invalid"})
        assert result.is_valid is False
        assert "Option A" in result.guidance
        assert "Option B" in result.guidance


# ============================================================================
# TestUpdateAuraTools - Tool loading (8 tests)
# ============================================================================

class TestUpdateAuraTools:
    """Tests for update_aura_tools() in MUDDecisionStrategy and ToolHelper."""

    def test_update_aura_tools_ringable_loads(self, tool_user, tools_path):
        """Test that 'ringable' aura loads the ring tool."""
        helper = ToolHelper(tool_user)
        helper.update_aura_tools(["ringable"], tools_path)

        # Check that aura tools were loaded
        assert len(helper._aura_tools) > 0
        # Verify ring tool is present
        tool_names = [getattr(t.function, "name", None) for t in helper._aura_tools]
        assert "ring" in tool_names

    def test_update_aura_tools_empty_auras(self, tool_user, tools_path):
        """Test that empty auras list clears aura tools."""
        helper = ToolHelper(tool_user)
        # First load some tools
        helper.update_aura_tools(["ringable"], tools_path)
        assert len(helper._aura_tools) > 0

        # Then clear with empty list
        helper.update_aura_tools([], tools_path)
        assert len(helper._aura_tools) == 0

    def test_update_aura_tools_skip_if_same(self, tool_user, tools_path):
        """Test that tools are cached and not reloaded if auras unchanged."""
        helper = ToolHelper(tool_user)
        helper.update_aura_tools(["ringable"], tools_path)
        first_tools = helper._aura_tools
        first_auras = helper._active_auras

        # Call again with same auras
        helper.update_aura_tools(["ringable"], tools_path)

        # Should be the same (cached)
        assert helper._active_auras == first_auras
        assert helper._aura_tools is first_tools

    def test_update_aura_tools_missing_file(self, tool_user, tools_path):
        """Test that missing YAML file logs warning but doesn't crash."""
        helper = ToolHelper(tool_user)
        # This should not raise an exception
        helper.update_aura_tools(["nonexistent_aura"], tools_path)
        # Should have no aura tools loaded
        assert len(helper._aura_tools) == 0

    def test_update_aura_tools_normalized_auras(self, tool_user, tools_path):
        """Test that aura names are normalized (whitespace, case, deduped)."""
        helper = ToolHelper(tool_user)
        helper.update_aura_tools([" RINGABLE ", " ringable"], tools_path)

        # Should normalize to ["ringable"] (sorted, deduped, lowercase, stripped)
        assert helper._active_auras == ["ringable"]

    def test_update_aura_tools_real_ringable_yaml(self, tool_user, tools_path):
        """Test loading actual ringable.yaml file."""
        helper = ToolHelper(tool_user)
        helper.update_aura_tools(["ringable"], tools_path)

        # Verify the ring tool was loaded with correct structure
        assert len(helper._aura_tools) > 0
        ring_tool = helper._aura_tools[0]
        assert hasattr(ring_tool, "function")
        assert ring_tool.function.name == "ring"
        assert "Ring a ringable object" in ring_tool.function.description

    def test_update_aura_tools_decision_strategy(self, tools_path):
        """Test update_aura_tools in MUDDecisionStrategy."""
        # Create a minimal mock ChatManager
        mock_chat = MagicMock()
        mock_chat.config = MagicMock()

        strategy = MUDDecisionStrategy(mock_chat)
        strategy.update_aura_tools(["ringable"], tools_path)

        # Verify tools were loaded
        assert len(strategy._aura_tools) > 0
        tool_names = [getattr(t.function, "name", None) for t in strategy._aura_tools]
        assert "ring" in tool_names

    def test_update_aura_tools_tool_helper(self, tool_user, tools_path):
        """Test update_aura_tools in ToolHelper."""
        helper = ToolHelper(tool_user)
        helper.update_aura_tools(["ringable"], tools_path)

        # Verify tools were loaded
        assert len(helper._aura_tools) > 0
        tool_names = [getattr(t.function, "name", None) for t in helper._aura_tools]
        assert "ring" in tool_names


# ============================================================================
# TestRoomStateAuraValidator - Field normalization (5 tests)
# ============================================================================

class TestRoomStateAuraValidator:
    """Tests for RoomState aura field validator."""

    def test_room_aura_validator_none(self):
        """Test that None auras converts to empty list."""
        room = RoomState(room_id="#1", name="Test", auras=None)
        assert room.auras == []

    def test_room_aura_validator_empty_list(self):
        """Test that empty list is preserved."""
        room = RoomState(room_id="#1", name="Test", auras=[])
        assert room.auras == []

    def test_room_aura_validator_single_aura(self):
        """Test that single AuraState is preserved."""
        aura = AuraState(name="RINGABLE", source="Bell", source_id="bell_001")
        room = RoomState(room_id="#1", name="Test", auras=[aura])
        assert len(room.auras) == 1
        assert room.auras[0].name == "RINGABLE"
        assert room.auras[0].source == "Bell"

    def test_room_aura_validator_multiple_auras(self):
        """Test that multiple AuraStates are preserved."""
        auras = [
            AuraState(name="RINGABLE", source="Bell", source_id="bell_001"),
            AuraState(name="TELEPATHY", source="Mind", source_id="mind_001"),
        ]
        room = RoomState(room_id="#1", name="Test", auras=auras)
        assert len(room.auras) == 2
        assert room.auras[0].name == "RINGABLE"
        assert room.auras[1].name == "TELEPATHY"

    def test_room_aura_validator_preserves_source(self):
        """Test that source and source_id are preserved correctly."""
        aura = AuraState(
            name="RINGABLE",
            source="Grand Cathedral Bell",
            source_id="cathedral_bell_main"
        )
        room = RoomState(room_id="#1", name="Test", auras=[aura])
        assert room.auras[0].source == "Grand Cathedral Bell"
        assert room.auras[0].source_id == "cathedral_bell_main"


# ============================================================================
# TestAuraIntegration - End-to-end flows (6 tests)
# ============================================================================

class TestAuraIntegration:
    """End-to-end integration tests for aura workflows."""

    def test_aura_e2e_ringable_flow(self, tool_user, tools_path):
        """Test complete flow: room with RINGABLE → update tools → validate ring."""
        # Setup room with RINGABLE aura
        room = RoomState(
            room_id="#888",
            name="Bell Tower",
            auras=[
                AuraState(name="RINGABLE", source="Tower Bell", source_id="tower_bell")
            ]
        )
        session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            current_room=room,
            world_state=WorldState(room_state=room, entities_present=[], inventory=[]),
            entities_present=[]
        )

        # Load aura tools
        helper = ToolHelper(tool_user)
        aura_names = [a.name for a in room.auras]
        helper.update_aura_tools(aura_names, tools_path)

        # Verify ring tool is available
        tool_names = [getattr(t.function, "name", None) for t in helper._aura_tools]
        assert "ring" in tool_names

        # Validate ring action
        result = validate_ring(session, {"object": "Tower Bell"})
        assert result.is_valid is True

    def test_aura_e2e_no_tools_when_empty(self, tool_user, tools_path):
        """Test that clearing auras removes tools."""
        helper = ToolHelper(tool_user)

        # Load auras
        helper.update_aura_tools(["ringable"], tools_path)
        assert len(helper._aura_tools) > 0

        # Clear auras
        helper.update_aura_tools([], tools_path)
        assert len(helper._aura_tools) == 0

    def test_aura_e2e_user_guidance(self):
        """Test that ringable sources appear in validation guidance."""
        room = RoomState(
            room_id="#999",
            name="Test Room",
            auras=[
                AuraState(name="RINGABLE", source="User Guidance Bell", source_id="ugb_001")
            ]
        )
        session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            current_room=room,
            world_state=WorldState(room_state=room, entities_present=[], inventory=[]),
            entities_present=[]
        )

        # Try to ring invalid object
        result = validate_ring(session, {"object": "Invalid Bell"})
        assert result.is_valid is False
        assert "User Guidance Bell" in result.guidance
        # STRICT ASSERTION: No duplicate mentions in guidance
        guidance_count = result.guidance.count("User Guidance Bell")
        assert guidance_count == 1, f"Expected 'User Guidance Bell' to appear once in guidance, but it appears {guidance_count} times. Guidance: {result.guidance}"

    def test_aura_e2e_multiple_ringables_in_guidance(self):
        """Test that multiple ringables are shown in guidance."""
        room = RoomState(
            room_id="#1000",
            name="Multi Bell Room",
            auras=[
                AuraState(name="RINGABLE", source="First Bell", source_id="fb_001"),
                AuraState(name="RINGABLE", source="Second Bell", source_id="sb_002"),
            ]
        )
        session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            current_room=room,
            world_state=WorldState(room_state=room, entities_present=[], inventory=[]),
            entities_present=[]
        )

        # Request with empty object (requires disambiguation)
        result = validate_ring(session, {"object": ""})
        assert result.is_valid is False
        assert "First Bell" in result.guidance
        assert "Second Bell" in result.guidance
        # STRICT ASSERTION: Each bell should appear exactly once (no duplicates)
        first_count = result.guidance.count("First Bell")
        second_count = result.guidance.count("Second Bell")
        assert first_count == 1, f"Expected 'First Bell' to appear once, but it appears {first_count} times. Guidance: {result.guidance}"
        assert second_count == 1, f"Expected 'Second Bell' to appear once, but it appears {second_count} times. Guidance: {result.guidance}"

    def test_aura_e2e_decision_guidance_no_duplication(self):
        """Test that MUDDecisionStrategy._build_decision_guidance() doesn't triplicate auras.

        This tests the bug where lines 478-521 in decision.py have THREE identical loops
        that each append to aura_descriptions, causing each aura to appear 3 times.
        """
        # Create minimal mock ChatManager
        mock_chat = MagicMock()
        mock_chat.config = MagicMock()

        # Create strategy
        strategy = MUDDecisionStrategy(mock_chat)

        # Create session with auras
        room = RoomState(
            room_id="#test",
            name="Test Room",
            auras=[
                AuraState(name="RINGABLE", source="Bell Tower Bell", source_id="btb_001"),
                AuraState(name="TELEPATHY", source="Crystal Focus", source_id="cf_001"),
            ]
        )
        session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            current_room=room,
            world_state=WorldState(room_state=room, entities_present=[], inventory=[]),
            entities_present=[]
        )

        # Build guidance
        guidance = strategy._build_decision_guidance(session)

        # STRICT ASSERTIONS: Each aura should appear exactly once
        bell_count = guidance.count("RINGABLE")
        telepathy_count = guidance.count("TELEPATHY")
        bell_source_count = guidance.count("Bell Tower Bell")
        crystal_count = guidance.count("Crystal Focus")

        # Each aura name should appear exactly ONCE in the guidance
        assert bell_count == 1, f"Expected 'RINGABLE' to appear once, but it appears {bell_count} times. Guidance:\n{guidance}"
        assert telepathy_count == 1, f"Expected 'TELEPATHY' to appear once, but it appears {telepathy_count} times. Guidance:\n{guidance}"
        assert bell_source_count == 1, f"Expected 'Bell Tower Bell' to appear once, but it appears {bell_source_count} times. Guidance:\n{guidance}"
        assert crystal_count == 1, f"Expected 'Crystal Focus' to appear once, but it appears {crystal_count} times. Guidance:\n{guidance}"

    def test_aura_e2e_decision_guidance_single_aura(self):
        """Test decision guidance with a single aura to verify no triplication."""
        mock_chat = MagicMock()
        mock_chat.config = MagicMock()

        strategy = MUDDecisionStrategy(mock_chat)

        room = RoomState(
            room_id="#single",
            name="Single Aura Room",
            auras=[
                AuraState(name="RINGABLE", source="Unique Bell", source_id="ub_001"),
            ]
        )
        session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            current_room=room,
            world_state=WorldState(room_state=room, entities_present=[], inventory=[]),
            entities_present=[]
        )

        guidance = strategy._build_decision_guidance(session)

        # With the triple-loop bug, "Unique Bell" would appear 3 times
        unique_bell_count = guidance.count("Unique Bell")
        assert unique_bell_count == 1, f"Expected 'Unique Bell' to appear once, but it appears {unique_bell_count} times. Guidance:\n{guidance}"

    def test_aura_e2e_agent_action_hints(self):
        """Test that auras appear in agent action hints without NameError."""
        # Create minimal mock ChatManager
        mock_chat = MagicMock()
        mock_chat.config = MagicMock()

        # Create strategy
        strategy = MUDDecisionStrategy(mock_chat)

        # Create session with auras
        room = RoomState(
            room_id="#900",
            name="Agent Test Room",
            auras=[
                AuraState(name="RINGABLE", source="Test Bell", source_id="bell_900")
            ]
        )
        session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            current_room=room,
            world_state=WorldState(room_state=room, entities_present=[], inventory=[]),
            entities_present=[]
        )

        # This should not raise NameError
        hints = strategy._build_agent_action_hints(session)

        # Verify auras are included in hints
        aura_hints = [h for h in hints if "Auras:" in h]
        assert len(aura_hints) == 1
        assert "RINGABLE (source: Test Bell)" in aura_hints[0]
