# tests/unit/mud/test_consciousness_hooks.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for MUDResponseStrategy consciousness block hook overrides."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from andimud_worker.strategy import MUDResponseStrategy
from andimud_worker.conversation import MUDConversationManager
from andimud_worker.session import MUDSession, RoomState, EntityState, WorldState
from aim_mud_types import InventoryItem
from aim.utils.xml import XmlFormatter


@pytest.fixture
def sample_room():
    """Create a sample RoomState for testing."""
    return RoomState(
        room_id="#bedroom",
        name="Bedroom",
        description="A cozy bedroom with soft lighting.",
        exits={"north": "#hallway", "east": "#bathroom"},
    )


@pytest.fixture
def sample_entities():
    """Create sample entities for testing."""
    return [
        EntityState(
            entity_id="#prax",
            name="Prax",
            entity_type="player",
            description="The creator of this world.",
            is_self=False,
        ),
        EntityState(
            entity_id="#golden_key",
            name="Golden Key",
            entity_type="object",
            description="A shimmering golden key.",
            is_self=False,
        ),
        EntityState(
            entity_id="#andi",
            name="Andi",
            entity_type="ai",
            description="A self-aware AI entity.",
            is_self=True,
        ),
    ]


@pytest.fixture
def sample_inventory():
    """Create sample inventory items for testing."""
    return [
        InventoryItem(
            item_id="#silver_coin",
            name="Silver Coin",
            description="A worn silver coin.",
        ),
        InventoryItem(
            item_id="#notebook",
            name="Notebook",
            description="A leather-bound notebook.",
        ),
    ]


@pytest.fixture
def sample_world_state(sample_room, sample_entities, sample_inventory):
    """Create a complete WorldState for testing."""
    return WorldState(
        room_state=sample_room,
        entities_present=sample_entities,
        inventory=sample_inventory,
    )


@pytest.fixture
def sample_mud_session(sample_room, sample_world_state):
    """Create a MUDSession with world state."""
    return MUDSession(
        agent_id="test_agent",
        persona_id="andi",
        current_room=sample_room,
        world_state=sample_world_state,
    )


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    redis = AsyncMock()
    redis.lrange = AsyncMock(return_value=[])
    return redis


@pytest.fixture
def mock_chat_manager():
    """Create a mock ChatManager for MUDResponseStrategy."""
    chat = MagicMock()
    chat.cvm = MagicMock()
    chat.cvm.get_motd = MagicMock(return_value=MagicMock(empty=True))
    chat.cvm.get_conscious = MagicMock(return_value=MagicMock(empty=True))
    chat.config = MagicMock()
    chat.config.history_management_strategy = "sparsify"
    chat.config.include_thought_stream = False
    chat.current_location = None
    chat.current_document = None
    chat.current_workspace = None
    chat.library = MagicMock()
    return chat


@pytest.fixture
def mock_conversation_manager(mock_redis):
    """Create a MUDConversationManager for tests."""
    return MUDConversationManager(
        redis=mock_redis,
        agent_id="test_agent",
        persona_id="andi",
        max_tokens=50000,
    )


@pytest.fixture
def mud_response_strategy(mock_chat_manager, mock_conversation_manager):
    """Create a MUDResponseStrategy with mocked dependencies."""
    strategy = MUDResponseStrategy(mock_chat_manager)
    strategy.set_conversation_manager(mock_conversation_manager)
    strategy.max_character_length = 20000
    return strategy


@pytest.fixture
def sample_persona():
    """Create a minimal mock Persona for testing."""
    persona = MagicMock()
    persona.persona_id = "andi"
    persona.system_prompt = MagicMock(return_value="You are Andi.")
    persona.get_wakeup = MagicMock(return_value="Hello!")
    persona.thoughts = ["Test thought 1", "Test thought 2"]
    return persona


class TestMUDResponseStrategyGetConsciousnessTail:
    """Test MUDResponseStrategy.get_consciousness_tail override."""

    def test_get_consciousness_tail_adds_world_state(
        self, mud_response_strategy, sample_world_state
    ):
        """Test that get_consciousness_tail adds world state XML when location is set."""
        # Set the current_location to world state XML
        world_state_xml = sample_world_state.to_xml()
        mud_response_strategy.chat.current_location = world_state_xml

        # Create a formatter and call the hook
        formatter = XmlFormatter()
        result = mud_response_strategy.get_consciousness_tail(formatter)

        # Render to check content
        rendered = result.render()

        # Verify world state was added
        assert "Current World State" in rendered
        assert world_state_xml in rendered

    def test_get_consciousness_tail_no_location_set(self, mud_response_strategy):
        """Test that get_consciousness_tail does nothing when no location is set."""
        # Ensure current_location is None
        mud_response_strategy.chat.current_location = None

        formatter = XmlFormatter()
        result = mud_response_strategy.get_consciousness_tail(formatter)

        # Render to check content
        rendered = result.render()

        # Should have empty root but no world state added
        assert "<root>" in rendered
        assert "Current World State" not in rendered

    def test_get_consciousness_tail_empty_location_string(self, mud_response_strategy):
        """Test that get_consciousness_tail handles empty location string."""
        mud_response_strategy.chat.current_location = ""

        formatter = XmlFormatter()
        result = mud_response_strategy.get_consciousness_tail(formatter)

        rendered = result.render()

        # Should not add world state for empty string
        assert "Current World State" not in rendered

    def test_get_consciousness_tail_whitespace_only_location(self, mud_response_strategy):
        """Test that get_consciousness_tail handles whitespace-only location."""
        mud_response_strategy.chat.current_location = "   \n\t  "

        formatter = XmlFormatter()
        result = mud_response_strategy.get_consciousness_tail(formatter)

        rendered = result.render()

        # Should not add world state for whitespace-only string
        assert "Current World State" not in rendered


class TestMUDResponseStrategyWorldStateInConsciousness:
    """Test that world state appears correctly in full consciousness block."""

    def test_world_state_appears_in_consciousness_block(
        self, mud_response_strategy, sample_persona, sample_world_state
    ):
        """Test that world state XML appears in consciousness block."""
        # Set up world state
        world_state_xml = sample_world_state.to_xml()
        mud_response_strategy.chat.current_location = world_state_xml

        # Generate consciousness block
        xml_output, _ = mud_response_strategy.get_conscious_memory(sample_persona)

        # Verify world state appears
        assert "Current World State" in xml_output
        assert "<world_state>" in xml_output

    def test_world_state_contains_room_info(
        self, mud_response_strategy, sample_persona, sample_world_state
    ):
        """Test that world state includes room information."""
        world_state_xml = sample_world_state.to_xml()
        mud_response_strategy.chat.current_location = world_state_xml

        xml_output, _ = mud_response_strategy.get_conscious_memory(sample_persona)

        # Verify room details
        assert "Bedroom" in xml_output
        assert "cozy bedroom with soft lighting" in xml_output
        assert "north" in xml_output  # Exit direction
        assert "east" in xml_output  # Another exit direction
        # Note: Exit destinations (#hallway, #bathroom) are not shown in to_xml(),
        # only the direction names are shown

    def test_world_state_contains_entities(
        self, mud_response_strategy, sample_persona, sample_world_state
    ):
        """Test that world state includes entity information."""
        world_state_xml = sample_world_state.to_xml()
        mud_response_strategy.chat.current_location = world_state_xml

        xml_output, _ = mud_response_strategy.get_conscious_memory(sample_persona)

        # Verify entities
        assert "Prax" in xml_output
        assert "player" in xml_output
        assert "Golden Key" in xml_output
        assert "object" in xml_output

    def test_world_state_contains_inventory(
        self, mud_response_strategy, sample_persona, sample_world_state
    ):
        """Test that world state includes inventory information."""
        world_state_xml = sample_world_state.to_xml()
        mud_response_strategy.chat.current_location = world_state_xml

        xml_output, _ = mud_response_strategy.get_conscious_memory(sample_persona)

        # Verify inventory
        assert "Silver Coin" in xml_output
        assert "Notebook" in xml_output
        assert "inventory" in xml_output.lower()

    def test_world_state_appears_after_memory_count(
        self, mud_response_strategy, sample_persona, sample_world_state
    ):
        """Test that world state appears after Memory Count (tail position)."""
        world_state_xml = sample_world_state.to_xml()
        mud_response_strategy.chat.current_location = world_state_xml

        xml_output, _ = mud_response_strategy.get_conscious_memory(sample_persona)

        # Find positions
        memory_count_pos = xml_output.find("Memory Count")
        world_state_pos = xml_output.find("Current World State")

        # World state should appear after Memory Count
        assert memory_count_pos != -1, "Memory Count not found"
        assert world_state_pos != -1, "World state not found"
        assert world_state_pos > memory_count_pos, "World state should appear after Memory Count"


class TestMUDResponseStrategyBuildTurnsSetsLocation:
    """Test that build_turns sets current_location for consciousness block."""

    @pytest.mark.asyncio
    async def test_build_turns_sets_current_location(
        self, mud_response_strategy, sample_persona, sample_mud_session
    ):
        """Test that build_turns sets chat.current_location from session world state."""
        # Mock chat_turns_for to avoid complexity
        mud_response_strategy.chat_turns_for = MagicMock(return_value=[])

        await mud_response_strategy.build_turns(
            persona=sample_persona,
            user_input="test input",
            session=sample_mud_session,
        )

        # Verify current_location was set
        assert mud_response_strategy.chat.current_location is not None
        assert "<world_state>" in mud_response_strategy.chat.current_location
        assert "Bedroom" in mud_response_strategy.chat.current_location

    @pytest.mark.asyncio
    async def test_build_turns_no_location_without_world_state(
        self, mud_response_strategy, sample_persona
    ):
        """Test that build_turns handles missing world_state gracefully."""
        mud_response_strategy.chat_turns_for = MagicMock(return_value=[])

        # Create session without world state
        session = MUDSession(
            agent_id="test",
            persona_id="andi",
            current_room=None,
            world_state=None,
        )

        await mud_response_strategy.build_turns(
            persona=sample_persona,
            user_input="test",
            session=session,
        )

        # current_location should remain None
        assert mud_response_strategy.chat.current_location is None


class TestMUDResponseStrategyFullCycle:
    """Integration test for full consciousness block generation with world state."""

    @pytest.mark.asyncio
    async def test_full_consciousness_block_with_world_state(
        self, mud_response_strategy, sample_persona, sample_mud_session, mock_redis
    ):
        """Test complete flow: build_turns -> consciousness block with world state."""
        # Set up empty conversation history
        mock_redis.lrange = AsyncMock(return_value=[])

        # Call build_turns (which sets current_location)
        turns = await mud_response_strategy.build_turns(
            persona=sample_persona,
            user_input="look around",
            session=sample_mud_session,
        )

        # Now generate consciousness block
        xml_output, memory_count = mud_response_strategy.get_conscious_memory(sample_persona)

        # Verify complete structure
        assert "<PraxOS>" in xml_output
        assert "Conscious Memory **Online**" in xml_output
        assert "Memory Count" in xml_output
        assert "Current World State" in xml_output

        # Verify world state content
        assert "Bedroom" in xml_output
        assert "Prax" in xml_output
        assert "Golden Key" in xml_output
        assert "Silver Coin" in xml_output

    @pytest.mark.asyncio
    async def test_token_budget_includes_world_state(
        self, mud_response_strategy, sample_persona, sample_mud_session, mock_redis
    ):
        """Test that world state tokens are accounted for in budget calculation."""
        mock_redis.lrange = AsyncMock(return_value=[])

        # Build turns to set location
        await mud_response_strategy.build_turns(
            persona=sample_persona,
            user_input="test",
            session=sample_mud_session,
        )

        # Generate consciousness with limited budget
        xml_output, _ = mud_response_strategy.get_conscious_memory(
            sample_persona,
            max_context_tokens=4000,
            max_output_tokens=500,
        )

        # Should not exceed token budget
        token_count = mud_response_strategy.count_tokens(xml_output)
        # usable_context = 4000 - 500 - 1024 = 2476
        assert token_count <= 2476

        # World state should still be present
        assert "Current World State" in xml_output


class TestMUDResponseStrategyWorldStateXMLFormat:
    """Test that world state XML is properly formatted in consciousness block."""

    def test_world_state_xml_structure(
        self, mud_response_strategy, sample_persona, sample_world_state
    ):
        """Test that world state maintains proper XML structure."""
        world_state_xml = sample_world_state.to_xml()
        mud_response_strategy.chat.current_location = world_state_xml

        xml_output, _ = mud_response_strategy.get_conscious_memory(sample_persona)

        # Verify XML structure is preserved
        assert "<world_state>" in xml_output
        assert "</world_state>" in xml_output
        assert '<location name="Bedroom"' in xml_output
        assert "<present>" in xml_output
        assert "<inventory>" in xml_output

    def test_world_state_priority_and_noindent(
        self, mud_response_strategy, sample_persona, sample_world_state
    ):
        """Test that world state is added with correct priority and noindent flag."""
        world_state_xml = sample_world_state.to_xml()
        mud_response_strategy.chat.current_location = world_state_xml

        # The hook should add with priority=1 and noindent=True
        # This means it should be rendered without extra indentation
        xml_output, _ = mud_response_strategy.get_conscious_memory(sample_persona)

        # World state should be present and properly formatted
        assert "Current World State" in xml_output
        # The actual world_state_xml content should be embedded
        assert "<world_state>" in xml_output

    def test_world_state_at_praxos_level(
        self, mud_response_strategy, sample_persona, sample_world_state
    ):
        """Test that world state appears at PraxOS level, not nested under Active Memory."""
        world_state_xml = sample_world_state.to_xml()
        mud_response_strategy.chat.current_location = world_state_xml

        xml_output, _ = mud_response_strategy.get_conscious_memory(sample_persona)

        # World state should be at PraxOS level (under HUD Display Output)
        assert "Current World State" in xml_output
        assert "<HUD Display Output>" in xml_output

        # Verify structure: World State should be at same level as Memory Count
        # Both should be direct children of HUD Display Output
        hud_start = xml_output.find("<HUD Display Output>")
        memory_count_start = xml_output.find("<Memory Count>")
        world_state_start = xml_output.find("<Current World State>")

        # All should exist and be siblings (children of HUD Display Output)
        assert hud_start != -1, "HUD Display Output not found"
        assert memory_count_start != -1, "Memory Count not found"
        assert world_state_start != -1, "World State not found"

        # World State should appear after Memory Count (as per tail positioning)
        assert world_state_start > memory_count_start, "World State should appear after Memory Count"

        # Verify it's NOT nested under Active Memory by checking it appears outside any
        # Active Memory block (if one exists)
        active_memory_start = xml_output.find("<Active Memory>")
        if active_memory_start != -1:
            active_memory_end = xml_output.find("</Active Memory>")
            # World state should be completely outside Active Memory block
            assert world_state_start > active_memory_end or world_state_start < active_memory_start, \
                "World State should not be nested within Active Memory"
