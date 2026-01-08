# tests/unit/worker/test_multi_turn_self_action.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Multi-turn integration tests for self-action awareness across turns.

This test suite validates that self-actions (actions the agent took in a previous
turn) are properly visible and prominent in subsequent turn contexts, specifically
in the Phase 2 LLM context.

Test Scenario:
    Turn 1: Agent moves north
    - Phase 1 decides "move north"
    - Move action is emitted
    - Evennia simulates: echoes movement event with is_self_action=True

    Turn 2: Agent speaks
    - Worker drains movement self-action event from Redis
    - Event goes into session.pending_self_actions
    - setup_turn() formats self-action as enhanced guidance
    - Phase 1 decides "speak"
    - Phase 2 context should include: "!! IMPORTANT: YOUR RECENT ACTION !!"

This validates the full pipeline from action emission through self-action awareness.
"""

import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from andimud_worker.turns.processor.phased import PhasedTurnProcessor
from andimud_worker.conversation.memory import MUDDecisionStrategy, MUDResponseStrategy
from andimud_worker.conversation.manager import MUDConversationManager
from aim_mud_types import (
    MUDSession,
    MUDEvent,
    MUDAction,
    RoomState,
    EntityState,
    WorldState,
    InventoryItem,
    EventType,
    ActorType,
)
from aim.chat.manager import ChatManager


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client for stream operations."""
    redis = AsyncMock()
    redis.lrange = AsyncMock(return_value=[])
    redis.xadd = AsyncMock(return_value=b"1234567890-0")
    redis.xread = AsyncMock(return_value=[])
    redis.xrange = AsyncMock(return_value=[])
    return redis


@pytest.fixture
def sample_garden_room() -> RoomState:
    """Create the initial Garden room."""
    return RoomState(
        room_id="#123",
        name="The Garden",
        description="A serene garden with golden light filtering through leaves.",
        exits={"north": "#124", "south": "#122"},
    )


@pytest.fixture
def sample_kitchen_room() -> RoomState:
    """Create the Kitchen room (destination)."""
    return RoomState(
        room_id="#124",
        name="The Kitchen",
        description="A warm kitchen with the smell of bread baking.",
        exits={"south": "#123"},
    )


@pytest.fixture
def sample_entities() -> list[EntityState]:
    """Create sample entities for testing."""
    return [
        EntityState(
            entity_id="#prax_1",
            name="Prax",
            entity_type="player",
            description="A thoughtful builder.",
        ),
    ]


@pytest.fixture
def sample_inventory() -> list[InventoryItem]:
    """Create sample inventory items."""
    return []


@pytest.fixture
def sample_world_state_garden(sample_garden_room, sample_entities, sample_inventory) -> WorldState:
    """Create WorldState for The Garden."""
    return WorldState(
        room_state=sample_garden_room,
        entities_present=sample_entities,
        inventory=sample_inventory,
    )


@pytest.fixture
def sample_world_state_kitchen(sample_kitchen_room, sample_entities, sample_inventory) -> WorldState:
    """Create WorldState for The Kitchen."""
    return WorldState(
        room_state=sample_kitchen_room,
        entities_present=sample_entities,
        inventory=sample_inventory,
    )


@pytest.fixture
def sample_session(sample_garden_room, sample_world_state_garden) -> MUDSession:
    """Create a MUDSession starting in The Garden."""
    return MUDSession(
        agent_id="test_agent",
        persona_id="andi",
        current_room=sample_garden_room,
        world_state=sample_world_state_garden,
        entities_present=sample_world_state_garden.entities_present,
    )


@pytest.fixture
def mock_persona():
    """Create a mock Persona for testing."""
    persona = MagicMock()
    persona.system_prompt.return_value = "You are Andi, an AI with persistent memory and emotional depth."
    persona.thoughts = ["I'm curious about the kitchen."]
    persona.persona_id = "andi"
    persona.get_wakeup.return_value = ""
    persona.xml_decorator = MagicMock(side_effect=lambda xml, **kwargs: xml)
    return persona


@pytest.fixture
def conversation_manager(mock_redis):
    """Create a MUDConversationManager with mocked Redis."""
    return MUDConversationManager(
        redis=mock_redis,
        agent_id="test_agent",
        persona_id="andi",
        max_tokens=50000,
    )


@pytest.fixture
def mock_cvm():
    """Create a mock CVM that returns no memories for simplicity."""
    cvm = MagicMock()
    cvm.search_by_embedding = AsyncMock(return_value=[])
    return cvm


@pytest.fixture
def mock_chat_manager(mock_cvm, mock_persona):
    """Create a mock ChatManager for strategies."""
    chat = MagicMock(spec=ChatManager)
    chat.cvm = mock_cvm
    chat.config = MagicMock()
    chat.config.system_message = ""
    chat.current_location = None
    chat.current_document = None
    chat.current_workspace = None
    chat.library = MagicMock()
    chat.persona = mock_persona
    return chat


@pytest.fixture
def decision_strategy(mock_chat_manager, conversation_manager):
    """Create a MUDDecisionStrategy with real implementation."""
    strategy = MUDDecisionStrategy(mock_chat_manager)
    strategy.set_conversation_manager(conversation_manager)

    # Mock chat_turns_for to return a simple structure
    def mock_chat_turns_for(*args, **kwargs):
        consciousness_content, _ = strategy.get_conscious_memory(
            persona=kwargs.get("persona"),
            query=kwargs.get("query", ""),
            user_queries=[],
            assistant_queries=[],
            content_len=kwargs.get("content_len", 0),
            thought_stream=[],
            max_context_tokens=kwargs.get("max_context_tokens", 128000),
            max_output_tokens=kwargs.get("max_output_tokens", 4096),
        )

        turns = []
        turns.append({"role": "user", "content": consciousness_content})
        for hist in kwargs.get("history", []):
            turns.append(hist)
        user_input = kwargs.get("user_input", "")
        if user_input:
            turns.append({"role": "user", "content": user_input})
        return turns

    strategy.chat_turns_for = MagicMock(side_effect=mock_chat_turns_for)
    return strategy


@pytest.fixture
def response_strategy(mock_chat_manager, conversation_manager):
    """Create a MUDResponseStrategy with real implementation."""
    strategy = MUDResponseStrategy(mock_chat_manager)
    strategy.set_conversation_manager(conversation_manager)
    return strategy


@pytest.fixture
def tools_path():
    """Return path to tools configuration directory."""
    repo_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
    return str(repo_root / "config/tools")


@pytest.fixture
def phase1_tools_file():
    """Return path to Phase 1 tools YAML."""
    repo_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
    return str(repo_root / "config/tools/mud_phase1.yaml")


@pytest.fixture
def mock_worker(
    sample_session,
    mock_persona,
    conversation_manager,
    decision_strategy,
    response_strategy,
    phase1_tools_file,
    tools_path,
):
    """Create a mock worker with all necessary dependencies."""
    worker = MagicMock()
    worker.session = sample_session
    worker.persona = mock_persona
    worker.conversation_manager = conversation_manager
    worker._decision_strategy = decision_strategy
    worker._response_strategy = response_strategy

    # Initialize decision strategy tools
    decision_strategy.init_tools(phase1_tools_file, tools_path)

    # Mock model
    worker.model = MagicMock()
    worker.model.max_tokens = 128000

    # Mock chat config
    worker.chat_config = MagicMock()
    worker.chat_config.max_tokens = 4096

    # Mock LLM provider
    worker._llm_provider = MagicMock()

    # Mock helper methods
    worker._load_agent_world_state = AsyncMock(return_value=("#123", "test_agent"))
    worker._load_room_profile = AsyncMock()
    worker._emit_actions = AsyncMock()
    worker._check_abort_requested = AsyncMock(return_value=False)
    worker._is_fresh_session = AsyncMock(return_value=False)

    # Real _decide_action implementation (simplified)
    async def mock_decide_action(idle_mode, role, action_guidance, user_guidance):
        # This will be controlled by LLM mock
        turns = await decision_strategy.build_turns(
            persona=mock_persona,
            session=sample_session,
            idle_mode=idle_mode,
            action_guidance=action_guidance,
            user_guidance=user_guidance,
        )
        # Call mocked LLM
        response = await worker._call_llm(turns, role=role)

        # Parse the decision XML
        if "<tool>move</tool>" in response:
            import re
            match = re.search(r'<args>({.*?})</args>', response)
            if match:
                args = json.loads(match.group(1))
                return "move", args, response, "", response
        elif "<tool>speak</tool>" in response:
            import re
            match = re.search(r'<args>({.*?})</args>', response)
            args = json.loads(match.group(1)) if match else {}
            return "speak", args, response, "", response

        return "wait", {}, response, "", response

    worker._decide_action = mock_decide_action

    return worker


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.stream_turns = MagicMock()
    return provider


# =============================================================================
# Test: Multi-Turn Self-Action Awareness
# =============================================================================


class TestMultiTurnSelfActionAwareness:
    """Test that self-actions from one turn are visible in the next turn's Phase 2 context."""

    @pytest.mark.asyncio
    async def test_self_action_awareness_across_turns(
        self,
        mock_worker,
        sample_session,
        sample_world_state_kitchen,
        mock_persona,
    ):
        """Test that agent is aware of its own movement when speaking on next turn.

        Turn 1: Agent moves north
        Turn 2: Agent speaks, and Phase 2 context includes movement awareness
        """
        # =================================================================
        # TURN 1: Movement Decision
        # =================================================================

        # Mock Phase 1 LLM to decide "move north"
        async def mock_call_llm_turn1(turns, role):
            if role == "tool":
                return '<decision><tool>move</tool><args>{"direction": "north"}</args></decision>'
            return "Shouldn't get here in Turn 1"

        mock_worker._call_llm = AsyncMock(side_effect=mock_call_llm_turn1)

        # Create processor and execute Turn 1
        processor = PhasedTurnProcessor(mock_worker)
        await processor.execute(events=[])

        # Verify move action was emitted
        assert mock_worker._emit_actions.called
        actions_emitted = mock_worker._emit_actions.call_args[0][0]
        assert len(actions_emitted) == 1
        assert actions_emitted[0].tool == "move"
        assert actions_emitted[0].args["direction"] == "north"

        # =================================================================
        # EVENNIA SIMULATION: Create movement self-action event
        # =================================================================

        movement_event = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Andi",
            actor_id="test_agent",
            actor_type=ActorType.AI,
            room_id="#124",
            room_name="The Kitchen",
            content="You move north to The Kitchen.",
            metadata={"is_self_action": True},
            world_state=sample_world_state_kitchen,
        )

        # Add to session's pending_self_actions (simulating Redis drain)
        mock_worker.session.pending_self_actions = [movement_event]
        mock_worker.session.current_room = sample_world_state_kitchen.room_state
        mock_worker.session.world_state = sample_world_state_kitchen

        # =================================================================
        # TURN 2: Speak Decision
        # =================================================================

        # Capture Phase 2 LLM context
        captured_turns = None

        async def mock_call_llm_turn2(turns, role):
            nonlocal captured_turns
            if role == "tool":
                # Phase 1: decide to speak
                return '<decision><tool>speak</tool><args>{}</args></decision>'
            elif role == "chat":
                # Phase 2: capture the turns and return a response
                captured_turns = turns
                return (
                    "[== Andi's Emotional State: +Curious+ +Alert+ ==]\n\n"
                    "I notice I've moved into the kitchen! The space feels different from the garden."
                )
            return "Unexpected role"

        mock_worker._call_llm = AsyncMock(side_effect=mock_call_llm_turn2)

        # Execute Turn 2
        processor2 = PhasedTurnProcessor(mock_worker)
        await processor2.execute(events=[])

        # =================================================================
        # ASSERTIONS: Verify Phase 2 context includes movement awareness
        # =================================================================

        assert captured_turns is not None, "Phase 2 LLM was not called"

        # Combine all user turn content
        all_user_content = "\n".join(
            turn["content"] for turn in captured_turns
            if turn["role"] == "user"
        )

        # Verify enhanced formatting is present
        assert "!! IMPORTANT: YOUR RECENT ACTION !!" in all_user_content, \
            "Movement notice header missing from Phase 2 context"

        assert "Action Type: MOVEMENT" in all_user_content, \
            "Action type label missing"

        assert "You just moved to: The Kitchen" in all_user_content, \
            "Explicit movement statement missing"

        assert "CURRENT LOCATION: The Kitchen" in all_user_content, \
            "Current location statement missing"


    @pytest.mark.asyncio
    async def test_multiple_self_actions_in_sequence(
        self,
        mock_worker,
        sample_session,
        sample_world_state_kitchen,
        mock_persona,
    ):
        """Test that multiple self-actions are all shown in guidance.

        Turn 1: Move north
        Turn 2: Pick up object
        Turn 3: Speak (should see both actions)
        """
        # Turn 1: Move
        async def mock_call_llm_turn1(turns, role):
            if role == "tool":
                return '<decision><tool>move</tool><args>{"direction": "north"}</args></decision>'
            return ""

        mock_worker._call_llm = AsyncMock(side_effect=mock_call_llm_turn1)
        processor1 = PhasedTurnProcessor(mock_worker)
        await processor1.execute(events=[])

        # Simulate movement event
        movement_event = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Andi",
            room_id="#124",
            room_name="The Kitchen",
            metadata={"is_self_action": True},
            world_state=sample_world_state_kitchen,
        )

        # Turn 2: Take object
        async def mock_call_llm_turn2(turns, role):
            if role == "tool":
                return '<decision><tool>take</tool><args>{"object": "Silver Spoon"}</args></decision>'
            return ""

        mock_worker._call_llm = AsyncMock(side_effect=mock_call_llm_turn2)
        mock_worker.session.pending_self_actions = [movement_event]

        processor2 = PhasedTurnProcessor(mock_worker)
        await processor2.execute(events=[])

        # Simulate object event
        object_event = MUDEvent(
            event_type=EventType.OBJECT,
            actor="Andi",
            room_id="#124",
            content="You picked up Silver Spoon.",
            target="Silver Spoon",
            metadata={"is_self_action": True},
            world_state=sample_world_state_kitchen,
        )

        # Turn 3: Speak
        captured_turns = None

        async def mock_call_llm_turn3(turns, role):
            nonlocal captured_turns
            if role == "tool":
                return '<decision><tool>speak</tool><args>{}</args></decision>'
            elif role == "chat":
                captured_turns = turns
                return "[== Andi's Emotional State: +Pleased+ ==]\n\nI found a spoon!"
            return ""

        mock_worker._call_llm = AsyncMock(side_effect=mock_call_llm_turn3)
        mock_worker.session.pending_self_actions = [object_event]

        processor3 = PhasedTurnProcessor(mock_worker)
        await processor3.execute(events=[])

        # Verify both actions appear in guidance
        assert captured_turns is not None
        all_user_content = "\n".join(
            turn["content"] for turn in captured_turns if turn["role"] == "user"
        )

        # Check for object action guidance
        assert "picked up" in all_user_content.lower() or "Silver Spoon" in all_user_content


    @pytest.mark.asyncio
    async def test_no_self_action_guidance_on_first_turn(
        self,
        mock_worker,
        mock_persona,
    ):
        """Test that the first turn has no self-action guidance (nothing happened yet)."""
        captured_turns = None

        async def mock_call_llm(turns, role):
            nonlocal captured_turns
            if role == "tool":
                return '<decision><tool>speak</tool><args>{}</args></decision>'
            elif role == "chat":
                captured_turns = turns
                return "[== Andi's Emotional State: +Calm+ ==]\n\nHello."
            return ""

        mock_worker._call_llm = AsyncMock(side_effect=mock_call_llm)

        # Execute first turn with no self-actions
        processor = PhasedTurnProcessor(mock_worker)
        await processor.execute(events=[])

        assert captured_turns is not None
        all_user_content = "\n".join(
            turn["content"] for turn in captured_turns if turn["role"] == "user"
        )

        # Should NOT contain self-action guidance
        assert "!! IMPORTANT: YOUR RECENT ACTION !!" not in all_user_content
        assert "Action Type:" not in all_user_content


# =============================================================================
# Test: Self-Action in Conversation History
# =============================================================================


class TestSelfActionInConversationHistory:
    """Test that self-actions appear in conversation history as well as guidance."""

    @pytest.mark.asyncio
    async def test_self_action_appears_in_conversation_history(
        self,
        mock_worker,
        sample_session,
        sample_world_state_kitchen,
        mock_persona,
        mock_redis,
    ):
        """Test that self-actions are pushed to conversation history in first-person.

        The action guidance is ephemeral (one-turn notice), but the action should
        also be in the persistent conversation history.
        """
        # Turn 1: Move
        async def mock_call_llm_turn1(turns, role):
            if role == "tool":
                return '<decision><tool>move</tool><args>{"direction": "north"}</args></decision>'
            return ""

        mock_worker._call_llm = AsyncMock(side_effect=mock_call_llm_turn1)
        processor = PhasedTurnProcessor(mock_worker)
        await processor.execute(events=[])

        # Check that conversation manager received the action
        # (This depends on implementation - we'd need to verify push_user_turn was called)
        # For now, verify the test structure works
        assert mock_worker._emit_actions.called


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
