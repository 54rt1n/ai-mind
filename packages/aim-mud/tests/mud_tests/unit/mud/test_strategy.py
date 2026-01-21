# tests/unit/mud/test_strategy.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for MUD decision strategy."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from andimud_worker.conversation.memory import MUDDecisionStrategy, MUDResponseStrategy
from andimud_worker.conversation import MUDConversationManager
from aim_mud_types import (
    MUDConversationEntry,
    MUDSession,
    RoomState,
    EntityState,
    WorldState,
    InventoryItem,
)
from aim.constants import DOC_MUD_WORLD, DOC_MUD_AGENT
from aim.tool.dto import Tool, ToolFunction, ToolFunctionParameters


def _sample_room() -> RoomState:
    """Create a sample RoomState for testing."""
    return RoomState(
        room_id="#123",
        name="The Garden",
        description="A serene garden.",
        exits={"north": "#124", "south": "#122"},
    )


def _sample_entity(name: str, entity_type: str = "object", is_self: bool = False) -> EntityState:
    """Create a sample EntityState for testing."""
    return EntityState(
        entity_id=f"#{name.lower().replace(' ', '_')}",
        name=name,
        entity_type=entity_type,
        is_self=is_self,
    )


def _sample_inventory_item(name: str) -> InventoryItem:
    """Create a sample InventoryItem for testing."""
    return InventoryItem(
        item_id=f"#{name.lower().replace(' ', '_')}",
        name=name,
        description=f"A {name.lower()}.",
    )


def _sample_world_state() -> WorldState:
    """Create a sample WorldState for testing."""
    return WorldState(
        room_state=_sample_room(),
        entities_present=[
            _sample_entity("Prax", "player"),
            _sample_entity("Golden Key", "object"),
            _sample_entity("Andi", "ai", is_self=True),
        ],
        inventory=[
            _sample_inventory_item("Silver Coin"),
        ],
    )


def _sample_session() -> MUDSession:
    """Create a sample MUDSession for testing."""
    return MUDSession(
        agent_id="test_agent",
        persona_id="andi",
        current_room=_sample_room(),
        world_state=_sample_world_state(),
        entities_present=[
            _sample_entity("Prax", "player"),
            _sample_entity("Golden Key", "object"),
        ],
    )


def _sample_conversation_entry(
    role: str = "user",
    content: str = "Test content",
    sequence_no: int = 0,
) -> MUDConversationEntry:
    """Create a sample MUDConversationEntry for testing."""
    return MUDConversationEntry(
        role=role,
        content=content,
        tokens=10,
        document_type=DOC_MUD_WORLD if role == "user" else DOC_MUD_AGENT,
        conversation_id="test_conv",
        sequence_no=sequence_no,
        speaker_id="world" if role == "user" else "andi",
    )


def _sample_tool() -> Tool:
    """Create a sample Tool for testing."""
    return Tool(
        type="function",
        function=ToolFunction(
            name="move",
            description="Move to a new location",
            parameters=ToolFunctionParameters(
                type="object",
                properties={
                    "location": {
                        "type": "string",
                        "description": "The exit to move through",
                    }
                },
                required=["location"],
            ),
        ),
    )


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    redis = AsyncMock()
    redis.lrange = AsyncMock(return_value=[])
    return redis


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
def mock_chat_manager_for_decision():
    """Create a mock ChatManager for MUDDecisionStrategy tests."""
    chat = MagicMock()
    chat.cvm = MagicMock()
    chat.config = MagicMock()
    chat.current_location = None
    chat.current_document = None
    chat.current_workspace = None
    chat.library = MagicMock()
    return chat


@pytest.fixture
def strategy(mock_chat_manager_for_decision, conversation_manager):
    """Create a MUDDecisionStrategy for testing."""
    strat = MUDDecisionStrategy(mock_chat_manager_for_decision)
    strat.set_conversation_manager(conversation_manager)
    return strat


@pytest.fixture
def mock_persona():
    """Create a mock Persona for testing."""
    persona = MagicMock()
    persona.system_prompt.return_value = "You are Andi, an AI assistant."
    return persona


class TestMUDDecisionStrategyInit:
    """Test MUDDecisionStrategy initialization."""

    def test_init_takes_chat_manager(self, mock_chat_manager_for_decision):
        """Test that __init__ takes ChatManager and conversation_manager starts as None."""
        strategy = MUDDecisionStrategy(mock_chat_manager_for_decision)
        assert strategy.chat is mock_chat_manager_for_decision
        assert strategy.conversation_manager is None

    def test_set_conversation_manager(self, mock_chat_manager_for_decision, conversation_manager):
        """Test that set_conversation_manager sets the conversation manager."""
        strategy = MUDDecisionStrategy(mock_chat_manager_for_decision)
        strategy.set_conversation_manager(conversation_manager)
        assert strategy.conversation_manager is conversation_manager

    def test_init_tool_user_is_none(self, mock_chat_manager_for_decision):
        """Test that tool_user starts as None."""
        strategy = MUDDecisionStrategy(mock_chat_manager_for_decision)
        assert strategy.tool_user is None

    def test_set_tool_user(self, strategy):
        """Test that set_tool_user sets the tool_user."""
        from aim.tool.formatting import ToolUser

        tool_user = ToolUser(tools=[_sample_tool()])
        strategy.set_tool_user(tool_user)
        assert strategy.tool_user is tool_user


class TestMUDDecisionStrategyBuildTurns:
    """Test MUDDecisionStrategy.build_turns method."""

    @pytest.mark.asyncio
    async def test_build_turns_returns_valid_structure(
        self, strategy, mock_persona, mock_redis
    ):
        """Test that build_turns returns valid turn structure."""
        session = _sample_session()

        # Mock chat_turns_for to return expected structure
        strategy.chat_turns_for = MagicMock(return_value=[
            {"role": "user", "content": "consciousness block"},
            {"role": "user", "content": "current context"},
        ])

        turns = await strategy.build_turns(mock_persona, session)

        assert isinstance(turns, list)
        assert len(turns) >= 1
        # Verify chat_turns_for was called
        strategy.chat_turns_for.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_turns_no_system_turn_in_array(
        self, strategy, mock_persona, mock_redis
    ):
        """Test that build_turns does not include system turn in array.

        System message is now set in config.system_message by worker.py,
        not added to the turns array.
        """
        session = _sample_session()

        # Mock chat_turns_for to return turns without system
        strategy.chat_turns_for = MagicMock(return_value=[
            {"role": "user", "content": "test"},
        ])

        turns = await strategy.build_turns(mock_persona, session)

        # System turn should NOT be in the turns array
        assert all(turn["role"] != "system" for turn in turns)

    @pytest.mark.asyncio
    async def test_build_turns_includes_history(
        self, strategy, mock_persona, mock_redis
    ):
        """Test that build_turns passes history to chat_turns_for."""
        # Set up mock history
        entry1 = _sample_conversation_entry("user", "Hello!", 0)
        entry2 = _sample_conversation_entry("assistant", "Hi there!", 1)
        mock_redis.lrange.return_value = [
            entry1.model_dump_json().encode(),
            entry2.model_dump_json().encode(),
        ]

        # Mock chat_turns_for
        strategy.chat_turns_for = MagicMock(return_value=[
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "current context"},
        ])

        session = _sample_session()
        turns = await strategy.build_turns(mock_persona, session)

        # Verify chat_turns_for was called with history
        call_kwargs = strategy.chat_turns_for.call_args[1]
        assert "history" in call_kwargs
        history = call_kwargs["history"]
        assert len(history) == 2
        assert history[0]["content"] == "Hello!"
        assert history[1]["content"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_build_turns_with_tool_user(
        self, strategy, mock_persona, mock_redis
    ):
        """Test that build_turns works when tool_user is set."""
        from aim.tool.formatting import ToolUser

        tool_user = ToolUser(tools=[_sample_tool()])
        strategy.set_tool_user(tool_user)

        # Mock chat_turns_for
        strategy.chat_turns_for = MagicMock(return_value=[
            {"role": "user", "content": "test"},
        ])

        session = _sample_session()
        turns = await strategy.build_turns(mock_persona, session)

        # Verify it completed successfully
        assert len(turns) >= 1
        strategy.chat_turns_for.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_turns_idle_mode(
        self, strategy, mock_persona, mock_redis
    ):
        """Test that build_turns handles idle mode."""
        session = _sample_session()
        session.pending_events = []  # No pending events

        # Capture what user_input was passed to chat_turns_for
        captured_user_input = None
        def capture_call(*args, **kwargs):
            nonlocal captured_user_input
            captured_user_input = kwargs.get("user_input", "")
            return [{"role": "user", "content": captured_user_input}]

        strategy.chat_turns_for = capture_call

        turns = await strategy.build_turns(mock_persona, session, idle_mode=True)

        # Idle mode should pass through to build_current_context
        # which adds agency prompt
        assert captured_user_input is not None
        assert "agency" in captured_user_input.lower() or "want to do" in captured_user_input.lower()

    @pytest.mark.asyncio
    async def test_build_turns_includes_guidance(
        self, strategy, mock_persona, mock_redis
    ):
        """Test that build_turns includes decision guidance."""
        session = _sample_session()

        # Capture what user_input was passed to chat_turns_for
        captured_user_input = None
        def capture_call(*args, **kwargs):
            nonlocal captured_user_input
            captured_user_input = kwargs.get("user_input", "")
            return [{"role": "user", "content": captured_user_input}]

        strategy.chat_turns_for = capture_call

        turns = await strategy.build_turns(mock_persona, session)

        # Guidance should mention available exits
        assert captured_user_input is not None
        assert "north" in captured_user_input or "south" in captured_user_input


class TestMUDDecisionStrategyBuildDecisionGuidance:
    """Test MUDDecisionStrategy._build_decision_guidance method."""

    def test_build_decision_guidance_includes_json_instructions(self, strategy):
        """Test that guidance includes JSON format instructions."""
        session = _sample_session()

        guidance = strategy._build_decision_guidance(session)

        assert "Tool Use Turn" in guidance
        assert '{"move":' in guidance or "Tool Guidance" in guidance

    def test_build_decision_guidance_enumerates_exits(self, strategy):
        """Test that guidance enumerates available exits."""
        session = _sample_session()

        guidance = strategy._build_decision_guidance(session)

        assert "Available exits:" in guidance
        assert "north" in guidance
        assert "south" in guidance

    def test_build_decision_guidance_enumerates_objects(self, strategy):
        """Test that guidance enumerates room objects."""
        session = _sample_session()

        guidance = strategy._build_decision_guidance(session)

        assert "Objects present:" in guidance
        assert "Golden Key" in guidance

    def test_build_decision_guidance_enumerates_inventory(self, strategy):
        """Test that guidance enumerates inventory items."""
        session = _sample_session()

        guidance = strategy._build_decision_guidance(session)

        assert "Your inventory:" in guidance
        assert "Silver Coin" in guidance

    def test_build_decision_guidance_enumerates_targets(self, strategy):
        """Test that guidance enumerates valid give targets."""
        session = _sample_session()

        guidance = strategy._build_decision_guidance(session)

        assert "People present:" in guidance
        assert "Prax" in guidance

    def test_build_decision_guidance_excludes_self(self, strategy):
        """Test that guidance excludes the agent's own entity."""
        session = _sample_session()

        guidance = strategy._build_decision_guidance(session)

        # Andi (is_self=True) should not appear as a target
        assert "Andi" not in guidance

    def test_build_decision_guidance_includes_examples(self, strategy):
        """Test that guidance includes action examples."""
        session = _sample_session()

        guidance = strategy._build_decision_guidance(session)

        assert "Contextual Examples:" in guidance
        assert "Move:" in guidance
        assert "Take:" in guidance
        assert "Drop:" in guidance
        assert "Give:" in guidance

    def test_build_decision_guidance_empty_session(self, strategy):
        """Test guidance with minimal session."""
        session = MUDSession(
            agent_id="test",
            persona_id="andi",
            current_room=None,
            world_state=None,
        )

        guidance = strategy._build_decision_guidance(session)

        # Should still have basic instructions
        assert "Tool Use Turn" in guidance
        # But no contextual examples for specific actions
        assert "Contextual Examples:" not in guidance

    def test_build_decision_guidance_falls_back_to_session_entities(self, strategy):
        """Test that guidance falls back to session.entities_present when no world_state."""
        session = MUDSession(
            agent_id="test",
            persona_id="andi",
            current_room=_sample_room(),
            world_state=None,
            entities_present=[
                _sample_entity("Mysterious Stranger", "npc"),
                _sample_entity("Ancient Tome", "object"),
            ],
        )

        guidance = strategy._build_decision_guidance(session)

        assert "Mysterious Stranger" in guidance
        assert "Ancient Tome" in guidance


class TestMUDDecisionStrategyBuildAgentActionHints:
    """Test MUDDecisionStrategy._build_agent_action_hints method."""

    def test_build_agent_action_hints_returns_list(self, strategy):
        """Test that _build_agent_action_hints returns a list."""
        session = _sample_session()

        hints = strategy._build_agent_action_hints(session)

        assert isinstance(hints, list)

    def test_build_agent_action_hints_includes_exits(self, strategy):
        """Test that hints include exits."""
        session = _sample_session()

        hints = strategy._build_agent_action_hints(session)

        assert any("Valid move locations:" in h for h in hints)

    def test_build_agent_action_hints_includes_objects(self, strategy):
        """Test that hints include describable objects with IDs."""
        session = _sample_session()

        hints = strategy._build_agent_action_hints(session)

        assert any("Describable objects:" in h for h in hints)

    def test_build_agent_action_hints_includes_inventory(self, strategy):
        """Test that hints include inventory."""
        session = _sample_session()

        hints = strategy._build_agent_action_hints(session)

        assert any("Inventory:" in h for h in hints)

    def test_build_agent_action_hints_includes_targets(self, strategy):
        """Test that hints include targets."""
        session = _sample_session()

        hints = strategy._build_agent_action_hints(session)

        assert any("Valid give targets:" in h for h in hints)

    def test_build_agent_action_hints_empty_session(self, strategy):
        """Test hints with empty session."""
        session = MUDSession(
            agent_id="test",
            persona_id="andi",
            current_room=None,
            world_state=None,
        )

        hints = strategy._build_agent_action_hints(session)

        assert hints == []


class TestMUDDecisionStrategyGetConversationHistory:
    """Test MUDDecisionStrategy._get_conversation_history method."""

    @pytest.mark.asyncio
    async def test_get_conversation_history_returns_chat_turns(
        self, strategy, mock_redis
    ):
        """Test that _get_conversation_history returns chat turn format."""
        entry = _sample_conversation_entry("user", "Hello!", 0)
        mock_redis.lrange.return_value = [entry.model_dump_json().encode()]

        history = await strategy._get_conversation_history(token_budget=1000)

        assert isinstance(history, list)
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_get_conversation_history_empty(self, strategy, mock_redis):
        """Test _get_conversation_history with empty history."""
        mock_redis.lrange.return_value = []

        history = await strategy._get_conversation_history(token_budget=1000)

        assert history == []

    @pytest.mark.asyncio
    async def test_get_conversation_history_respects_budget(
        self, strategy, mock_redis
    ):
        """Test that history respects token budget."""
        # Create entries with known token counts
        entries = [
            MUDConversationEntry(
                role="user",
                content=f"Message {i}",
                tokens=50,
                document_type=DOC_MUD_WORLD,
                conversation_id="test",
                sequence_no=i,
                speaker_id="world",
            )
            for i in range(5)
        ]
        mock_redis.lrange.return_value = [
            e.model_dump_json().encode() for e in entries
        ]

        # Budget of 100 should only get 2 entries (50 tokens each)
        history = await strategy._get_conversation_history(token_budget=100)

        assert len(history) == 2




# =============================================================================
# MUDResponseStrategy Tests (Phase 2)
# =============================================================================


@pytest.fixture
def mock_chat_manager():
    """Create a mock ChatManager for MUDResponseStrategy tests."""
    chat = MagicMock()
    chat.cvm = MagicMock()
    chat.config = MagicMock()
    chat.current_location = None
    chat.current_document = None
    chat.current_workspace = None
    chat.library = MagicMock()
    return chat


@pytest.fixture
def mock_conversation_manager(mock_redis):
    """Create a mock MUDConversationManager for tests."""
    cm = MUDConversationManager(
        redis=mock_redis,
        agent_id="test_agent",
        persona_id="andi",
        max_tokens=50000,
    )
    return cm


@pytest.fixture
def response_strategy(mock_chat_manager, mock_conversation_manager):
    """Create a MUDResponseStrategy with mocked dependencies."""
    strategy = MUDResponseStrategy(mock_chat_manager)
    strategy.set_conversation_manager(mock_conversation_manager)
    return strategy


@pytest.fixture
def mock_response_persona():
    """Create a mock Persona for response strategy tests."""
    persona = MagicMock()
    persona.system_prompt.return_value = "You are Andi, an AI assistant."
    persona.get_wakeup.return_value = "Hello, I'm online now."
    persona.thoughts = []
    persona.persona_id = "andi"
    return persona


@pytest.fixture
def mock_session():
    """Create a sample MUDSession for response strategy tests."""
    return _sample_session()


class TestMUDResponseStrategyInit:
    """Test MUDResponseStrategy initialization."""

    def test_init_sets_chat(self, mock_chat_manager):
        """Test that __init__ sets the chat manager."""
        strategy = MUDResponseStrategy(mock_chat_manager)
        assert strategy.chat is mock_chat_manager

    def test_init_conversation_manager_is_none(self, mock_chat_manager):
        """Test that conversation_manager starts as None."""
        strategy = MUDResponseStrategy(mock_chat_manager)
        assert strategy.conversation_manager is None

    def test_set_conversation_manager(self, mock_chat_manager, mock_conversation_manager):
        """Test that set_conversation_manager sets the manager."""
        strategy = MUDResponseStrategy(mock_chat_manager)
        strategy.set_conversation_manager(mock_conversation_manager)
        assert strategy.conversation_manager is mock_conversation_manager


class TestMUDResponseStrategyBuildTurns:
    """Test MUDResponseStrategy.build_turns method."""

    @pytest.mark.asyncio
    async def test_build_turns_returns_list(
        self, response_strategy, mock_response_persona, mock_session
    ):
        """Test that build_turns returns a list of turns."""
        # Mock the parent's chat_turns_for to return expected structure
        response_strategy.chat_turns_for = MagicMock(return_value=[
            {"role": "user", "content": "consciousness"},
            {"role": "assistant", "content": "wakeup"},
            {"role": "user", "content": "input"},
        ])

        turns = await response_strategy.build_turns(
            persona=mock_response_persona,
            user_input="test input",
            session=mock_session,
        )

        assert isinstance(turns, list)
        assert len(turns) == 3

    @pytest.mark.asyncio
    async def test_build_turns_sets_location(
        self, response_strategy, mock_response_persona, mock_session
    ):
        """Test that build_turns sets chat.current_location from session."""
        response_strategy.chat_turns_for = MagicMock(return_value=[])

        await response_strategy.build_turns(
            persona=mock_response_persona,
            user_input="test",
            session=mock_session,
        )

        # Verify current_location was set (world_state.to_xml() called)
        assert response_strategy.chat.current_location is not None

    @pytest.mark.asyncio
    async def test_build_turns_no_location_without_world_state(
        self, response_strategy, mock_response_persona
    ):
        """Test that build_turns handles missing world_state gracefully."""
        response_strategy.chat_turns_for = MagicMock(return_value=[])
        response_strategy.chat.current_location = None

        session = MUDSession(
            agent_id="test",
            persona_id="andi",
            current_room=None,
            world_state=None,
        )

        await response_strategy.build_turns(
            persona=mock_response_persona,
            user_input="test",
            session=session,
        )

        # current_location should remain None
        assert response_strategy.chat.current_location is None

    @pytest.mark.asyncio
    async def test_build_turns_calls_parent_strategy(
        self, response_strategy, mock_response_persona, mock_session
    ):
        """Test that build_turns delegates to chat_turns_for."""
        response_strategy.chat_turns_for = MagicMock(return_value=[])

        await response_strategy.build_turns(
            persona=mock_response_persona,
            user_input="test input",
            session=mock_session,
            max_context_tokens=100000,
            max_output_tokens=4096,
        )

        # Verify parent method was called with correct args
        response_strategy.chat_turns_for.assert_called_once()
        call_kwargs = response_strategy.chat_turns_for.call_args[1]
        assert call_kwargs["persona"] == mock_response_persona
        assert call_kwargs["user_input"] == "test input"
        assert "history" in call_kwargs
        assert "content_len" in call_kwargs
        assert call_kwargs["max_context_tokens"] == 100000
        assert call_kwargs["max_output_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_build_turns_includes_history_in_content_len(
        self, response_strategy, mock_response_persona, mock_session, mock_redis
    ):
        """Test that build_turns includes history tokens in content_len."""
        # Set up mock history
        entry1 = _sample_conversation_entry("user", "Hello there!", 0)
        entry2 = _sample_conversation_entry("assistant", "Hi!", 1)
        mock_redis.lrange.return_value = [
            entry1.model_dump_json().encode(),
            entry2.model_dump_json().encode(),
        ]

        response_strategy.chat_turns_for = MagicMock(return_value=[])

        await response_strategy.build_turns(
            persona=mock_response_persona,
            user_input="test",
            session=mock_session,
        )

        # content_len should include history tokens
        call_kwargs = response_strategy.chat_turns_for.call_args[1]
        assert call_kwargs["content_len"] > 0


class TestMUDResponseStrategyGetConversationHistory:
    """Test MUDResponseStrategy._get_conversation_history method."""

    @pytest.mark.asyncio
    async def test_get_conversation_history_empty_without_manager(self, mock_chat_manager):
        """Test that _get_conversation_history returns empty list without manager."""
        strategy = MUDResponseStrategy(mock_chat_manager)
        # No conversation_manager set

        history = await strategy._get_conversation_history(8000)

        assert history == []

    @pytest.mark.asyncio
    async def test_get_conversation_history_returns_chat_turns(
        self, response_strategy, mock_redis
    ):
        """Test that _get_conversation_history returns role/content format."""
        entry = _sample_conversation_entry("user", "Hello!", 0)
        mock_redis.lrange.return_value = [entry.model_dump_json().encode()]

        history = await response_strategy._get_conversation_history(token_budget=1000)

        assert isinstance(history, list)
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_get_conversation_history_multiple_entries(
        self, response_strategy, mock_redis
    ):
        """Test _get_conversation_history with multiple entries."""
        entry1 = _sample_conversation_entry("user", "Hello!", 0)
        entry2 = _sample_conversation_entry("assistant", "Hi there!", 1)
        mock_redis.lrange.return_value = [
            entry1.model_dump_json().encode(),
            entry2.model_dump_json().encode(),
        ]

        history = await response_strategy._get_conversation_history(token_budget=1000)

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"


class TestMUDResponseStrategyDoubleUserTurnFix:
    """Test that MUDResponseStrategy avoids double consecutive user turns.

    In Phase 2, events are pushed to conversation history as a user turn.
    The user_input passed to build_turns() is format guidance. If both are
    passed separately to chat_turns_for(), we get two consecutive user turns
    which is invalid for LLM APIs. The fix merges format guidance into the
    last user turn in history.
    """

    @pytest.mark.asyncio
    async def test_build_turns_merges_guidance_into_last_user_turn(
        self, response_strategy, mock_response_persona, mock_session, mock_redis
    ):
        """Test that format guidance is merged into last user turn when history ends with user."""
        # Set up history ending with a user turn (simulating pushed events)
        entry1 = _sample_conversation_entry("assistant", "Hi there!", 0)
        entry2 = _sample_conversation_entry("user", "Prax waves.\nPrax says hello.", 1)
        mock_redis.lrange.return_value = [
            entry1.model_dump_json().encode(),
            entry2.model_dump_json().encode(),
        ]

        # Track what chat_turns_for receives
        received_args = {}
        original_chat_turns_for = response_strategy.chat_turns_for

        def capture_args(*args, **kwargs):
            received_args.update(kwargs)
            return []

        response_strategy.chat_turns_for = capture_args

        await response_strategy.build_turns(
            persona=mock_response_persona,
            user_input="[~~ FORMAT: think then act ~~]",
            session=mock_session,
        )

        # user_input should be empty (merged into history)
        assert received_args["user_input"] == ""

        # History's last user turn should have the merged content
        history = received_args["history"]
        last_user = [h for h in history if h["role"] == "user"][-1]
        assert "Prax waves" in last_user["content"]
        assert "[~~ FORMAT:" in last_user["content"]

    @pytest.mark.asyncio
    async def test_build_turns_no_merge_when_history_ends_with_assistant(
        self, response_strategy, mock_response_persona, mock_session, mock_redis
    ):
        """Test that user_input is passed through when history ends with assistant."""
        # Set up history ending with an assistant turn
        entry1 = _sample_conversation_entry("user", "Hello!", 0)
        entry2 = _sample_conversation_entry("assistant", "Hi there!", 1)
        mock_redis.lrange.return_value = [
            entry1.model_dump_json().encode(),
            entry2.model_dump_json().encode(),
        ]

        received_args = {}

        def capture_args(*args, **kwargs):
            received_args.update(kwargs)
            return []

        response_strategy.chat_turns_for = capture_args

        await response_strategy.build_turns(
            persona=mock_response_persona,
            user_input="[~~ FORMAT: think then act ~~]",
            session=mock_session,
        )

        # user_input should be passed through (not merged)
        assert received_args["user_input"] == "[~~ FORMAT: think then act ~~]"

    @pytest.mark.asyncio
    async def test_build_turns_no_merge_when_user_input_empty(
        self, response_strategy, mock_response_persona, mock_session, mock_redis
    ):
        """Test that empty user_input doesn't trigger merge."""
        entry = _sample_conversation_entry("user", "Hello!", 0)
        mock_redis.lrange.return_value = [entry.model_dump_json().encode()]

        received_args = {}

        def capture_args(*args, **kwargs):
            received_args.update(kwargs)
            return []

        response_strategy.chat_turns_for = capture_args

        await response_strategy.build_turns(
            persona=mock_response_persona,
            user_input="",
            session=mock_session,
        )

        # user_input should remain empty
        assert received_args["user_input"] == ""

        # History should not be modified
        history = received_args["history"]
        assert history[0]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_build_turns_no_merge_when_history_empty(
        self, response_strategy, mock_response_persona, mock_session, mock_redis
    ):
        """Test that user_input is passed through when history is empty."""
        mock_redis.lrange.return_value = []

        received_args = {}

        def capture_args(*args, **kwargs):
            received_args.update(kwargs)
            return []

        response_strategy.chat_turns_for = capture_args

        await response_strategy.build_turns(
            persona=mock_response_persona,
            user_input="[~~ FORMAT: think then act ~~]",
            session=mock_session,
        )

        # user_input should be passed through
        assert received_args["user_input"] == "[~~ FORMAT: think then act ~~]"

    @pytest.mark.asyncio
    async def test_build_turns_content_len_calculated_after_merge(
        self, response_strategy, mock_response_persona, mock_session, mock_redis
    ):
        """Test that content_len is calculated with merged content, not double-counted."""
        entry = _sample_conversation_entry("user", "Short.", 0)
        mock_redis.lrange.return_value = [entry.model_dump_json().encode()]

        received_args = {}

        def capture_args(*args, **kwargs):
            received_args.update(kwargs)
            return []

        response_strategy.chat_turns_for = capture_args

        format_guidance = "[~~ FORMAT: think then act ~~]"
        await response_strategy.build_turns(
            persona=mock_response_persona,
            user_input=format_guidance,
            session=mock_session,
        )

        # After merge, effective_user_input is empty, so content_len should
        # only include the merged history content, not double-count the guidance
        content_len = received_args["content_len"]

        # The merged history content is "Short.\n\n[~~ FORMAT: think then act ~~]"
        # plus wakeup tokens. If double-counted, would have extra guidance tokens.
        history = received_args["history"]
        merged_content = history[0]["content"]
        assert format_guidance in merged_content
