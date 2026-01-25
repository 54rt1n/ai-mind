# tests/unit/conversation/code/test_code_strategies.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for CODE_RAG worker strategies.

Tests CodeDecisionStrategy and CodeResponseStrategy for code-focused agents.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from andimud_worker.conversation.code import CodeDecisionStrategy, CodeResponseStrategy
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


def _sample_room() -> RoomState:
    """Create a sample RoomState for testing."""
    return RoomState(
        room_id="#123",
        name="Code Room",
        description="A room filled with source code.",
        exits={"north": "#124", "south": "#122"},
    )


def _sample_entity(
    name: str, entity_type: str = "object", is_self: bool = False
) -> EntityState:
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
            _sample_entity("config.py", "object"),
            _sample_entity("Blip", "ai", is_self=True),
        ],
        inventory=[
            _sample_inventory_item("worker.py"),
        ],
    )


def _sample_session() -> MUDSession:
    """Create a sample MUDSession for testing."""
    return MUDSession(
        agent_id="blip",
        persona_id="blip",
        current_room=_sample_room(),
        world_state=_sample_world_state(),
        entities_present=[
            _sample_entity("Prax", "player"),
            _sample_entity("config.py", "object"),
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
        speaker_id="world" if role == "user" else "blip",
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
        agent_id="blip",
        persona_id="blip",
        max_tokens=50000,
    )


@pytest.fixture
def mock_chat_manager():
    """Create a mock ChatManager for strategy tests."""
    chat = MagicMock()
    chat.cvm = MagicMock()
    chat.config = MagicMock()
    chat.current_location = None
    chat.current_document = None
    chat.current_workspace = None
    chat.library = MagicMock()
    return chat


@pytest.fixture
def mock_code_graph():
    """Create a mock CodeGraph."""
    graph = MagicMock()
    graph.calls = {}
    graph.callers = {}
    graph.get_neighborhood = MagicMock(return_value=set())
    return graph


@pytest.fixture
def decision_strategy(mock_chat_manager, conversation_manager, mock_code_graph):
    """Create a CodeDecisionStrategy for testing."""
    strat = CodeDecisionStrategy(mock_chat_manager)
    strat.set_conversation_manager(conversation_manager)
    strat.set_code_graph(mock_code_graph)
    return strat


@pytest.fixture
def response_strategy(mock_chat_manager, conversation_manager, mock_code_graph):
    """Create a CodeResponseStrategy for testing."""
    strat = CodeResponseStrategy(mock_chat_manager)
    strat.set_conversation_manager(conversation_manager)
    strat.set_code_graph(mock_code_graph)
    return strat


@pytest.fixture
def mock_persona():
    """Create a mock Persona for testing."""
    persona = MagicMock()
    persona.system_prompt.return_value = "You are Blip, a code agent."
    persona.get_wakeup.return_value = "Code analysis ready."
    persona.thoughts = []
    persona.persona_id = "blip"
    persona.xml_decorator = MagicMock(return_value=MagicMock(render=MagicMock(return_value="<persona/>")))
    return persona


# =============================================================================
# CodeDecisionStrategy Tests
# =============================================================================


class TestCodeDecisionStrategyInit:
    """Test CodeDecisionStrategy initialization."""

    def test_init_takes_chat_manager(self, mock_chat_manager):
        """Test that __init__ takes ChatManager."""
        strategy = CodeDecisionStrategy(mock_chat_manager)
        assert strategy.chat is mock_chat_manager
        assert strategy.conversation_manager is None

    def test_init_sets_interface_compatibility_attrs(self, mock_chat_manager):
        """Test that __init__ sets interface compatibility attributes."""
        strategy = CodeDecisionStrategy(mock_chat_manager)
        # These are accessed by mixins but not used for code agents
        assert strategy._active_plan is None
        assert strategy._redis_client is None
        assert strategy._agent_id is None
        assert strategy._plan_tool_impl is None
        assert strategy._emote_allowed is True
        assert strategy._workspace_active is False
        assert strategy.thought_content == ""

    def test_set_conversation_manager(self, mock_chat_manager, conversation_manager):
        """Test that set_conversation_manager sets the manager."""
        strategy = CodeDecisionStrategy(mock_chat_manager)
        strategy.set_conversation_manager(conversation_manager)
        assert strategy.conversation_manager is conversation_manager

    def test_set_code_graph(self, mock_chat_manager, mock_code_graph):
        """Test that set_code_graph sets the graph."""
        strategy = CodeDecisionStrategy(mock_chat_manager)
        strategy.set_code_graph(mock_code_graph)
        assert strategy.code_graph is mock_code_graph


class TestCodeDecisionStrategyInterfaceCompatibility:
    """Test interface compatibility methods for mixins."""

    def test_set_emote_allowed_is_noop(self, decision_strategy):
        """Test that set_emote_allowed is a no-op."""
        decision_strategy.set_emote_allowed(False)
        assert decision_strategy._emote_allowed is False
        decision_strategy.set_emote_allowed(True)
        assert decision_strategy._emote_allowed is True

    def test_set_workspace_active_is_noop(self, decision_strategy):
        """Test that set_workspace_active is a no-op."""
        decision_strategy.set_workspace_active(True)
        assert decision_strategy._workspace_active is True
        decision_strategy.set_workspace_active(False)
        assert decision_strategy._workspace_active is False

    def test_set_context_stores_values(self, decision_strategy):
        """Test that set_context stores Redis client and agent ID."""
        mock_redis = MagicMock()
        decision_strategy.set_context(mock_redis, "blip")
        assert decision_strategy._redis_client is mock_redis
        assert decision_strategy._agent_id == "blip"

    def test_get_plan_tool_impl_returns_none(self, decision_strategy):
        """Test that get_plan_tool_impl returns None for code agents."""
        assert decision_strategy.get_plan_tool_impl() is None

    def test_get_plan_guidance_returns_empty(self, decision_strategy):
        """Test that get_plan_guidance returns empty string."""
        assert decision_strategy.get_plan_guidance() == ""

    def test_build_agent_action_hints_returns_empty(self, decision_strategy):
        """Test that _build_agent_action_hints returns empty list."""
        session = _sample_session()
        hints = decision_strategy._build_agent_action_hints(session)
        assert hints == []


class TestCodeDecisionStrategyBuildTurns:
    """Test CodeDecisionStrategy.build_turns method."""

    @pytest.mark.asyncio
    async def test_build_turns_raises_without_conversation_manager(
        self, mock_chat_manager, mock_persona
    ):
        """Test that build_turns raises if conversation_manager not set."""
        strategy = CodeDecisionStrategy(mock_chat_manager)
        session = _sample_session()

        with pytest.raises(ValueError, match="conversation_manager not set"):
            await strategy.build_turns(mock_persona, session)

    @pytest.mark.asyncio
    async def test_build_turns_returns_valid_structure(
        self, decision_strategy, mock_persona, mock_redis
    ):
        """Test that build_turns returns valid turn structure."""
        session = _sample_session()

        # Mock chat_turns_for to return expected structure
        decision_strategy.chat_turns_for = MagicMock(
            return_value=[
                {"role": "user", "content": "consciousness block"},
                {"role": "user", "content": "current context"},
            ]
        )

        turns = await decision_strategy.build_turns(mock_persona, session)

        assert isinstance(turns, list)
        assert len(turns) >= 1
        decision_strategy.chat_turns_for.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_turns_includes_history(
        self, decision_strategy, mock_persona, mock_redis
    ):
        """Test that build_turns passes history to chat_turns_for."""
        entry1 = _sample_conversation_entry("user", "Hello!", 0)
        entry2 = _sample_conversation_entry("assistant", "Hi!", 1)
        mock_redis.lrange.return_value = [
            entry1.model_dump_json().encode(),
            entry2.model_dump_json().encode(),
        ]

        decision_strategy.chat_turns_for = MagicMock(return_value=[])
        session = _sample_session()

        await decision_strategy.build_turns(mock_persona, session)

        call_kwargs = decision_strategy.chat_turns_for.call_args[1]
        assert "history" in call_kwargs
        history = call_kwargs["history"]
        assert len(history) == 2


class TestCodeDecisionStrategyBuildDecisionGuidance:
    """Test CodeDecisionStrategy._build_decision_guidance method."""

    def test_build_decision_guidance_includes_code_header(self, decision_strategy):
        """Test that guidance includes code-specific header."""
        session = _sample_session()

        guidance = decision_strategy._build_decision_guidance(session)

        assert "Code Tool Use Turn" in guidance

    def test_build_decision_guidance_enumerates_exits(self, decision_strategy):
        """Test that guidance enumerates available exits."""
        session = _sample_session()

        guidance = decision_strategy._build_decision_guidance(session)

        assert "Available exits:" in guidance
        assert "north" in guidance
        assert "south" in guidance

    def test_build_decision_guidance_enumerates_objects(self, decision_strategy):
        """Test that guidance enumerates room objects."""
        session = _sample_session()

        guidance = decision_strategy._build_decision_guidance(session)

        assert "Objects present:" in guidance
        assert "config.py" in guidance

    def test_build_decision_guidance_enumerates_inventory(self, decision_strategy):
        """Test that guidance enumerates inventory items."""
        session = _sample_session()

        guidance = decision_strategy._build_decision_guidance(session)

        assert "Your inventory:" in guidance
        assert "worker.py" in guidance

    def test_build_decision_guidance_enumerates_targets(self, decision_strategy):
        """Test that guidance enumerates valid targets."""
        session = _sample_session()

        guidance = decision_strategy._build_decision_guidance(session)

        assert "People present:" in guidance
        assert "Prax" in guidance

    def test_build_decision_guidance_excludes_self(self, decision_strategy):
        """Test that guidance excludes the agent's own entity."""
        session = _sample_session()

        guidance = decision_strategy._build_decision_guidance(session)

        # Blip (is_self=True) should not appear as a target
        assert "Blip" not in guidance


class TestCodeDecisionStrategyGetConversationHistory:
    """Test CodeDecisionStrategy._get_conversation_history method."""

    @pytest.mark.asyncio
    async def test_get_conversation_history_returns_chat_turns(
        self, decision_strategy, mock_redis
    ):
        """Test that _get_conversation_history returns chat turn format."""
        entry = _sample_conversation_entry("user", "Hello!", 0)
        mock_redis.lrange.return_value = [entry.model_dump_json().encode()]

        history = await decision_strategy._get_conversation_history(token_budget=1000)

        assert isinstance(history, list)
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_get_conversation_history_empty(self, decision_strategy, mock_redis):
        """Test _get_conversation_history with empty history."""
        mock_redis.lrange.return_value = []

        history = await decision_strategy._get_conversation_history(token_budget=1000)

        assert history == []


# =============================================================================
# CodeResponseStrategy Tests
# =============================================================================


class TestCodeResponseStrategyInit:
    """Test CodeResponseStrategy initialization."""

    def test_init_takes_chat_manager(self, mock_chat_manager):
        """Test that __init__ takes ChatManager."""
        strategy = CodeResponseStrategy(mock_chat_manager)
        assert strategy.chat is mock_chat_manager
        assert strategy.conversation_manager is None

    def test_init_sets_interface_compatibility_attrs(self, mock_chat_manager):
        """Test that __init__ sets interface compatibility attributes."""
        strategy = CodeResponseStrategy(mock_chat_manager)
        # thought_content is accessed by ProfileMixin
        assert strategy.thought_content == ""

    def test_set_conversation_manager(self, mock_chat_manager, conversation_manager):
        """Test that set_conversation_manager sets the manager."""
        strategy = CodeResponseStrategy(mock_chat_manager)
        strategy.set_conversation_manager(conversation_manager)
        assert strategy.conversation_manager is conversation_manager

    def test_set_code_graph(self, mock_chat_manager, mock_code_graph):
        """Test that set_code_graph sets the graph."""
        strategy = CodeResponseStrategy(mock_chat_manager)
        strategy.set_code_graph(mock_code_graph)
        assert strategy.code_graph is mock_code_graph


class TestCodeResponseStrategyBuildTurns:
    """Test CodeResponseStrategy.build_turns method."""

    @pytest.mark.asyncio
    async def test_build_turns_returns_list(
        self, response_strategy, mock_persona, mock_redis
    ):
        """Test that build_turns returns a list of turns."""
        session = _sample_session()

        response_strategy.chat_turns_for = MagicMock(
            return_value=[
                {"role": "user", "content": "consciousness"},
                {"role": "assistant", "content": "acknowledged"},
                {"role": "user", "content": "input"},
            ]
        )

        turns = await response_strategy.build_turns(
            persona=mock_persona,
            user_input="test input",
            session=session,
        )

        assert isinstance(turns, list)
        assert len(turns) == 3

    @pytest.mark.asyncio
    async def test_build_turns_sets_location(
        self, response_strategy, mock_persona, mock_redis
    ):
        """Test that build_turns sets chat.current_location from session."""
        session = _sample_session()
        response_strategy.chat_turns_for = MagicMock(return_value=[])

        await response_strategy.build_turns(
            persona=mock_persona,
            user_input="test",
            session=session,
        )

        assert response_strategy.chat.current_location is not None

    @pytest.mark.asyncio
    async def test_build_turns_no_location_without_world_state(
        self, response_strategy, mock_persona
    ):
        """Test that build_turns handles missing world_state."""
        session = MUDSession(
            agent_id="blip",
            persona_id="blip",
            current_room=None,
            world_state=None,
        )
        response_strategy.chat.current_location = None
        response_strategy.chat_turns_for = MagicMock(return_value=[])

        await response_strategy.build_turns(
            persona=mock_persona,
            user_input="test",
            session=session,
        )

        assert response_strategy.chat.current_location is None


class TestCodeResponseStrategyGetConversationHistory:
    """Test CodeResponseStrategy._get_conversation_history method."""

    @pytest.mark.asyncio
    async def test_get_conversation_history_empty_without_manager(
        self, mock_chat_manager
    ):
        """Test that _get_conversation_history returns empty list without manager."""
        strategy = CodeResponseStrategy(mock_chat_manager)

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


class TestCodeResponseStrategyDoubleUserTurnFix:
    """Test that CodeResponseStrategy avoids double consecutive user turns."""

    @pytest.mark.asyncio
    async def test_build_turns_merges_guidance_into_last_user_turn(
        self, response_strategy, mock_persona, mock_redis
    ):
        """Test that format guidance is merged into last user turn."""
        entry1 = _sample_conversation_entry("assistant", "Hi there!", 0)
        entry2 = _sample_conversation_entry("user", "Prax says hello.", 1)
        mock_redis.lrange.return_value = [
            entry1.model_dump_json().encode(),
            entry2.model_dump_json().encode(),
        ]

        received_args = {}

        def capture_args(*args, **kwargs):
            received_args.update(kwargs)
            return []

        response_strategy.chat_turns_for = capture_args
        session = _sample_session()

        await response_strategy.build_turns(
            persona=mock_persona,
            user_input="[~~ FORMAT: code response ~~]",
            session=session,
        )

        assert received_args["user_input"] == ""
        history = received_args["history"]
        last_user = [h for h in history if h["role"] == "user"][-1]
        assert "Prax says hello" in last_user["content"]
        assert "[~~ FORMAT:" in last_user["content"]

    @pytest.mark.asyncio
    async def test_build_turns_no_merge_when_history_ends_with_assistant(
        self, response_strategy, mock_persona, mock_redis
    ):
        """Test that user_input is passed through when history ends with assistant."""
        entry1 = _sample_conversation_entry("user", "Hello!", 0)
        entry2 = _sample_conversation_entry("assistant", "Hi!", 1)
        mock_redis.lrange.return_value = [
            entry1.model_dump_json().encode(),
            entry2.model_dump_json().encode(),
        ]

        received_args = {}

        def capture_args(*args, **kwargs):
            received_args.update(kwargs)
            return []

        response_strategy.chat_turns_for = capture_args
        session = _sample_session()

        await response_strategy.build_turns(
            persona=mock_persona,
            user_input="[~~ FORMAT: code response ~~]",
            session=session,
        )

        assert received_args["user_input"] == "[~~ FORMAT: code response ~~]"
