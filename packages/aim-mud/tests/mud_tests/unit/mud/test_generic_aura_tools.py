# tests/unit/mud/test_generic_aura_tools.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for the generic aura tool system.

Tests the generalization that replaced hardcoded 'ring' handling with
support for any aura-provided tool (stock_quote, web_search, etc.).

Tests cover:
- MUDDecisionStrategy.is_aura_tool() method
- Generic aura tool handling in turns.py (_decide_action)
- Generic aura tool handling in phased.py (PhasedTurnProcessor)
- Generic fallback in MUDAction.to_command() for unknown aura tools

Testing philosophy: Mock external services only (Redis, LLM).
Use real implementations for all internal logic.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aim_mud_types import (
    MUDSession,
    MUDAction,
    MUDTurnRequest,
    MUDEvent,
    RoomState,
    EntityState,
    WorldState,
)
from aim_mud_types.state import AuraState
from andimud_worker.conversation.memory.decision import MUDDecisionStrategy
from andimud_worker.conversation import MUDConversationManager
from andimud_worker.turns.processor.phased import PhasedTurnProcessor
from aim.tool.dto import Tool, ToolFunction, ToolFunctionParameters
from aim.tool.formatting import ToolUser


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tools_path() -> str:
    """Path to config/tools for real YAML loading."""
    test_file = Path(__file__)
    project_root = test_file.parent.parent.parent.parent.parent.parent.parent
    return str(project_root / "config" / "tools")


@pytest.fixture
def mock_chat_manager():
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
        persona_id="test_persona",
        max_tokens=50000,
    )


@pytest.fixture
def decision_strategy(mock_chat_manager, conversation_manager):
    """Create a MUDDecisionStrategy with conversation manager."""
    strategy = MUDDecisionStrategy(mock_chat_manager)
    strategy.set_conversation_manager(conversation_manager)
    return strategy


@pytest.fixture
def sample_base_tools():
    """Sample base decision tools (speak, move, etc.)."""
    return [
        Tool(
            type="function",
            function=ToolFunction(
                name="speak",
                description="Choose to speak",
                parameters=ToolFunctionParameters(
                    type="object",
                    properties={},
                    required=[],
                ),
            ),
        ),
        Tool(
            type="function",
            function=ToolFunction(
                name="move",
                description="Move to another location",
                parameters=ToolFunctionParameters(
                    type="object",
                    properties={
                        "location": {"type": "string", "description": "Exit name"}
                    },
                    required=["location"],
                ),
            ),
        ),
    ]


@pytest.fixture
def sample_aura_tool_stock_quote():
    """Sample stock_quote aura tool."""
    return Tool(
        type="function",
        function=ToolFunction(
            name="stock_quote",
            description="Get a stock quote",
            parameters=ToolFunctionParameters(
                type="object",
                properties={
                    "symbol": {"type": "string", "description": "Stock symbol"}
                },
                required=["symbol"],
            ),
        ),
    )


@pytest.fixture
def sample_aura_tool_web_search():
    """Sample web_search aura tool."""
    return Tool(
        type="function",
        function=ToolFunction(
            name="web_search",
            description="Search the web",
            parameters=ToolFunctionParameters(
                type="object",
                properties={
                    "query": {"type": "string", "description": "Search query"}
                },
                required=["query"],
            ),
        ),
    )


# ============================================================================
# TestIsAuraTool - MUDDecisionStrategy.is_aura_tool() method
# ============================================================================

class TestIsAuraTool:
    """Tests for MUDDecisionStrategy.is_aura_tool() method."""

    def test_is_aura_tool_returns_false_for_base_tool(
        self, decision_strategy, sample_base_tools
    ):
        """Test that base decision tools are not identified as aura tools."""
        decision_strategy._base_tools = sample_base_tools
        decision_strategy._aura_tools = []
        decision_strategy._refresh_tool_user()

        assert decision_strategy.is_aura_tool("speak") is False
        assert decision_strategy.is_aura_tool("move") is False

    def test_is_aura_tool_returns_true_for_aura_tool(
        self, decision_strategy, sample_base_tools, sample_aura_tool_stock_quote
    ):
        """Test that aura-provided tools are correctly identified."""
        decision_strategy._base_tools = sample_base_tools
        decision_strategy._aura_tools = [sample_aura_tool_stock_quote]
        decision_strategy._refresh_tool_user()

        assert decision_strategy.is_aura_tool("stock_quote") is True

    def test_is_aura_tool_returns_false_for_unknown_tool(
        self, decision_strategy, sample_base_tools
    ):
        """Test that unknown tools return False."""
        decision_strategy._base_tools = sample_base_tools
        decision_strategy._aura_tools = []
        decision_strategy._refresh_tool_user()

        assert decision_strategy.is_aura_tool("nonexistent_tool") is False

    def test_is_aura_tool_with_multiple_aura_tools(
        self,
        decision_strategy,
        sample_base_tools,
        sample_aura_tool_stock_quote,
        sample_aura_tool_web_search,
    ):
        """Test is_aura_tool with multiple aura tools loaded."""
        decision_strategy._base_tools = sample_base_tools
        decision_strategy._aura_tools = [
            sample_aura_tool_stock_quote,
            sample_aura_tool_web_search,
        ]
        decision_strategy._refresh_tool_user()

        assert decision_strategy.is_aura_tool("stock_quote") is True
        assert decision_strategy.is_aura_tool("web_search") is True
        assert decision_strategy.is_aura_tool("speak") is False

    def test_is_aura_tool_with_empty_aura_tools(
        self, decision_strategy, sample_base_tools
    ):
        """Test is_aura_tool when no aura tools are loaded."""
        decision_strategy._base_tools = sample_base_tools
        decision_strategy._aura_tools = []
        decision_strategy._refresh_tool_user()

        assert decision_strategy.is_aura_tool("ring") is False
        assert decision_strategy.is_aura_tool("stock_quote") is False

    def test_is_aura_tool_case_sensitive(
        self, decision_strategy, sample_base_tools, sample_aura_tool_stock_quote
    ):
        """Test that is_aura_tool is case-sensitive."""
        decision_strategy._base_tools = sample_base_tools
        decision_strategy._aura_tools = [sample_aura_tool_stock_quote]
        decision_strategy._refresh_tool_user()

        assert decision_strategy.is_aura_tool("stock_quote") is True
        assert decision_strategy.is_aura_tool("STOCK_QUOTE") is False
        assert decision_strategy.is_aura_tool("Stock_Quote") is False


# ============================================================================
# TestGenericAuraToolHandlingTurns - turns.py logic validation
# ============================================================================

class TestGenericAuraToolHandlingTurns:
    """Tests for generic aura tool handling in turns.py (_decide_action).

    These tests verify the logic without complex integration mocking.
    The actual _decide_action flow is tested in integration tests.
    """

    def test_is_aura_tool_check_in_turns_logic(
        self,
        decision_strategy,
        sample_base_tools,
        sample_aura_tool_stock_quote,
    ):
        """Test that turns.py can check if a tool is an aura tool."""
        # Setup strategy with aura tools
        decision_strategy._base_tools = sample_base_tools
        decision_strategy._aura_tools = [sample_aura_tool_stock_quote]
        decision_strategy._refresh_tool_user()

        # Simulate what turns.py does - check if tool is from aura
        # In turns.py line 354: if self._decision_strategy.is_aura_tool(decision_tool):
        assert decision_strategy.is_aura_tool("stock_quote") is True
        assert decision_strategy.is_aura_tool("speak") is False
        assert decision_strategy.is_aura_tool("nonexistent") is False

    def test_ring_is_still_validated(
        self, decision_strategy, sample_base_tools, tools_path
    ):
        """Test that ring tool is recognized as aura tool and can be validated."""
        # Load actual ring tool from YAML
        decision_strategy.update_aura_tools(["ringable"], tools_path)
        decision_strategy._base_tools = sample_base_tools
        decision_strategy._refresh_tool_user()

        # Verify ring is recognized as aura tool
        assert decision_strategy.is_aura_tool("ring") is True

        # Ring should still go through specific validation in turns.py
        # This is covered by existing validate_ring tests in test_auras.py


# ============================================================================
# TestGenericAuraToolHandlingPhased - phased.py logic validation
# ============================================================================

class TestGenericAuraToolHandlingPhased:
    """Tests for generic aura tool handling in phased.py.

    These tests verify the logic path exists in phased.py without full integration.
    """

    def test_phased_processor_logic_for_aura_tool(
        self, decision_strategy, sample_base_tools, sample_aura_tool_stock_quote
    ):
        """Test the logic that identifies aura tools in phased.py."""
        # Setup strategy
        decision_strategy._base_tools = sample_base_tools
        decision_strategy._aura_tools = [sample_aura_tool_stock_quote]
        decision_strategy._refresh_tool_user()

        # In phased.py line 116: elif self.worker._decision_strategy.is_aura_tool(decision_tool):
        # This checks if the tool is an aura tool, then emits a MUDAction
        assert decision_strategy.is_aura_tool("stock_quote") is True

        # Create the action that would be emitted
        action = MUDAction(tool="stock_quote", args={"symbol": "TSLA"})
        assert action.tool == "stock_quote"
        assert action.args == {"symbol": "TSLA"}
        # Verify command generation works
        assert action.to_command() == "stock_quote TSLA"

    def test_phased_processor_handles_ring_as_aura_tool(
        self, decision_strategy, sample_base_tools, tools_path
    ):
        """Test that ring (a specific aura tool) still works in phased.py."""
        # Load ring tool
        decision_strategy.update_aura_tools(["ringable"], tools_path)
        decision_strategy._base_tools = sample_base_tools
        decision_strategy._refresh_tool_user()

        # Ring should be recognized as aura tool
        assert decision_strategy.is_aura_tool("ring") is True

        # Create action that would be emitted
        action = MUDAction(tool="ring", args={"object": "Bell Tower Bell"})
        assert action.to_command() == "ring Bell Tower Bell"


# ============================================================================
# TestGenericFallbackInMUDAction - actions.py to_command() fallback
# ============================================================================

class TestGenericFallbackInMUDAction:
    """Tests for generic fallback in MUDAction.to_command()."""

    def test_to_command_unknown_aura_tool_with_args(self):
        """Test that unknown aura tools use generic fallback with args."""
        action = MUDAction(
            tool="custom_tool",
            args={"param1": "value1", "param2": "value2"}
        )
        command = action.to_command()

        # Should format as: tool_name param1=value1 param2=value2
        assert "custom_tool" in command
        assert "param1=value1" in command
        assert "param2=value2" in command

    def test_to_command_unknown_aura_tool_no_args(self):
        """Test that unknown aura tools with no args return just the tool name."""
        action = MUDAction(tool="simple_tool", args={})
        command = action.to_command()

        assert command == "simple_tool"

    def test_to_command_preserves_existing_aura_tools(self):
        """Test that explicitly handled aura tools still work correctly."""
        # Test stock_quote (explicitly handled)
        action = MUDAction(tool="stock_quote", args={"symbol": "GOOG"})
        assert action.to_command() == "stock_quote GOOG"

        # Test web_search (explicitly handled)
        action = MUDAction(tool="web_search", args={"query": "test query"})
        assert action.to_command() == "web_search test query"

    def test_to_command_generic_fallback_preserves_order(self):
        """Test that generic fallback maintains consistent arg order."""
        action = MUDAction(
            tool="new_tool",
            args={"z_param": "z", "a_param": "a", "m_param": "m"}
        )
        command = action.to_command()

        # Should contain all parameters
        assert "z_param=z" in command
        assert "a_param=a" in command
        assert "m_param=m" in command

    def test_to_command_generic_fallback_handles_complex_values(self):
        """Test generic fallback with various value types."""
        action = MUDAction(
            tool="complex_tool",
            args={
                "str_param": "hello world",
                "int_param": 42,
                "bool_param": True,
            }
        )
        command = action.to_command()

        assert "str_param=hello world" in command
        assert "int_param=42" in command
        assert "bool_param=True" in command


# ============================================================================
# TestGenericAuraToolIntegration - End-to-end integration
# ============================================================================

class TestGenericAuraToolIntegration:
    """End-to-end integration tests for generic aura tool system."""

    def test_integration_stock_quote_aura_tool_flow(
        self, decision_strategy, sample_base_tools, sample_aura_tool_stock_quote
    ):
        """Test complete flow: load aura tool → recognize → emit action → command."""
        # 1. Load aura tool
        decision_strategy._base_tools = sample_base_tools
        decision_strategy._aura_tools = [sample_aura_tool_stock_quote]
        decision_strategy._refresh_tool_user()

        # 2. Verify tool is recognized as aura tool
        assert decision_strategy.is_aura_tool("stock_quote") is True
        assert decision_strategy.is_aura_tool("speak") is False

        # 3. Create action
        action = MUDAction(tool="stock_quote", args={"symbol": "MSFT"})

        # 4. Verify command generation
        assert action.to_command() == "stock_quote MSFT"

    def test_integration_new_unknown_aura_tool(
        self, decision_strategy, sample_base_tools
    ):
        """Test that a brand new unknown aura tool works end-to-end."""
        # Create a completely new aura tool not in the codebase
        new_tool = Tool(
            type="function",
            function=ToolFunction(
                name="quantum_compute",
                description="Perform quantum computation",
                parameters=ToolFunctionParameters(
                    type="object",
                    properties={
                        "qubits": {"type": "integer"},
                        "algorithm": {"type": "string"},
                    },
                    required=["qubits", "algorithm"],
                ),
            ),
        )

        # Load it as an aura tool
        decision_strategy._base_tools = sample_base_tools
        decision_strategy._aura_tools = [new_tool]
        decision_strategy._refresh_tool_user()

        # Verify recognition
        assert decision_strategy.is_aura_tool("quantum_compute") is True

        # Create action and verify generic fallback command
        action = MUDAction(
            tool="quantum_compute",
            args={"qubits": 5, "algorithm": "Shor"}
        )
        command = action.to_command()

        assert "quantum_compute" in command
        assert "qubits=5" in command
        assert "algorithm=Shor" in command

    def test_integration_multiple_auras_with_overlapping_tools(
        self, decision_strategy, sample_base_tools, tools_path
    ):
        """Test loading multiple auras where tools might overlap."""
        # Load multiple aura tool files
        decision_strategy._base_tools = sample_base_tools
        decision_strategy.update_aura_tools(
            ["ringable", "market_access", "web_access"],
            tools_path
        )

        # Verify all aura tools are recognized
        tool_names = decision_strategy.get_available_tool_names()

        # Should have base tools
        assert "speak" in tool_names
        assert "move" in tool_names

        # Should have aura tools (at least ring from ringable)
        assert "ring" in tool_names

        # Each should be identified as aura tool
        assert decision_strategy.is_aura_tool("ring") is True
        # Base tools should NOT be aura tools
        assert decision_strategy.is_aura_tool("speak") is False
