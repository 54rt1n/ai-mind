# tests/unit/chat/strategy/test_consciousness_hooks.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for consciousness block hook mechanism in XMLMemoryTurnStrategy."""

import pytest
from unittest.mock import MagicMock

from aim.chat.strategy.xmlmemory import XMLMemoryTurnStrategy
from aim.utils.xml import XmlFormatter


class TestConsciousnessHooksBaseStrategy:
    """Test consciousness hook methods in base XMLMemoryTurnStrategy."""

    def test_get_consciousness_head_returns_unchanged_formatter(self, unit_test_xml_strategy):
        """Test that default get_consciousness_head returns formatter unchanged."""
        formatter = XmlFormatter()
        formatter.add_element("test", "element", content="before hook")

        result = unit_test_xml_strategy.get_consciousness_head(formatter)

        # Should be the same formatter instance
        assert result is formatter
        # Content should be unchanged
        rendered = result.render()
        assert "before hook" in rendered
        assert "test" in rendered

    def test_get_consciousness_tail_returns_unchanged_formatter(self, unit_test_xml_strategy):
        """Test that default get_consciousness_tail returns formatter unchanged."""
        formatter = XmlFormatter()
        formatter.add_element("test", "element", content="before hook")

        result = unit_test_xml_strategy.get_consciousness_tail(formatter)

        # Should be the same formatter instance
        assert result is formatter
        # Content should be unchanged
        rendered = result.render()
        assert "before hook" in rendered
        assert "test" in rendered

    def test_get_consciousness_head_with_empty_formatter(self, unit_test_xml_strategy):
        """Test get_consciousness_head with empty formatter."""
        formatter = XmlFormatter()

        result = unit_test_xml_strategy.get_consciousness_head(formatter)

        assert result is formatter
        # Empty formatter renders as <root></root>
        rendered = result.render()
        assert "<root>" in rendered
        assert "</root>" in rendered

    def test_get_consciousness_tail_with_empty_formatter(self, unit_test_xml_strategy):
        """Test get_consciousness_tail with empty formatter."""
        formatter = XmlFormatter()

        result = unit_test_xml_strategy.get_consciousness_tail(formatter)

        assert result is formatter
        # Empty formatter renders as <root></root>
        rendered = result.render()
        assert "<root>" in rendered
        assert "</root>" in rendered


class TestConsciousnessHooksInGetConsciousMemory:
    """Test that hooks are called during get_conscious_memory execution."""

    def test_hooks_called_during_consciousness_generation(
        self, unit_test_xml_strategy, sample_persona, mock_chat_manager, mocker
    ):
        """Test that both hooks are called during get_conscious_memory."""
        # Spy on the hook methods to verify they're called
        head_spy = mocker.spy(unit_test_xml_strategy, 'get_consciousness_head')
        tail_spy = mocker.spy(unit_test_xml_strategy, 'get_consciousness_tail')

        xml_output, _ = unit_test_xml_strategy.get_conscious_memory(sample_persona)

        # Hooks are called twice: once for token estimation (temp formatter),
        # once for actual rendering
        assert head_spy.call_count == 2, "Head hook should be called twice (estimation + render)"
        assert tail_spy.call_count == 2, "Tail hook should be called twice (estimation + render)"

        # Verify they received XmlFormatter instances
        for call in head_spy.call_args_list:
            assert isinstance(call[0][0], XmlFormatter)
        for call in tail_spy.call_args_list:
            assert isinstance(call[0][0], XmlFormatter)

    def test_hook_content_appears_in_consciousness_block(
        self, unit_test_xml_strategy, sample_persona, mock_chat_manager
    ):
        """Test that consciousness block includes standard elements around hook positions."""
        xml_output, _ = unit_test_xml_strategy.get_conscious_memory(sample_persona)

        # Verify consciousness structure elements are present
        assert "<PraxOS>" in xml_output
        assert "Conscious Memory **Online**" in xml_output

        # Verify thoughts appear (standard content)
        for thought in sample_persona.thoughts:
            assert thought in xml_output

        # Memory count should appear at the end (before tail hook)
        assert "<Memory Count>" in xml_output or "Memory Count" in xml_output


class TestConsciousnessHookTokenCalculation:
    """Test that hook token estimates are calculated and included in budget."""

    def test_head_hook_token_estimation(
        self, unit_test_xml_strategy, sample_persona, mock_chat_manager, mocker
    ):
        """Test that head hook tokens are estimated using temporary formatter."""
        # Spy on get_consciousness_head to track calls
        call_count = {'count': 0}
        original_head = unit_test_xml_strategy.get_consciousness_head

        def tracked_head(formatter):
            call_count['count'] += 1
            return original_head(formatter)

        mocker.patch.object(unit_test_xml_strategy, 'get_consciousness_head', tracked_head)

        xml_output, _ = unit_test_xml_strategy.get_conscious_memory(sample_persona)

        # Should be called twice: once for token estimation (temp formatter),
        # once for actual rendering
        assert call_count['count'] == 2

    def test_tail_hook_token_estimation(
        self, unit_test_xml_strategy, sample_persona, mock_chat_manager, mocker
    ):
        """Test that tail hook tokens are estimated using temporary formatter."""
        # Spy on get_consciousness_tail to track calls
        call_count = {'count': 0}
        original_tail = unit_test_xml_strategy.get_consciousness_tail

        def tracked_tail(formatter):
            call_count['count'] += 1
            return original_tail(formatter)

        mocker.patch.object(unit_test_xml_strategy, 'get_consciousness_tail', tracked_tail)

        xml_output, _ = unit_test_xml_strategy.get_conscious_memory(sample_persona)

        # Should be called twice: once for token estimation (temp formatter),
        # once for actual rendering
        assert call_count['count'] == 2

    def test_hook_tokens_affect_available_budget(
        self, unit_test_xml_strategy, sample_persona, mock_chat_manager, mocker
    ):
        """Test that hook token estimates reduce available_tokens_for_dynamic_queries."""
        # Create a subclass with non-empty hook content
        class HookWithContent(XMLMemoryTurnStrategy):
            def get_consciousness_head(self, formatter):
                formatter.add_element("TestHook", "head", content="X" * 100)
                return formatter

            def get_consciousness_tail(self, formatter):
                formatter.add_element("TestHook", "tail", content="Y" * 100)
                return formatter

        strategy_with_hooks = HookWithContent(mock_chat_manager)
        strategy_with_hooks.max_character_length = 20000

        # Get conscious memory - hooks should consume tokens
        xml_output, _ = strategy_with_hooks.get_conscious_memory(sample_persona)

        # Verify hook content appears in output
        assert "TestHook" in xml_output
        assert "X" * 100 in xml_output
        assert "Y" * 100 in xml_output

    def test_large_hook_content_reduces_dynamic_query_space(
        self, unit_test_xml_strategy, sample_persona, mock_chat_manager, mocker
    ):
        """Test that large hook content reduces space for dynamic queries."""
        # Create a version with large hook content
        class LargeHookStrategy(XMLMemoryTurnStrategy):
            def get_consciousness_tail(self, formatter):
                # Add substantial content to consume tokens
                formatter.add_element(
                    "LargeHook", "test",
                    content="Large content. " * 500  # ~1000+ tokens
                )
                return formatter

        large_hook_strategy = LargeHookStrategy(mock_chat_manager)
        large_hook_strategy.max_character_length = 20000

        # The strategy should still complete without error
        xml_output, _ = large_hook_strategy.get_conscious_memory(
            sample_persona,
            max_context_tokens=4000,
            max_output_tokens=500
        )

        # Large hook content should be present
        assert "LargeHook" in xml_output
        assert "Large content." in xml_output

        # Should not exceed token budget (verify no error was raised)
        token_count = unit_test_xml_strategy.count_tokens(xml_output)
        # usable_context = 4000 - 500 - 1024 = 2476
        assert token_count <= 2476


class TestConsciousnessHookOrdering:
    """Test that hooks appear in correct positions in consciousness block."""

    def test_head_hook_appears_before_praxos(
        self, unit_test_xml_strategy, sample_persona, mock_chat_manager
    ):
        """Test that head hook content would appear before PraxOS header."""
        # Create a subclass with identifiable head content
        class HeadHookStrategy(XMLMemoryTurnStrategy):
            def get_consciousness_head(self, formatter):
                formatter.add_element("HeadMarker", content="START_OF_CONSCIOUSNESS")
                return formatter

        head_strategy = HeadHookStrategy(mock_chat_manager)
        head_strategy.max_character_length = 20000

        xml_output, _ = head_strategy.get_conscious_memory(sample_persona)

        # Find positions
        head_pos = xml_output.find("START_OF_CONSCIOUSNESS")
        praxos_pos = xml_output.find("PraxOS")

        # Head hook should appear before PraxOS
        assert head_pos != -1, "Head hook content not found"
        assert praxos_pos != -1, "PraxOS header not found"
        assert head_pos < praxos_pos, "Head hook should appear before PraxOS"

    def test_tail_hook_appears_after_memory_count(
        self, unit_test_xml_strategy, sample_persona, mock_chat_manager
    ):
        """Test that tail hook content appears after Memory Count."""
        # Create a subclass with identifiable tail content
        class TailHookStrategy(XMLMemoryTurnStrategy):
            def get_consciousness_tail(self, formatter):
                # Add to the Conscious Memory section (same as other elements)
                formatter.add_element(self.hud_name, "TailMarker", content="END_OF_CONSCIOUSNESS", priority=1)
                return formatter

        tail_strategy = TailHookStrategy(mock_chat_manager)
        tail_strategy.max_character_length = 20000

        xml_output, _ = tail_strategy.get_conscious_memory(sample_persona)

        # Find positions
        memory_count_pos = xml_output.find("Memory Count")
        tail_pos = xml_output.find("END_OF_CONSCIOUSNESS")

        # Tail hook should appear after Memory Count
        assert memory_count_pos != -1, "Memory Count not found"
        assert tail_pos != -1, "Tail hook content not found"
        assert tail_pos > memory_count_pos, "Tail hook should appear after Memory Count"

    def test_both_hooks_maintain_correct_order(
        self, unit_test_xml_strategy, sample_persona, mock_chat_manager
    ):
        """Test that both hooks maintain correct relative positions."""
        # Create a subclass with both hooks
        class BothHooksStrategy(XMLMemoryTurnStrategy):
            def get_consciousness_head(self, formatter):
                # Head hook adds to root before PraxOS is added
                formatter.add_element("HeadHook", content="HEAD_MARKER", priority=3)
                return formatter

            def get_consciousness_tail(self, formatter):
                # Tail hook adds to Conscious Memory after Memory Count
                formatter.add_element(self.hud_name, "TailHook", content="TAIL_MARKER", priority=1)
                return formatter

        both_strategy = BothHooksStrategy(mock_chat_manager)
        both_strategy.max_character_length = 20000

        xml_output, _ = both_strategy.get_conscious_memory(sample_persona)

        # Find all key positions
        head_pos = xml_output.find("HEAD_MARKER")
        praxos_pos = xml_output.find("PraxOS")
        memory_count_pos = xml_output.find("Memory Count")
        tail_pos = xml_output.find("TAIL_MARKER")

        # Verify order: HEAD < PraxOS < Memory Count < TAIL
        assert head_pos < praxos_pos, "Head should come before PraxOS"
        assert praxos_pos < memory_count_pos, "PraxOS should come before Memory Count"
        assert memory_count_pos < tail_pos, "Memory Count should come before tail"
