# packages/aim-core/tests/core_tests/unit/utils/test_think.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for think utilities."""

import pytest
from aim.utils.think import extract_think_tags, extract_reasoning_block


class TestExtractThinkTags:
    """Tests for extract_think_tags function."""

    def test_extract_complete_think_block(self):
        """Test extraction of complete think block."""
        response = "Some text <think>Internal reasoning</think> More text"
        cleaned, think = extract_think_tags(response)

        assert "Internal reasoning" in think
        assert "<think>" not in cleaned
        assert "Some text" in cleaned
        assert "More text" in cleaned

    def test_extract_multiple_think_blocks(self):
        """Test extraction of multiple think blocks."""
        response = "<think>First</think> Middle <think>Second</think>"
        cleaned, think = extract_think_tags(response)

        assert "First" in think
        assert "Second" in think
        assert "Middle" in cleaned

    def test_no_think_tags(self):
        """Test response without think tags."""
        response = "Plain text without think tags"
        cleaned, think = extract_think_tags(response)

        assert cleaned == response
        assert think is None

    def test_truncated_think_tag(self):
        """Test truncated think tag (no closing)."""
        response = "Start <think>Truncated content"
        cleaned, think = extract_think_tags(response)

        # Truncated think is treated as content
        assert think is None
        assert "Start" in cleaned
        assert "Truncated content" in cleaned

    def test_orphan_close_tag(self):
        """Test orphan close tag (started mid-stream)."""
        response = "Continuation from stream</think> Rest"
        cleaned, think = extract_think_tags(response)

        assert "Continuation from stream" in think
        assert "Rest" in cleaned


class TestExtractReasoningBlock:
    """Tests for extract_reasoning_block function."""

    def test_extract_complete_reasoning_block(self):
        """Test extraction of complete reasoning block."""
        response = "Preamble <reasoning>Test reasoning content</reasoning> Epilogue"
        cleaned, reasoning = extract_reasoning_block(response)

        assert reasoning == "Test reasoning content"
        assert "Preamble" in cleaned
        assert "Epilogue" in cleaned
        assert "<reasoning>" not in cleaned

    def test_extract_multiline_reasoning_block(self):
        """Test extraction of multiline reasoning block."""
        response = """<reasoning>
    <inspiration>First observation.</inspiration>
    <exploration>First exploration.</exploration>
</reasoning>"""
        cleaned, reasoning = extract_reasoning_block(response)

        assert "<inspiration>" in reasoning
        assert "<exploration>" in reasoning
        assert cleaned == ""

    def test_no_reasoning_block(self):
        """Test response without reasoning block."""
        response = "Plain text without reasoning block"
        cleaned, reasoning = extract_reasoning_block(response)

        assert cleaned == response
        assert reasoning is None

    def test_reasoning_with_nested_xml(self):
        """Test reasoning block with nested XML elements."""
        response = """<reasoning>
    <inspiration>I notice something.</inspiration>
    <inspiration>I feel something.</inspiration>
    <synthesis>These connect.</synthesis>
</reasoning>"""
        cleaned, reasoning = extract_reasoning_block(response)

        assert "<inspiration>I notice something.</inspiration>" in reasoning
        assert "<inspiration>I feel something.</inspiration>" in reasoning
        assert "<synthesis>These connect.</synthesis>" in reasoning
        assert reasoning is not None

    def test_reasoning_after_think_extraction(self):
        """Test reasoning extraction works after think tag extraction."""
        response = "<think>Internal</think><reasoning>External</reasoning>"

        # First extract think
        after_think, think = extract_think_tags(response)
        assert think == "Internal"

        # Then extract reasoning
        cleaned, reasoning = extract_reasoning_block(after_think)
        assert reasoning == "External"

    def test_empty_reasoning_block(self):
        """Test empty reasoning block."""
        response = "<reasoning></reasoning>"
        cleaned, reasoning = extract_reasoning_block(response)

        assert reasoning == ""
        assert cleaned == ""

    def test_reasoning_block_with_whitespace_only(self):
        """Test reasoning block with only whitespace."""
        response = "<reasoning>   \n  \t  </reasoning>"
        cleaned, reasoning = extract_reasoning_block(response)

        assert reasoning == ""  # Stripped

    def test_unclosed_reasoning_block(self):
        """Test unclosed reasoning block returns None."""
        response = "<reasoning>Unclosed content"
        cleaned, reasoning = extract_reasoning_block(response)

        assert reasoning is None
        assert cleaned == response

    def test_reasoning_with_xml_special_chars(self):
        """Test reasoning block with XML special characters."""
        response = """<reasoning>
    <inspiration>She said &lt;pause&gt; and I noticed.</inspiration>
    <exploration>Does &amp; mean something here?</exploration>
</reasoning>"""
        cleaned, reasoning = extract_reasoning_block(response)

        assert "&lt;pause&gt;" in reasoning
        assert "&amp;" in reasoning
        assert reasoning is not None

    def test_multiple_reasoning_blocks(self):
        """Test first reasoning block content extracted, all blocks removed from output."""
        response = "<reasoning>First block</reasoning> Text <reasoning>Second block</reasoning>"
        cleaned, reasoning = extract_reasoning_block(response)

        # Only first block's content is extracted
        assert reasoning == "First block"
        # But ALL reasoning blocks are removed from cleaned output
        assert cleaned == "Text"
        assert "<reasoning>" not in cleaned


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
