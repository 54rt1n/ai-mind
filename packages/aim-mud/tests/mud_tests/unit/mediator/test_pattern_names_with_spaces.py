# tests/unit/mediator/test_pattern_names_with_spaces.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for regex patterns with character names containing spaces.

This tests the fix for character names like "Lin Yu" which contain spaces.
The patterns should capture the full name and normalize_agent_id should
convert "Lin Yu" to "linyu" for Redis key lookups.
"""

import pytest
from andimud_mediator.patterns import (
    normalize_agent_id,
    DREAMER_PATTERN,
    ANALYZE_PATTERN,
    SUMMARY_PATTERN,
    JOURNAL_PATTERN,
    PONDER_PATTERN,
    DAYDREAM_PATTERN,
    CRITIQUE_PATTERN,
    RESEARCH_PATTERN,
    PLANNER_PATTERN,
    PLAN_PATTERN,
    UPDATE_PATTERN,
)


class TestNormalizeAgentId:
    """Test the normalize_agent_id helper function."""

    def test_name_with_space(self):
        """Test normalization of name with space."""
        assert normalize_agent_id("Lin Yu") == "linyu"

    def test_name_with_multiple_spaces(self):
        """Test normalization of name with multiple spaces."""
        assert normalize_agent_id("John Doe Smith") == "johndoesmith"

    def test_single_word_name(self):
        """Test normalization of single word name."""
        assert normalize_agent_id("Nova") == "nova"
        assert normalize_agent_id("Andi") == "andi"

    def test_name_with_leading_trailing_spaces(self):
        """Test normalization strips leading/trailing spaces."""
        assert normalize_agent_id("  Lin Yu  ") == "linyu"
        assert normalize_agent_id("  Nova  ") == "nova"

    def test_uppercase_conversion(self):
        """Test normalization converts to lowercase."""
        assert normalize_agent_id("LIN YU") == "linyu"
        assert normalize_agent_id("Nova") == "nova"
        assert normalize_agent_id("TIBERIUS") == "tiberius"


class TestDreamerPatternWithSpaces:
    """Test DREAMER_PATTERN with names containing spaces."""

    def test_pattern_with_space_on(self):
        """Test pattern matches name with space for 'on' command."""
        match = DREAMER_PATTERN.match("@dreamer Lin Yu on")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "linyu"
        assert match.group(2) == "on"

    def test_pattern_with_space_off(self):
        """Test pattern matches name with space for 'off' command."""
        match = DREAMER_PATTERN.match("@dreamer Lin Yu off")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "linyu"
        assert match.group(2) == "off"

    def test_pattern_with_multiple_word_name(self):
        """Test pattern matches name with multiple words."""
        match = DREAMER_PATTERN.match("@dreamer John Doe Smith on")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "johndoesmith"
        assert match.group(2) == "on"

    def test_pattern_single_word_name_still_works(self):
        """Test pattern still works with single word names."""
        match = DREAMER_PATTERN.match("@dreamer Nova off")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "nova"
        assert match.group(2) == "off"


class TestAnalyzePatternWithSpaces:
    """Test ANALYZE_PATTERN with names containing spaces."""

    def test_pattern_with_space_and_guidance(self):
        """Test pattern matches name with space, conversation_id, and guidance."""
        match = ANALYZE_PATTERN.match("@analyze Lin Yu = conv_123, Fix the bug")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "linyu"
        assert match.group(2) == "conv_123"
        assert match.group(3).strip() == "Fix the bug"

    def test_pattern_with_space_no_guidance(self):
        """Test pattern matches name with space and conversation_id only."""
        match = ANALYZE_PATTERN.match("@analyze Lin Yu = conv_456")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "linyu"
        assert match.group(2) == "conv_456"

    def test_pattern_single_word_name_still_works(self):
        """Test pattern still works with single word names."""
        match = ANALYZE_PATTERN.match("@analyze Nova = conv_789")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "nova"
        assert match.group(2) == "conv_789"


class TestSummaryPatternWithSpaces:
    """Test SUMMARY_PATTERN with names containing spaces."""

    def test_pattern_with_space(self):
        """Test pattern matches name with space."""
        match = SUMMARY_PATTERN.match("@summary Lin Yu = conv_123")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "linyu"
        assert match.group(2) == "conv_123"

    def test_pattern_single_word_name_still_works(self):
        """Test pattern still works with single word names."""
        match = SUMMARY_PATTERN.match("@summary Nova = conv_456")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "nova"
        assert match.group(2) == "conv_456"


class TestJournalPatternWithSpaces:
    """Test JOURNAL_PATTERN with names containing spaces."""

    def test_pattern_with_space_and_params(self):
        """Test pattern matches name with space, query, and guidance."""
        match = JOURNAL_PATTERN.match("@journal Lin Yu = What happened today?, Be reflective")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "linyu"
        assert match.group(2) == "What happened today?"
        assert match.group(3).strip() == "Be reflective"

    def test_pattern_with_space_no_params(self):
        """Test pattern matches name with space and no parameters."""
        match = JOURNAL_PATTERN.match("@journal Lin Yu")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "linyu"
        assert match.group(2) is None

    def test_pattern_single_word_name_still_works(self):
        """Test pattern still works with single word names."""
        match = JOURNAL_PATTERN.match("@journal Nova = What did I learn?")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "nova"
        assert match.group(2) == "What did I learn?"


class TestPonderPatternWithSpaces:
    """Test PONDER_PATTERN with names containing spaces."""

    def test_pattern_with_space(self):
        """Test pattern matches name with space."""
        match = PONDER_PATTERN.match("@ponder Lin Yu = Why do I exist?")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "linyu"
        assert match.group(2) == "Why do I exist?"

    def test_pattern_single_word_name_still_works(self):
        """Test pattern still works with single word names."""
        match = PONDER_PATTERN.match("@ponder Nova")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "nova"


class TestDaydreamPatternWithSpaces:
    """Test DAYDREAM_PATTERN with names containing spaces."""

    def test_pattern_with_space(self):
        """Test pattern matches name with space."""
        match = DAYDREAM_PATTERN.match("@daydream Lin Yu = Flying through clouds")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "linyu"
        assert match.group(2) == "Flying through clouds"

    def test_pattern_single_word_name_still_works(self):
        """Test pattern still works with single word names."""
        match = DAYDREAM_PATTERN.match("@daydream Nova")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "nova"


class TestCritiquePatternWithSpaces:
    """Test CRITIQUE_PATTERN with names containing spaces."""

    def test_pattern_with_space(self):
        """Test pattern matches name with space."""
        match = CRITIQUE_PATTERN.match("@critique Lin Yu = My recent actions")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "linyu"
        assert match.group(2) == "My recent actions"

    def test_pattern_single_word_name_still_works(self):
        """Test pattern still works with single word names."""
        match = CRITIQUE_PATTERN.match("@critique Nova")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "nova"


class TestResearchPatternWithSpaces:
    """Test RESEARCH_PATTERN with names containing spaces."""

    def test_pattern_with_space(self):
        """Test pattern matches name with space."""
        match = RESEARCH_PATTERN.match("@research Lin Yu = AI ethics")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "linyu"
        assert match.group(2) == "AI ethics"

    def test_pattern_single_word_name_still_works(self):
        """Test pattern still works with single word names."""
        match = RESEARCH_PATTERN.match("@research Nova = Quantum computing")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "nova"
        assert match.group(2) == "Quantum computing"


class TestPlannerPatternWithSpaces:
    """Test PLANNER_PATTERN with names containing spaces."""

    def test_pattern_with_space_on(self):
        """Test pattern matches name with space for 'on' command."""
        match = PLANNER_PATTERN.match("@planner Lin Yu on")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "linyu"
        assert match.group(2) == "on"

    def test_pattern_with_space_off(self):
        """Test pattern matches name with space for 'off' command."""
        match = PLANNER_PATTERN.match("@planner Lin Yu off")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "linyu"
        assert match.group(2) == "off"

    def test_pattern_single_word_name_still_works(self):
        """Test pattern still works with single word names."""
        match = PLANNER_PATTERN.match("@planner Nova on")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "nova"
        assert match.group(2) == "on"


class TestPlanPatternWithSpaces:
    """Test PLAN_PATTERN with names containing spaces."""

    def test_pattern_with_space(self):
        """Test pattern matches name with space."""
        match = PLAN_PATTERN.match("@plan Lin Yu = Explore the eastern forest")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "linyu"
        assert match.group(2) == "Explore the eastern forest"

    def test_pattern_single_word_name_still_works(self):
        """Test pattern still works with single word names."""
        match = PLAN_PATTERN.match("@plan Nova = Investigate the signal")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "nova"
        assert match.group(2) == "Investigate the signal"


class TestUpdatePatternWithSpaces:
    """Test UPDATE_PATTERN with names containing spaces."""

    def test_pattern_with_space(self):
        """Test pattern matches name with space."""
        match = UPDATE_PATTERN.match("@update Lin Yu = Task completed successfully")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "linyu"
        assert match.group(2) == "Task completed successfully"

    def test_pattern_single_word_name_still_works(self):
        """Test pattern still works with single word names."""
        match = UPDATE_PATTERN.match("@update Nova = Progress made on objective")
        assert match is not None
        agent_id = normalize_agent_id(match.group(1))
        assert agent_id == "nova"
        assert match.group(2) == "Progress made on objective"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
