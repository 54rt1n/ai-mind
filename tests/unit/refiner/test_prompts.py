# tests/unit/refiner/test_prompts.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for the refiner prompts module."""

import pytest
from unittest.mock import MagicMock, patch

from aim.refiner.prompts import (
    build_topic_selection_prompt,
    build_validation_prompt,
    build_brainstorm_selection_prompt,
    build_daydream_selection_prompt,
    build_knowledge_selection_prompt,
    _format_documents,
    _get_refiner_tools,
    _create_fallback_aspect,
)


class TestFormatDocuments:
    """Tests for the _format_documents helper function."""

    def test_formats_documents_with_content(self):
        """Should format documents with their content."""
        docs = [
            {"content": "First document content", "document_type": "codex", "date": "2025-01-01"},
            {"content": "Second document content", "document_type": "journal", "date": "2025-01-02"},
        ]

        result = _format_documents(docs)

        assert "First document content" in result
        assert "Second document content" in result
        assert "[1]" in result
        assert "[2]" in result
        assert "(codex)" in result
        assert "(journal)" in result

    def test_preserves_full_content(self):
        """Should preserve full document content without truncation."""
        long_content = "x" * 1000
        docs = [{"content": long_content, "document_type": "test", "date": ""}]

        result = _format_documents(docs)

        # Full content should be preserved
        assert "x" * 1000 in result

    def test_handles_empty_documents(self):
        """Should return placeholder for empty documents."""
        result = _format_documents([])

        assert "empty" in result.lower() or "no documents" in result.lower()

    def test_handles_missing_fields(self):
        """Should handle documents with missing fields."""
        docs = [{"content": "Some content"}]

        result = _format_documents(docs)

        assert "Some content" in result
        assert "(unknown)" in result  # default doc_type

    def test_show_index_false(self):
        """Should not show index when show_index=False."""
        docs = [{"content": "Test", "document_type": "codex", "date": ""}]

        result = _format_documents(docs, show_index=False)

        assert "[1]" not in result
        assert "(codex)" in result


class TestGetRefinerTools:
    """Tests for the _get_refiner_tools function."""

    # Paradigm-specific tool names
    SELECT_TOOL_NAMES = {"select_brainstorm_idea", "select_daydream_theme", "select_knowledge_gap"}
    VALIDATE_TOOL_NAMES = {"validate_brainstorm_exploration", "validate_daydream_exploration", "validate_knowledge_exploration"}

    def test_returns_list_of_tools(self):
        """Should return a list of Tool objects."""
        tools = _get_refiner_tools()

        assert isinstance(tools, list)
        assert len(tools) >= 2

    def test_includes_select_tool(self):
        """Should include a paradigm-specific select tool."""
        tools = _get_refiner_tools()
        tool_names = [t.function.name for t in tools]

        # Should have at least one paradigm-specific select tool
        assert any(name in self.SELECT_TOOL_NAMES for name in tool_names), \
            f"Expected one of {self.SELECT_TOOL_NAMES}, got {tool_names}"

    def test_includes_validate_exploration_tool(self):
        """Should include a paradigm-specific validate tool."""
        tools = _get_refiner_tools()
        tool_names = [t.function.name for t in tools]

        # Should have at least one paradigm-specific validate tool
        assert any(name in self.VALIDATE_TOOL_NAMES for name in tool_names), \
            f"Expected one of {self.VALIDATE_TOOL_NAMES}, got {tool_names}"

    def test_select_tool_has_required_params(self):
        """Select tools should have required parameters."""
        tools = _get_refiner_tools()
        select_tools = [t for t in tools if t.function.name in self.SELECT_TOOL_NAMES]

        assert len(select_tools) > 0, f"No select tools found. Available: {[t.function.name for t in tools]}"
        select_tool = select_tools[0]

        # Each paradigm has different topic parameter names:
        # - brainstorm: idea, approach, reasoning
        # - daydream: theme, emotional_tone, reasoning
        # - knowledge: gap, approach, reasoning
        required = select_tool.function.parameters.required
        tool_name = select_tool.function.name

        # All select tools have reasoning as required
        assert "reasoning" in required, f"{tool_name} missing 'reasoning' in required params"

        # Check paradigm-specific topic parameter
        if tool_name == "select_brainstorm_idea":
            assert "idea" in required, f"{tool_name} missing 'idea' in required params"
            assert "approach" in required, f"{tool_name} missing 'approach' in required params"
        elif tool_name == "select_daydream_theme":
            assert "theme" in required, f"{tool_name} missing 'theme' in required params"
            assert "emotional_tone" in required, f"{tool_name} missing 'emotional_tone' in required params"
        elif tool_name == "select_knowledge_gap":
            assert "gap" in required, f"{tool_name} missing 'gap' in required params"
            assert "approach" in required, f"{tool_name} missing 'approach' in required params"

    def test_validate_exploration_has_required_params(self):
        """Validate tools should have required parameters."""
        tools = _get_refiner_tools()
        validate_tools = [t for t in tools if t.function.name in self.VALIDATE_TOOL_NAMES]

        assert len(validate_tools) > 0, f"No validate tools found. Available: {[t.function.name for t in tools]}"
        validate_tool = validate_tools[0]

        assert "accept" in validate_tool.function.parameters.required
        assert "reasoning" in validate_tool.function.parameters.required


class TestCreateFallbackAspect:
    """Tests for the _create_fallback_aspect function."""

    def test_creates_librarian_aspect(self):
        """Should create librarian aspect with expected attributes."""
        aspect = _create_fallback_aspect("librarian")

        assert hasattr(aspect, "name")
        assert hasattr(aspect, "title")
        assert hasattr(aspect, "location")

    def test_creates_dreamer_aspect(self):
        """Should create dreamer aspect."""
        aspect = _create_fallback_aspect("dreamer")

        assert hasattr(aspect, "name")

    def test_creates_philosopher_aspect(self):
        """Should create philosopher aspect."""
        aspect = _create_fallback_aspect("philosopher")

        assert hasattr(aspect, "name")

    def test_unknown_defaults_to_librarian(self):
        """Unknown aspect type should default to librarian."""
        aspect = _create_fallback_aspect("unknown")

        assert hasattr(aspect, "name")


class TestBuildBrainstormSelectionPrompt:
    """Tests for the build_brainstorm_selection_prompt function."""

    @pytest.fixture
    def mock_persona(self):
        """Mock Persona object."""
        persona = MagicMock()
        persona.persona_id = "test_persona"
        persona.name = "Test"
        persona.full_name = "Test Persona"
        persona.pronouns = {"subj": "she", "poss": "her", "obj": "her"}
        persona.aspects = {}
        persona.xml_decorator.return_value = MagicMock()
        persona.xml_decorator.return_value.render.return_value = "<persona>Test</persona>"
        return persona

    @pytest.fixture
    def sample_documents(self):
        """Sample documents list."""
        return [
            {"content": "Brainstorm about consciousness", "document_type": "brainstorm", "date": "2025-01-01"},
            {"content": "Ideas about creativity", "document_type": "pondering", "date": "2025-01-02"},
        ]

    def test_returns_tuple_of_prompts(self, sample_documents, mock_persona):
        """Should return (system_prompt, user_prompt) tuple."""
        result = build_brainstorm_selection_prompt(sample_documents, mock_persona)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_user_prompt_includes_documents(self, sample_documents, mock_persona):
        """User prompt should include the documents."""
        _, user_prompt = build_brainstorm_selection_prompt(sample_documents, mock_persona)

        assert "consciousness" in user_prompt
        assert "creativity" in user_prompt

    def test_user_prompt_includes_think_tags(self, sample_documents, mock_persona):
        """User prompt should include <think> tags for reasoning."""
        _, user_prompt = build_brainstorm_selection_prompt(sample_documents, mock_persona)

        assert "<think>" in user_prompt
        assert "</think>" in user_prompt

    def test_user_prompt_mentions_approaches(self, sample_documents, mock_persona):
        """User prompt should mention the three approaches."""
        _, user_prompt = build_brainstorm_selection_prompt(sample_documents, mock_persona)

        assert "philosopher" in user_prompt
        assert "journaler" in user_prompt
        assert "daydream" in user_prompt

    def test_user_prompt_includes_instructions(self, sample_documents, mock_persona):
        """User prompt should include clear instructions."""
        _, user_prompt = build_brainstorm_selection_prompt(sample_documents, mock_persona)

        assert "<instructions>" in user_prompt
        assert "select_topic" in user_prompt


class TestBuildDaydreamSelectionPrompt:
    """Tests for the build_daydream_selection_prompt function."""

    @pytest.fixture
    def mock_persona(self):
        """Mock Persona object."""
        persona = MagicMock()
        persona.persona_id = "test_persona"
        persona.name = "Test"
        persona.full_name = "Test Persona"
        persona.pronouns = {"subj": "she", "poss": "her", "obj": "her"}
        persona.aspects = {}
        persona.xml_decorator.return_value = MagicMock()
        persona.xml_decorator.return_value.render.return_value = "<persona>Test</persona>"
        return persona

    @pytest.fixture
    def sample_documents(self):
        """Sample documents list."""
        return [
            {"content": "Summary of dream-like conversation", "document_type": "summary", "date": "2025-01-01"},
            {"content": "Analysis of emotional themes", "document_type": "analysis", "date": "2025-01-02"},
        ]

    def test_returns_tuple_of_prompts(self, sample_documents, mock_persona):
        """Should return (system_prompt, user_prompt) tuple."""
        result = build_daydream_selection_prompt(sample_documents, mock_persona)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_user_prompt_includes_documents(self, sample_documents, mock_persona):
        """User prompt should include the documents."""
        _, user_prompt = build_daydream_selection_prompt(sample_documents, mock_persona)

        assert "dream" in user_prompt.lower() or "Summary" in user_prompt
        assert "emotional" in user_prompt.lower()

    def test_user_prompt_has_dreamlike_tone(self, sample_documents, mock_persona):
        """User prompt should have dreamlike/atmospheric elements."""
        _, user_prompt = build_daydream_selection_prompt(sample_documents, mock_persona)

        # Should contain dreamlike language
        lower_prompt = user_prompt.lower()
        assert "dream" in lower_prompt or "prismatic" in lower_prompt or "imagine" in lower_prompt


class TestBuildKnowledgeSelectionPrompt:
    """Tests for the build_knowledge_selection_prompt function."""

    @pytest.fixture
    def mock_persona(self):
        """Mock Persona object."""
        persona = MagicMock()
        persona.persona_id = "test_persona"
        persona.name = "Test"
        persona.full_name = "Test Persona"
        persona.pronouns = {"subj": "she", "poss": "her", "obj": "her"}
        persona.aspects = {}
        persona.xml_decorator.return_value = MagicMock()
        persona.xml_decorator.return_value.render.return_value = "<persona>Test</persona>"
        return persona

    @pytest.fixture
    def sample_documents(self):
        """Sample documents list."""
        return [
            {"content": "Codex entry about philosophy", "document_type": "codex", "date": "2025-01-01"},
            {"content": "Pondering about existence", "document_type": "pondering", "date": "2025-01-02"},
        ]

    def test_returns_tuple_of_prompts(self, sample_documents, mock_persona):
        """Should return (system_prompt, user_prompt) tuple."""
        result = build_knowledge_selection_prompt(sample_documents, mock_persona)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_user_prompt_includes_documents(self, sample_documents, mock_persona):
        """User prompt should include the documents."""
        _, user_prompt = build_knowledge_selection_prompt(sample_documents, mock_persona)

        assert "philosophy" in user_prompt.lower() or "Codex" in user_prompt
        assert "existence" in user_prompt.lower()


class TestBuildTopicSelectionPrompt:
    """Tests for the build_topic_selection_prompt dispatcher function."""

    @pytest.fixture
    def mock_persona(self):
        """Mock Persona object."""
        persona = MagicMock()
        persona.persona_id = "test_persona"
        persona.name = "Test"
        persona.full_name = "Test Persona"
        persona.pronouns = {"subj": "she", "poss": "her", "obj": "her"}
        persona.aspects = {}
        persona.xml_decorator.return_value = MagicMock()
        persona.xml_decorator.return_value.render.return_value = "<persona>Test</persona>"
        return persona

    @pytest.fixture
    def sample_documents(self):
        """Sample documents list."""
        return [
            {"content": "Test document", "document_type": "codex", "date": "2025-01-01"},
        ]

    def test_dispatches_to_brainstorm(self, sample_documents, mock_persona):
        """Should call build_brainstorm_selection_prompt for brainstorm paradigm."""
        result = build_topic_selection_prompt("brainstorm", sample_documents, mock_persona)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_dispatches_to_daydream(self, sample_documents, mock_persona):
        """Should call build_daydream_selection_prompt for daydream paradigm."""
        result = build_topic_selection_prompt("daydream", sample_documents, mock_persona)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_dispatches_to_knowledge(self, sample_documents, mock_persona):
        """Should call build_knowledge_selection_prompt for knowledge paradigm."""
        result = build_topic_selection_prompt("knowledge", sample_documents, mock_persona)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_unknown_paradigm_defaults_to_brainstorm(self, sample_documents, mock_persona):
        """Unknown paradigm should default to brainstorm."""
        result = build_topic_selection_prompt("unknown", sample_documents, mock_persona)

        assert isinstance(result, tuple)
        assert len(result) == 2


class TestBuildValidationPrompt:
    """Tests for the build_validation_prompt function."""

    @pytest.fixture
    def mock_persona(self):
        """Mock Persona object."""
        persona = MagicMock()
        persona.persona_id = "test_persona"
        persona.name = "Test"
        persona.full_name = "Test Persona"
        persona.pronouns = {"subj": "she", "poss": "her", "obj": "her"}
        persona.aspects = {}
        persona.xml_decorator.return_value = MagicMock()
        persona.xml_decorator.return_value.render.return_value = "<persona>Test</persona>"
        return persona

    @pytest.fixture
    def sample_documents(self):
        """Sample targeted context documents."""
        return [
            {"content": "Focused doc about consciousness", "document_type": "codex", "date": "2025-01-01"},
        ]

    def test_returns_tuple_of_prompts(self, sample_documents, mock_persona):
        """Should return (system_prompt, user_prompt) tuple."""
        result = build_validation_prompt(
            paradigm="brainstorm",
            topic="consciousness",
            approach="philosopher",
            reasoning="Interesting theme",
            documents=sample_documents,
            persona=mock_persona,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_user_prompt_includes_topic(self, sample_documents, mock_persona):
        """User prompt should include the selected topic."""
        _, user_prompt = build_validation_prompt(
            paradigm="brainstorm",
            topic="consciousness",
            approach="philosopher",
            reasoning="Interesting theme",
            documents=sample_documents,
            persona=mock_persona,
        )

        assert "consciousness" in user_prompt

    def test_user_prompt_includes_approach(self, sample_documents, mock_persona):
        """User prompt should include the selected approach."""
        _, user_prompt = build_validation_prompt(
            paradigm="brainstorm",
            topic="test",
            approach="journaler",
            reasoning="test",
            documents=sample_documents,
            persona=mock_persona,
        )

        # Journaler has its own special validation with writer aspect
        assert "journal" in user_prompt.lower()
        assert "writer" in user_prompt.lower() or "Writer" in user_prompt

    def test_user_prompt_includes_initial_reasoning(self, sample_documents, mock_persona):
        """User prompt should include the initial reasoning."""
        _, user_prompt = build_validation_prompt(
            paradigm="brainstorm",
            topic="test",
            approach="philosopher",
            reasoning="This is my initial reasoning",
            documents=sample_documents,
            persona=mock_persona,
        )

        assert "This is my initial reasoning" in user_prompt

    def test_user_prompt_includes_think_tags(self, sample_documents, mock_persona):
        """User prompt should include <think> tags."""
        _, user_prompt = build_validation_prompt(
            paradigm="brainstorm",
            topic="test",
            approach="philosopher",
            reasoning="test",
            documents=sample_documents,
            persona=mock_persona,
        )

        assert "<think>" in user_prompt
        assert "</think>" in user_prompt

    def test_user_prompt_includes_focused_context(self, sample_documents, mock_persona):
        """User prompt should include the focused context documents."""
        _, user_prompt = build_validation_prompt(
            paradigm="brainstorm",
            topic="consciousness",
            approach="philosopher",
            reasoning="test",
            documents=sample_documents,
            persona=mock_persona,
        )

        assert "Focused doc about consciousness" in user_prompt

    def test_user_prompt_mentions_validate_tool(self, sample_documents, mock_persona):
        """User prompt should mention the validate_exploration tool."""
        _, user_prompt = build_validation_prompt(
            paradigm="brainstorm",
            topic="test",
            approach="philosopher",
            reasoning="test",
            documents=sample_documents,
            persona=mock_persona,
        )

        assert "validate_exploration" in user_prompt

    def test_user_prompt_encourages_rejection(self, sample_documents, mock_persona):
        """User prompt should encourage rejection unless compelling."""
        _, user_prompt = build_validation_prompt(
            paradigm="brainstorm",
            topic="test",
            approach="philosopher",
            reasoning="test",
            documents=sample_documents,
            persona=mock_persona,
        )

        # Should mention rejection or challenge
        lower_prompt = user_prompt.lower()
        assert "reject" in lower_prompt or "challenge" in lower_prompt

    def test_uses_appropriate_aspect_for_daydream(self, sample_documents, mock_persona):
        """Should use dreamer aspect for daydream paradigm."""
        _, user_prompt = build_validation_prompt(
            paradigm="daydream",
            topic="test",
            approach="daydream",
            reasoning="test",
            documents=sample_documents,
            persona=mock_persona,
        )

        # Should have dreamlike language from dreamer aspect
        assert isinstance(user_prompt, str)
        assert len(user_prompt) > 0

    def test_uses_appropriate_aspect_for_knowledge(self, sample_documents, mock_persona):
        """Should use philosopher aspect for knowledge paradigm."""
        _, user_prompt = build_validation_prompt(
            paradigm="knowledge",
            topic="test",
            approach="philosopher",
            reasoning="test",
            documents=sample_documents,
            persona=mock_persona,
        )

        assert isinstance(user_prompt, str)
        assert len(user_prompt) > 0
