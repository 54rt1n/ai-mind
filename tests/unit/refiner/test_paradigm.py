# tests/unit/refiner/test_paradigm.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for the Paradigm class - the strategy pattern for exploration.

The Paradigm class encapsulates all paradigm-specific behavior:
- Document gathering configuration
- Scenario routing
- Prompt building for selection and validation phases
- Tool definitions
"""

import pytest
from unittest.mock import MagicMock


class TestParadigmLoading:
    """Tests for loading Paradigm objects from YAML config."""

    def test_load_brainstorm(self):
        """Should load brainstorm paradigm from config."""
        from aim.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("brainstorm")
        assert paradigm.name == "brainstorm"
        assert paradigm.aspect == "librarian"

    def test_load_daydream(self):
        """Should load daydream paradigm from config."""
        from aim.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("daydream")
        assert paradigm.name == "daydream"
        assert paradigm.aspect == "dreamer"

    def test_load_knowledge(self):
        """Should load knowledge paradigm from config."""
        from aim.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("knowledge")
        assert paradigm.name == "knowledge"
        assert paradigm.aspect == "philosopher"

    def test_load_critique(self):
        """Should load critique paradigm from config."""
        from aim.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("critique")
        assert paradigm.name == "critique"
        assert paradigm.aspect == "psychologist"

    def test_load_journaler(self):
        """Should load journaler paradigm from config."""
        from aim.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("journaler")
        assert paradigm.name == "journaler"
        assert paradigm.aspect == "writer"

    def test_load_invalid_raises(self):
        """Should raise ValueError for non-existent paradigm."""
        from aim.refiner.paradigm import Paradigm

        with pytest.raises(ValueError, match="No config found"):
            Paradigm.load("nonexistent")


class TestParadigmAvailable:
    """Tests for discovering available paradigms."""

    def test_available_returns_list(self):
        """available() should return list of paradigm names."""
        from aim.refiner.paradigm import Paradigm

        available = Paradigm.available()
        assert isinstance(available, list)
        assert len(available) >= 4  # At least brainstorm, daydream, knowledge, critique

    def test_available_contains_expected(self):
        """available() should include known paradigms."""
        from aim.refiner.paradigm import Paradigm

        available = Paradigm.available()
        assert "brainstorm" in available
        assert "daydream" in available
        assert "knowledge" in available
        assert "critique" in available

    def test_available_with_exclude(self):
        """available(exclude=...) should filter out specified paradigms."""
        from aim.refiner.paradigm import Paradigm

        available = Paradigm.available(exclude=["journaler"])
        assert "journaler" not in available
        assert "brainstorm" in available


class TestParadigmScenarioRouting:
    """Tests for scenario routing logic."""

    def test_brainstorm_routes_by_approach(self):
        """Brainstorm should route based on approach."""
        from aim.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("brainstorm")
        assert paradigm.get_scenario("philosopher") == "philosopher_dialogue"
        assert paradigm.get_scenario("journaler") == "journaler_dialogue"

    def test_daydream_always_routes_to_daydream(self):
        """Daydream should always route to daydream_dialogue scenario."""
        from aim.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("daydream")
        assert paradigm.get_scenario("daydream") == "daydream_dialogue"
        assert paradigm.get_scenario(None) == "daydream_dialogue"

    def test_knowledge_routes_to_researcher_or_approach(self):
        """Knowledge should route based on approach or default to researcher."""
        from aim.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("knowledge")
        # Knowledge has scenarios_by_approach
        assert paradigm.get_scenario("philosopher") == "philosopher"
        assert paradigm.get_scenario("journaler") == "journaler"

    def test_critique_routes_to_critique(self):
        """Critique should route to critique_dialogue scenario."""
        from aim.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("critique")
        assert paradigm.get_scenario("critique") == "critique_dialogue"


class TestParadigmDocTypes:
    """Tests for document type configuration."""

    def test_brainstorm_doc_types(self):
        """Brainstorm should have expected doc_types."""
        from aim.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("brainstorm")
        assert len(paradigm.doc_types) > 0
        assert "brainstorm" in paradigm.doc_types

    def test_approach_doc_types(self):
        """Should return approach-specific doc types."""
        from aim.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("brainstorm")
        philosopher_types = paradigm.get_approach_doc_types("philosopher")
        assert len(philosopher_types) > 0


class TestParadigmTools:
    """Tests for tool definitions."""

    def test_get_select_tool(self):
        """Should return select_topic tool."""
        from aim.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("brainstorm")
        tool = paradigm.get_select_tool()
        assert tool.function.name == "select_topic"

    def test_get_validate_tool(self):
        """Should return validate_exploration tool."""
        from aim.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("brainstorm")
        tool = paradigm.get_validate_tool()
        assert tool.function.name == "validate_exploration"


class TestParadigmImports:
    """Tests for Paradigm imports."""

    def test_import_from_paradigm_module(self):
        """Should be importable from aim.refiner.paradigm."""
        from aim.refiner.paradigm import Paradigm

        assert Paradigm is not None

    def test_import_from_refiner_package(self):
        """Should be importable from aim.refiner."""
        from aim.refiner import Paradigm

        assert Paradigm is not None

    def test_both_imports_are_same_class(self):
        """Both import paths should reference the same class."""
        from aim.refiner.paradigm import Paradigm as P1
        from aim.refiner import Paradigm as P2

        assert P1 is P2
