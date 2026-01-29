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

    def test_load_brainstorm(self, repo_root_cwd):
        """Should load brainstorm paradigm from config."""
        from aim_legacy.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("brainstorm")
        assert paradigm.name == "brainstorm"
        assert paradigm.aspect == "librarian"

    def test_load_daydream(self, repo_root_cwd):
        """Should load daydream paradigm from config."""
        from aim_legacy.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("daydream")
        assert paradigm.name == "daydream"
        assert paradigm.aspect == "dreamer"

    def test_load_knowledge(self, repo_root_cwd):
        """Should load knowledge paradigm from config."""
        from aim_legacy.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("knowledge")
        assert paradigm.name == "knowledge"
        assert paradigm.aspect == "philosopher"

    def test_load_critique(self, repo_root_cwd):
        """Should load critique paradigm from config."""
        from aim_legacy.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("critique")
        assert paradigm.name == "critique"
        assert paradigm.aspect == "psychologist"

    def test_load_journaler(self, repo_root_cwd):
        """Should load journaler paradigm from config."""
        from aim_legacy.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("journaler")
        assert paradigm.name == "journaler"
        assert paradigm.aspect == "writer"

    def test_load_invalid_raises(self):
        """Should raise ValueError for non-existent paradigm."""
        from aim_legacy.refiner.paradigm import Paradigm

        with pytest.raises(ValueError, match="No config found"):
            Paradigm.load("nonexistent")


class TestParadigmAvailable:
    """Tests for discovering available paradigms."""

    def test_available_returns_list(self, repo_root_cwd):
        """available() should return list of paradigm names."""
        from aim_legacy.refiner.paradigm import Paradigm

        available = Paradigm.available()
        assert isinstance(available, list)
        assert len(available) >= 4  # At least brainstorm, daydream, knowledge, critique

    def test_available_contains_expected(self, repo_root_cwd):
        """available() should include known paradigms."""
        from aim_legacy.refiner.paradigm import Paradigm

        available = Paradigm.available()
        assert "brainstorm" in available
        assert "daydream" in available
        assert "knowledge" in available
        assert "critique" in available

    def test_available_with_exclude(self, repo_root_cwd):
        """available(exclude=...) should filter out specified paradigms."""
        from aim_legacy.refiner.paradigm import Paradigm

        available = Paradigm.available(exclude=["journaler"])
        assert "journaler" not in available
        assert "brainstorm" in available


class TestParadigmScenarioRouting:
    """Tests for scenario routing logic."""

    def test_brainstorm_routes_by_approach(self, repo_root_cwd):
        """Brainstorm should route based on approach."""
        from aim_legacy.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("brainstorm")
        assert paradigm.get_scenario("philosopher") == "philosopher_dialogue"
        assert paradigm.get_scenario("journaler") == "journaler_dialogue"

    def test_daydream_always_routes_to_daydream(self, repo_root_cwd):
        """Daydream should always route to daydream_dialogue scenario."""
        from aim_legacy.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("daydream")
        assert paradigm.get_scenario("daydream") == "daydream_dialogue"
        assert paradigm.get_scenario(None) == "daydream_dialogue"

    def test_knowledge_routes_to_researcher_or_approach(self, repo_root_cwd):
        """Knowledge should route based on approach to dialogue scenarios."""
        from aim_legacy.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("knowledge")
        # Knowledge has scenarios_by_approach, now routing to dialogue scenarios
        assert paradigm.get_scenario("philosopher") == "philosopher_dialogue"
        assert paradigm.get_scenario("journaler") == "journaler_dialogue"
        assert paradigm.get_scenario("researcher") == "researcher_dialogue"

    def test_critique_routes_to_critique(self, repo_root_cwd):
        """Critique should route to critique_dialogue scenario."""
        from aim_legacy.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("critique")
        assert paradigm.get_scenario("critique") == "critique_dialogue"


class TestParadigmDocTypes:
    """Tests for document type configuration."""

    def test_brainstorm_doc_types(self, repo_root_cwd):
        """Brainstorm should have expected doc_types."""
        from aim_legacy.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("brainstorm")
        assert len(paradigm.doc_types) > 0
        assert "brainstorm" in paradigm.doc_types

    def test_approach_doc_types(self, repo_root_cwd):
        """Should return approach-specific doc types."""
        from aim_legacy.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("brainstorm")
        philosopher_types = paradigm.get_approach_doc_types("philosopher")
        assert len(philosopher_types) > 0


class TestParadigmTools:
    """Tests for tool definitions."""

    def test_get_select_tool(self, repo_root_cwd):
        """Should return select_topic tool."""
        from aim_legacy.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("brainstorm")
        tool = paradigm.get_select_tool()
        assert tool.function.name == "select_topic"

    def test_get_validate_tool(self, repo_root_cwd):
        """Should return validate_exploration tool."""
        from aim_legacy.refiner.paradigm import Paradigm

        paradigm = Paradigm.load("brainstorm")
        tool = paradigm.get_validate_tool()
        assert tool.function.name == "validate_exploration"


class TestParadigmImports:
    """Tests for Paradigm imports."""

    def test_import_from_paradigm_module(self):
        """Should be importable from aim_legacy.refiner.paradigm."""
        from aim_legacy.refiner.paradigm import Paradigm

        assert Paradigm is not None
