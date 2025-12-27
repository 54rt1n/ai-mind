# tests/unit/refiner/test_paradigm.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for the paradigm module - ExplorationPlan dataclass.

The ExplorationPlan dataclass is used by the ExplorationEngine to
represent the result of an exploration decision, containing the
scenario to run, query text, and optional guidance/reasoning.
"""

import pytest
from dataclasses import is_dataclass


class TestExplorationPlan:
    """Tests for the ExplorationPlan dataclass."""

    def test_exploration_plan_required_fields(self):
        """ExplorationPlan should require scenario and query_text."""
        from aim.refiner.paradigm import ExplorationPlan

        plan = ExplorationPlan(
            scenario="philosopher",
            query_text="What is consciousness?",
        )

        assert plan.scenario == "philosopher"
        assert plan.query_text == "What is consciousness?"

    def test_exploration_plan_optional_fields_default_none(self):
        """Optional fields should default to None."""
        from aim.refiner.paradigm import ExplorationPlan

        plan = ExplorationPlan(
            scenario="journaler",
            query_text="Reflect on today",
        )

        assert plan.guidance is None
        assert plan.reasoning is None

    def test_exploration_plan_all_fields(self):
        """ExplorationPlan should accept all fields."""
        from aim.refiner.paradigm import ExplorationPlan

        plan = ExplorationPlan(
            scenario="daydream",
            query_text="Imagine a peaceful garden",
            guidance="Focus on sensory details",
            reasoning="Recent conversations mentioned nature",
        )

        assert plan.scenario == "daydream"
        assert plan.query_text == "Imagine a peaceful garden"
        assert plan.guidance == "Focus on sensory details"
        assert plan.reasoning == "Recent conversations mentioned nature"

    def test_exploration_plan_valid_scenarios(self):
        """Should accept valid scenario names."""
        from aim.refiner.paradigm import ExplorationPlan

        valid_scenarios = ["philosopher", "journaler", "daydream"]

        for scenario in valid_scenarios:
            plan = ExplorationPlan(scenario=scenario, query_text="test")
            assert plan.scenario == scenario

    def test_exploration_plan_is_dataclass(self):
        """ExplorationPlan should be a dataclass."""
        from aim.refiner.paradigm import ExplorationPlan

        assert is_dataclass(ExplorationPlan)

    def test_exploration_plan_equality(self):
        """Two ExplorationPlans with same values should be equal."""
        from aim.refiner.paradigm import ExplorationPlan

        plan1 = ExplorationPlan(
            scenario="philosopher",
            query_text="What is consciousness?",
            guidance="Be thorough",
            reasoning="Important topic",
        )
        plan2 = ExplorationPlan(
            scenario="philosopher",
            query_text="What is consciousness?",
            guidance="Be thorough",
            reasoning="Important topic",
        )

        assert plan1 == plan2

    def test_exploration_plan_inequality(self):
        """Two ExplorationPlans with different values should not be equal."""
        from aim.refiner.paradigm import ExplorationPlan

        plan1 = ExplorationPlan(scenario="philosopher", query_text="What is consciousness?")
        plan2 = ExplorationPlan(scenario="journaler", query_text="What is consciousness?")

        assert plan1 != plan2

    def test_exploration_plan_repr(self):
        """ExplorationPlan should have a readable repr."""
        from aim.refiner.paradigm import ExplorationPlan

        plan = ExplorationPlan(scenario="daydream", query_text="Floating in clouds")

        repr_str = repr(plan)

        assert "ExplorationPlan" in repr_str
        assert "daydream" in repr_str
        assert "Floating in clouds" in repr_str

    def test_exploration_plan_can_be_created_with_kwargs(self):
        """ExplorationPlan should accept keyword arguments."""
        from aim.refiner.paradigm import ExplorationPlan

        plan = ExplorationPlan(
            query_text="A query",
            scenario="philosopher",
            reasoning="Some reason",
        )

        assert plan.scenario == "philosopher"
        assert plan.query_text == "A query"
        assert plan.reasoning == "Some reason"

    def test_exploration_plan_empty_strings_allowed(self):
        """ExplorationPlan should allow empty strings (though not recommended)."""
        from aim.refiner.paradigm import ExplorationPlan

        plan = ExplorationPlan(scenario="", query_text="")

        assert plan.scenario == ""
        assert plan.query_text == ""

    def test_exploration_plan_long_query_text(self):
        """ExplorationPlan should handle long query text."""
        from aim.refiner.paradigm import ExplorationPlan

        long_query = "x" * 10000
        plan = ExplorationPlan(scenario="philosopher", query_text=long_query)

        assert plan.query_text == long_query
        assert len(plan.query_text) == 10000


class TestExplorationPlanImports:
    """Tests for ExplorationPlan imports from various locations."""

    def test_import_from_paradigm_module(self):
        """Should be importable from aim.refiner.paradigm."""
        from aim.refiner.paradigm import ExplorationPlan

        assert ExplorationPlan is not None

    def test_import_from_refiner_package(self):
        """Should be importable from aim.refiner."""
        from aim.refiner import ExplorationPlan

        assert ExplorationPlan is not None

    def test_both_imports_are_same_class(self):
        """Both import paths should reference the same class."""
        from aim.refiner.paradigm import ExplorationPlan as EP1
        from aim.refiner import ExplorationPlan as EP2

        assert EP1 is EP2
