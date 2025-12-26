# tests/integration/dreamer/test_scenario_flow.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Integration tests for scenario loading and step flow validation.

These tests verify that each scenario:
1. Loads correctly from YAML
2. Has valid DAG structure (no orphan steps, proper dependencies)
3. Computes dependencies correctly
4. Has proper context DSL on root steps
5. Steps execute in correct topological order
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
import pandas as pd

from aim.dreamer.scenario import load_scenario
from aim.dreamer.models import (
    Scenario,
    StepDefinition,
    StepStatus,
    PipelineState,
    StepResult,
    ContextAction,
)
from aim.dreamer.context import prepare_step_context


# All production scenarios to test
PRODUCTION_SCENARIOS = [
    "analyst",
    "summarizer",
    "daydream",
    "philosopher",
    "journaler",
]


class TestScenarioLoading:
    """Test that all scenarios load correctly."""

    @pytest.mark.parametrize("scenario_name", PRODUCTION_SCENARIOS)
    def test_scenario_loads_successfully(self, scenario_name: str):
        """Each scenario should load without errors."""
        scenario = load_scenario(scenario_name)

        assert scenario is not None
        assert isinstance(scenario, Scenario)
        assert scenario.name == scenario_name
        assert scenario.version == 2

    @pytest.mark.parametrize("scenario_name", PRODUCTION_SCENARIOS)
    def test_scenario_has_steps(self, scenario_name: str):
        """Each scenario should have at least one step."""
        scenario = load_scenario(scenario_name)

        assert len(scenario.steps) > 0

    @pytest.mark.parametrize("scenario_name", PRODUCTION_SCENARIOS)
    def test_scenario_has_required_aspects(self, scenario_name: str):
        """Each scenario should declare required aspects."""
        scenario = load_scenario(scenario_name)

        assert scenario.context is not None
        assert isinstance(scenario.context.required_aspects, list)


class TestScenarioDAGStructure:
    """Test DAG structure validity for all scenarios."""

    @pytest.mark.parametrize("scenario_name", PRODUCTION_SCENARIOS)
    def test_scenario_has_root_steps(self, scenario_name: str):
        """Each scenario should have at least one root step (no dependencies)."""
        scenario = load_scenario(scenario_name)
        scenario.compute_dependencies()

        roots = scenario.get_root_steps()

        assert len(roots) > 0, f"Scenario {scenario_name} has no root steps"

    @pytest.mark.parametrize("scenario_name", PRODUCTION_SCENARIOS)
    def test_scenario_has_terminal_steps(self, scenario_name: str):
        """Each scenario should have at least one terminal step (next=[])."""
        scenario = load_scenario(scenario_name)

        terminal_steps = [
            step_id for step_id, step in scenario.steps.items()
            if not step.next
        ]

        assert len(terminal_steps) > 0, f"Scenario {scenario_name} has no terminal steps"

    @pytest.mark.parametrize("scenario_name", PRODUCTION_SCENARIOS)
    def test_next_references_valid_steps(self, scenario_name: str):
        """All 'next' references should point to valid step IDs."""
        scenario = load_scenario(scenario_name)

        step_ids = set(scenario.steps.keys())

        for step_id, step in scenario.steps.items():
            for next_step_id in step.next:
                assert next_step_id in step_ids, (
                    f"Step {step_id} in {scenario_name} references invalid step {next_step_id}"
                )

    @pytest.mark.parametrize("scenario_name", PRODUCTION_SCENARIOS)
    def test_depends_on_references_valid_steps(self, scenario_name: str):
        """All 'depends_on' references should point to valid step IDs."""
        scenario = load_scenario(scenario_name)

        step_ids = set(scenario.steps.keys())

        for step_id, step in scenario.steps.items():
            for dep_id in step.depends_on:
                assert dep_id in step_ids, (
                    f"Step {step_id} in {scenario_name} depends on invalid step {dep_id}"
                )

    @pytest.mark.parametrize("scenario_name", PRODUCTION_SCENARIOS)
    def test_no_self_references(self, scenario_name: str):
        """No step should reference itself in next or depends_on."""
        scenario = load_scenario(scenario_name)

        for step_id, step in scenario.steps.items():
            assert step_id not in step.next, (
                f"Step {step_id} in {scenario_name} references itself in 'next'"
            )
            assert step_id not in step.depends_on, (
                f"Step {step_id} in {scenario_name} references itself in 'depends_on'"
            )

    @pytest.mark.parametrize("scenario_name", PRODUCTION_SCENARIOS)
    def test_no_cycles(self, scenario_name: str):
        """DAG should have no cycles (topological sort should succeed)."""
        scenario = load_scenario(scenario_name)
        scenario.compute_dependencies()

        # topological_order() will raise ValueError if there's a cycle
        order = scenario.topological_order()

        assert len(order) == len(scenario.steps), (
            f"Scenario {scenario_name} topological order incomplete"
        )


class TestDependencyComputation:
    """Test that dependencies are computed correctly from 'next' edges."""

    @pytest.mark.parametrize("scenario_name", PRODUCTION_SCENARIOS)
    def test_compute_dependencies_infers_from_next(self, scenario_name: str):
        """compute_dependencies should infer depends_on from next edges."""
        scenario = load_scenario(scenario_name)

        # Clear any existing depends_on to test inference
        for step in scenario.steps.values():
            step.depends_on = []

        scenario.compute_dependencies()

        # Verify: if step A has B in next, then B should have A in depends_on
        for step_id, step in scenario.steps.items():
            for next_step_id in step.next:
                next_step = scenario.steps[next_step_id]
                assert step_id in next_step.depends_on, (
                    f"Step {next_step_id} should depend on {step_id} in {scenario_name}"
                )


class TestTopologicalOrder:
    """Test topological ordering of steps."""

    @pytest.mark.parametrize("scenario_name", PRODUCTION_SCENARIOS)
    def test_topological_order_respects_dependencies(self, scenario_name: str):
        """Steps should appear after their dependencies in topological order."""
        scenario = load_scenario(scenario_name)
        scenario.compute_dependencies()

        order = scenario.topological_order()
        position = {step_id: i for i, step_id in enumerate(order)}

        for step_id, step in scenario.steps.items():
            for dep_id in step.depends_on:
                assert position[dep_id] < position[step_id], (
                    f"In {scenario_name}: {dep_id} should come before {step_id}"
                )

    @pytest.mark.parametrize("scenario_name", PRODUCTION_SCENARIOS)
    def test_topological_order_includes_all_steps(self, scenario_name: str):
        """Topological order should include every step exactly once."""
        scenario = load_scenario(scenario_name)
        scenario.compute_dependencies()

        order = scenario.topological_order()

        assert set(order) == set(scenario.steps.keys()), (
            f"Topological order missing steps in {scenario_name}"
        )


class TestContextDSL:
    """Test that context DSL is properly configured on root steps."""

    @pytest.mark.parametrize("scenario_name", PRODUCTION_SCENARIOS)
    def test_root_steps_have_context_dsl(self, scenario_name: str):
        """Root steps should have context DSL to load initial context."""
        scenario = load_scenario(scenario_name)
        scenario.compute_dependencies()

        roots = scenario.get_root_steps()

        for root_id in roots:
            root_step = scenario.steps[root_id]
            assert root_step.context is not None, (
                f"Root step {root_id} in {scenario_name} has no context DSL"
            )
            assert len(root_step.context) > 0, (
                f"Root step {root_id} in {scenario_name} has empty context DSL"
            )

    @pytest.mark.parametrize("scenario_name", PRODUCTION_SCENARIOS)
    def test_context_dsl_has_load_action(self, scenario_name: str):
        """Root steps should have at least one load_conversation or query action."""
        scenario = load_scenario(scenario_name)
        scenario.compute_dependencies()

        roots = scenario.get_root_steps()

        for root_id in roots:
            root_step = scenario.steps[root_id]
            if root_step.context:
                load_actions = [
                    a for a in root_step.context
                    if a.action in ("load_conversation", "query")
                ]
                assert len(load_actions) > 0, (
                    f"Root step {root_id} in {scenario_name} has no load/query action"
                )

    @pytest.mark.parametrize("scenario_name", PRODUCTION_SCENARIOS)
    def test_context_actions_are_valid(self, scenario_name: str):
        """All context actions should have valid action types."""
        scenario = load_scenario(scenario_name)

        valid_actions = {"load_conversation", "query", "sort", "filter"}

        for step_id, step in scenario.steps.items():
            if step.context:
                for i, action in enumerate(step.context):
                    assert action.action in valid_actions, (
                        f"Step {step_id} in {scenario_name} has invalid action: {action.action}"
                    )


class TestStepFlowExecution:
    """Test that steps flow through correctly with mocked execution."""

    @pytest.mark.parametrize("scenario_name", PRODUCTION_SCENARIOS)
    def test_step_flow_simulation(self, scenario_name: str):
        """Simulate stepping through the scenario to verify flow."""
        scenario = load_scenario(scenario_name)
        scenario.compute_dependencies()

        order = scenario.topological_order()
        completed = set()

        for step_id in order:
            step = scenario.steps[step_id]

            # Verify all dependencies are complete before this step
            for dep_id in step.depends_on:
                assert dep_id in completed, (
                    f"Step {step_id} executed before dependency {dep_id} in {scenario_name}"
                )

            # Mark step as complete
            completed.add(step_id)

        # All steps should be completed
        assert completed == set(scenario.steps.keys())

    @pytest.mark.parametrize("scenario_name", PRODUCTION_SCENARIOS)
    def test_context_accumulation_through_flow(self, scenario_name: str):
        """Test that context accumulates correctly through step flow."""
        scenario = load_scenario(scenario_name)
        scenario.compute_dependencies()

        order = scenario.topological_order()

        # Mock CVM
        mock_cvm = MagicMock()
        mock_cvm.get_conversation_history.return_value = pd.DataFrame({
            'doc_id': ['doc1', 'doc2', 'doc3'],
        })
        mock_cvm.index = MagicMock()
        mock_cvm.index.search.return_value = pd.DataFrame({
            'doc_id': ['doc4', 'doc5'],
        })
        mock_cvm.get_by_doc_id.return_value = {'timestamp': 12345}

        # Create initial state
        state = PipelineState(
            pipeline_id="test",
            scenario_name=scenario_name,
            conversation_id="conv-test",
            persona_id="test",
            user_id="test",
            model="test",
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        # Simulate stepping through with context accumulation
        for step_id in order:
            step = scenario.steps[step_id]

            # Get context for this step
            context_doc_ids, is_initial = prepare_step_context(step, state, mock_cvm)

            # First step with context DSL should return is_initial=True
            if step.context:
                assert is_initial is True, (
                    f"Step {step_id} with context DSL should return is_initial=True"
                )

            # Simulate step execution adding its output
            step_output_doc_id = f"output_{step_id}"

            if is_initial:
                state.context_doc_ids = context_doc_ids + [step_output_doc_id]
            else:
                state.context_doc_ids.append(step_output_doc_id)

        # Final state should have accumulated context from all steps
        assert len(state.context_doc_ids) >= len(order), (
            f"Final context should have at least {len(order)} doc_ids, got {len(state.context_doc_ids)}"
        )


class TestExpectedStepSequences:
    """Test expected step sequences for each scenario."""

    def test_analyst_step_sequence(self):
        """Analyst should flow: ner → emotional_trace → ... → brainstorm."""
        scenario = load_scenario("analyst")
        scenario.compute_dependencies()

        order = scenario.topological_order()

        # First step should be ner
        assert order[0] == "ner"
        # Last step should be brainstorm
        assert order[-1] == "brainstorm"
        # Should have expected steps
        expected_steps = {
            "ner", "emotional_trace", "questions", "reflection",
            "draft", "review", "final_journal", "brainstorm", "codex", "motd"
        }
        assert set(order) == expected_steps

    def test_summarizer_step_sequence(self):
        """Summarizer should flow: timeline → summary → ... → resummarize_2."""
        scenario = load_scenario("summarizer")
        scenario.compute_dependencies()

        order = scenario.topological_order()

        assert order[0] == "timeline"
        assert order[-1] == "resummarize_2"
        expected_steps = {
            "timeline", "summary", "improve_1", "resummarize_1",
            "improve_2", "resummarize_2"
        }
        assert set(order) == expected_steps

    def test_daydream_step_sequence(self):
        """Daydream should flow: intro → agent_1 → partner_1 → ... → partner_3."""
        scenario = load_scenario("daydream")
        scenario.compute_dependencies()

        order = scenario.topological_order()

        assert order[0] == "intro"
        assert order[-1] == "partner_3"
        expected_steps = {
            "intro", "agent_1", "partner_1", "agent_2",
            "partner_2", "agent_3", "partner_3"
        }
        assert set(order) == expected_steps

    def test_philosopher_step_sequence(self):
        """Philosopher should flow: analyze → examine → ... → codex."""
        scenario = load_scenario("philosopher")
        scenario.compute_dependencies()

        order = scenario.topological_order()

        assert order[0] == "analyze"
        assert order[-1] == "codex"
        expected_steps = {
            "analyze", "examine", "keywords", "exposition_1",
            "self_rag", "exposition_2", "review", "draft",
            "final_notes", "brainstorm", "codex"
        }
        assert set(order) == expected_steps

    def test_journaler_step_sequence(self):
        """Journaler should flow: ponder → ner → ... → codex."""
        scenario = load_scenario("journaler")
        scenario.compute_dependencies()

        order = scenario.topological_order()

        assert order[0] == "ponder"
        assert order[-1] == "codex"
        expected_steps = {
            "ponder", "ner", "questions", "reflection",
            "review", "draft", "final_journal", "brainstorm", "codex"
        }
        assert set(order) == expected_steps


class TestStepOutputTypes:
    """Test that steps have appropriate output document types."""

    @pytest.mark.parametrize("scenario_name", PRODUCTION_SCENARIOS)
    def test_all_steps_have_output(self, scenario_name: str):
        """Every step should have an output configuration."""
        scenario = load_scenario(scenario_name)

        for step_id, step in scenario.steps.items():
            assert step.output is not None, (
                f"Step {step_id} in {scenario_name} has no output"
            )
            assert step.output.document_type, (
                f"Step {step_id} in {scenario_name} has no document_type"
            )

    @pytest.mark.parametrize("scenario_name", PRODUCTION_SCENARIOS)
    def test_terminal_steps_have_meaningful_output(self, scenario_name: str):
        """Terminal steps should have meaningful document types (not 'step')."""
        scenario = load_scenario(scenario_name)

        terminal_steps = [
            step_id for step_id, step in scenario.steps.items()
            if not step.next
        ]

        for step_id in terminal_steps:
            step = scenario.steps[step_id]
            # Terminal steps should output something meaningful
            assert step.output.document_type != "step", (
                f"Terminal step {step_id} in {scenario_name} should have meaningful output type"
            )


class TestNoEmptySeedActions:
    """Test that scenarios use step-level context DSL instead of seed actions."""

    @pytest.mark.parametrize("scenario_name", PRODUCTION_SCENARIOS)
    def test_seed_is_empty(self, scenario_name: str):
        """Seed actions should be empty (using step-level context DSL instead)."""
        scenario = load_scenario(scenario_name)

        assert scenario.seed == [], (
            f"Scenario {scenario_name} should have empty seed (use step-level context DSL)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
