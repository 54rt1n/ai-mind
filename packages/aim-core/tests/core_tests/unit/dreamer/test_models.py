# tests/unit/dreamer/test_models.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for dreamer models."""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from aim.dreamer.core.models import (
    StepStatus,
    StepConfig,
    StepOutput,
    StepDefinition,
    ScenarioContext,
    MemoryAction,
    Scenario,
    StepResult,
    StepJob,
    PipelineState,
    ScenarioTool,
    Condition,
    BaseStepDefinition,
    ContextOnlyStepDefinition,
    StandardStepDefinition,
    ToolCallingStepDefinition,
    RenderingStepDefinition,
    NewStepDefinition,
)


class TestStepStatus:
    """Tests for StepStatus enum."""

    def test_step_status_values(self):
        """Test that all expected status values exist."""
        assert StepStatus.PENDING == "pending"
        assert StepStatus.RUNNING == "running"
        assert StepStatus.COMPLETE == "complete"
        assert StepStatus.FAILED == "failed"


class TestStepConfig:
    """Tests for StepConfig model."""

    def test_step_config_defaults(self):
        """Test default values for StepConfig."""
        config = StepConfig()
        assert config.max_tokens == 1024
        assert config.use_guidance is False
        assert config.is_thought is False
        assert config.is_codex is False
        assert config.temperature is None
        assert config.model_override is None

    def test_step_config_custom_values(self):
        """Test StepConfig with custom values."""
        config = StepConfig(
            max_tokens=2048,
            use_guidance=True,
            is_thought=True,
            temperature=0.7,
            model_override="custom-model"
        )
        assert config.max_tokens == 2048
        assert config.use_guidance is True
        assert config.is_thought is True
        assert config.temperature == 0.7
        assert config.model_override == "custom-model"


class TestStepOutput:
    """Tests for StepOutput model."""

    def test_step_output_required_fields(self):
        """Test StepOutput requires document_type."""
        output = StepOutput(document_type="summary")
        assert output.document_type == "summary"
        assert output.weight == 1.0
        assert output.add_to_turns is True

    def test_step_output_custom_values(self):
        """Test StepOutput with custom values."""
        output = StepOutput(
            document_type="journal",
            weight=0.5,
            add_to_turns=False
        )
        assert output.document_type == "journal"
        assert output.weight == 0.5
        assert output.add_to_turns is False

    def test_step_output_missing_document_type(self):
        """Test that document_type is required."""
        with pytest.raises(ValidationError) as exc_info:
            StepOutput()
        assert "document_type" in str(exc_info.value)


class TestStepDefinition:
    """Tests for StepDefinition model."""

    def test_step_definition_minimal(self):
        """Test StepDefinition with minimal required fields."""
        step = StepDefinition(
            id="step1",
            prompt="Test prompt",
            output=StepOutput(document_type="test")
        )
        assert step.id == "step1"
        assert step.prompt == "Test prompt"
        assert step.output.document_type == "test"
        assert isinstance(step.config, StepConfig)
        assert step.context is None
        assert step.next == []
        assert step.depends_on == []

    def test_step_definition_complete(self):
        """Test StepDefinition with all fields."""
        step = StepDefinition(
            id="step2",
            prompt="Full prompt",
            config=StepConfig(max_tokens=2048, is_thought=True),
            output=StepOutput(document_type="analysis", weight=0.8),
            context=[
                MemoryAction(action="search_memories", top_n=5, document_types=["conversation"]),
            ],
            next=["step3", "step4"],
            depends_on=["step1"]
        )
        assert step.id == "step2"
        assert step.config.max_tokens == 2048
        assert step.config.is_thought is True
        assert step.context is not None
        assert len(step.context) == 1
        assert step.context[0].top_n == 5
        assert step.next == ["step3", "step4"]
        assert step.depends_on == ["step1"]


class TestScenarioContext:
    """Tests for ScenarioContext model."""

    def test_scenario_context_minimal(self):
        """Test ScenarioContext with minimal required fields."""
        context = ScenarioContext(required_aspects=["coder"])
        assert context.required_aspects == ["coder"]
        assert context.core_documents == []
        assert context.enhancement_documents == []
        assert context.location == ""
        assert context.thoughts == []

    def test_scenario_context_complete(self):
        """Test ScenarioContext with all fields."""
        context = ScenarioContext(
            required_aspects=["coder", "librarian"],
            core_documents=["summary", "codex"],
            enhancement_documents=["analysis"],
            location="The lab",
            thoughts=["Focus on details"]
        )
        assert context.required_aspects == ["coder", "librarian"]
        assert context.core_documents == ["summary", "codex"]
        assert context.enhancement_documents == ["analysis"]
        assert context.location == "The lab"
        assert context.thoughts == ["Focus on details"]


class TestMemoryAction:
    """Tests for MemoryAction model."""

    def test_memory_action_load_conversation(self):
        """Test MemoryAction with load_conversation action."""
        action = MemoryAction(
            action="load_conversation",
            document_types=["summary"],
            target="current"
        )
        assert action.action == "load_conversation"
        assert action.document_types == ["summary"]
        assert action.target == "current"

    def test_memory_action_get_memory(self):
        """Test MemoryAction with get_memory action."""
        action = MemoryAction(
            action="get_memory",
            document_types=["journal", "analysis"],
            top_n=10
        )
        assert action.action == "get_memory"
        assert action.document_types == ["journal", "analysis"]
        assert action.top_n == 10

    def test_memory_action_search_memories(self):
        """Test MemoryAction with search_memories action."""
        action = MemoryAction(
            action="search_memories",
            query_text="recent events",
            top_n=100,
            temporal_decay=0.9
        )
        assert action.action == "search_memories"
        assert action.query_text == "recent events"
        assert action.top_n == 100
        assert action.temporal_decay == 0.9

    def test_memory_action_sort(self):
        """Test MemoryAction with sort action."""
        action = MemoryAction(
            action="sort",
            by="timestamp",
            direction="ascending"
        )
        assert action.action == "sort"
        assert action.by == "timestamp"
        assert action.direction == "ascending"

    def test_memory_action_flush(self):
        """Test MemoryAction with flush action."""
        action = MemoryAction(action="flush")
        assert action.action == "flush"

    def test_memory_action_invalid_action(self):
        """Test that invalid action type raises error."""
        with pytest.raises(ValidationError) as exc_info:
            MemoryAction(
                action="invalid_action",  # type: ignore
            )
        assert "action" in str(exc_info.value).lower()


class TestScenario:
    """Tests for Scenario model and DAG operations."""

    def test_scenario_minimal(self):
        """Test Scenario with minimal required fields."""
        scenario = Scenario(
            name="test",
            context=ScenarioContext(required_aspects=["coder"]),
            steps={}
        )
        assert scenario.name == "test"
        assert scenario.version == 2
        assert scenario.description == ""
        assert scenario.seed == []
        assert scenario.steps == {}

    def test_scenario_complete(self):
        """Test Scenario with all fields."""
        scenario = Scenario(
            name="analyst",
            version=2,
            description="Analysis pipeline",
            context=ScenarioContext(required_aspects=["coder"]),
            seed=[
                MemoryAction(
                    action="load_conversation",
                    document_types=["conversation"]
                )
            ],
            steps={
                "step1": StepDefinition(
                    id="step1",
                    prompt="Test",
                    output=StepOutput(document_type="test")
                )
            }
        )
        assert scenario.name == "analyst"
        assert scenario.description == "Analysis pipeline"
        assert len(scenario.seed) == 1
        assert len(scenario.steps) == 1

    def test_get_root_steps_simple(self):
        """Test get_root_steps with simple linear flow."""
        scenario = Scenario(
            name="test",
            context=ScenarioContext(required_aspects=["coder"]),
            steps={
                "step1": StepDefinition(
                    id="step1",
                    prompt="First",
                    output=StepOutput(document_type="test"),
                    next=["step2"]
                ),
                "step2": StepDefinition(
                    id="step2",
                    prompt="Second",
                    output=StepOutput(document_type="test"),
                    depends_on=["step1"]
                )
            }
        )
        roots = scenario.get_root_steps()
        assert roots == ["step1"]

    def test_get_root_steps_multiple(self):
        """Test get_root_steps with multiple root steps."""
        scenario = Scenario(
            name="test",
            context=ScenarioContext(required_aspects=["coder"]),
            steps={
                "step1": StepDefinition(
                    id="step1",
                    prompt="First",
                    output=StepOutput(document_type="test"),
                    next=["step3"]
                ),
                "step2": StepDefinition(
                    id="step2",
                    prompt="Second",
                    output=StepOutput(document_type="test"),
                    next=["step3"]
                ),
                "step3": StepDefinition(
                    id="step3",
                    prompt="Third",
                    output=StepOutput(document_type="test"),
                    depends_on=["step1", "step2"]
                )
            }
        )
        roots = scenario.get_root_steps()
        assert set(roots) == {"step1", "step2"}

    def test_get_downstream(self):
        """Test get_downstream returns correct dependent steps."""
        scenario = Scenario(
            name="test",
            context=ScenarioContext(required_aspects=["coder"]),
            steps={
                "step1": StepDefinition(
                    id="step1",
                    prompt="First",
                    output=StepOutput(document_type="test"),
                    next=["step2", "step3"]
                ),
                "step2": StepDefinition(
                    id="step2",
                    prompt="Second",
                    output=StepOutput(document_type="test"),
                    depends_on=["step1"]
                ),
                "step3": StepDefinition(
                    id="step3",
                    prompt="Third",
                    output=StepOutput(document_type="test"),
                    depends_on=["step1"]
                )
            }
        )
        downstream = scenario.get_downstream("step1")
        assert set(downstream) == {"step2", "step3"}

    def test_compute_dependencies(self):
        """Test compute_dependencies infers depends_on from next."""
        scenario = Scenario(
            name="test",
            context=ScenarioContext(required_aspects=["coder"]),
            steps={
                "step1": StepDefinition(
                    id="step1",
                    prompt="First",
                    output=StepOutput(document_type="test"),
                    next=["step2"]
                ),
                "step2": StepDefinition(
                    id="step2",
                    prompt="Second",
                    output=StepOutput(document_type="test"),
                    next=["step3"]
                ),
                "step3": StepDefinition(
                    id="step3",
                    prompt="Third",
                    output=StepOutput(document_type="test")
                )
            }
        )

        # Before compute
        assert scenario.steps["step2"].depends_on == []
        assert scenario.steps["step3"].depends_on == []

        # Compute dependencies
        scenario.compute_dependencies()

        # After compute
        assert scenario.steps["step2"].depends_on == ["step1"]
        assert scenario.steps["step3"].depends_on == ["step2"]

    def test_compute_dependencies_fan_out(self):
        """Test compute_dependencies with fan-out pattern."""
        scenario = Scenario(
            name="test",
            context=ScenarioContext(required_aspects=["coder"]),
            steps={
                "step1": StepDefinition(
                    id="step1",
                    prompt="Root",
                    output=StepOutput(document_type="test"),
                    next=["step2", "step3", "step4"]
                ),
                "step2": StepDefinition(
                    id="step2",
                    prompt="Branch 1",
                    output=StepOutput(document_type="test")
                ),
                "step3": StepDefinition(
                    id="step3",
                    prompt="Branch 2",
                    output=StepOutput(document_type="test")
                ),
                "step4": StepDefinition(
                    id="step4",
                    prompt="Branch 3",
                    output=StepOutput(document_type="test")
                )
            }
        )

        scenario.compute_dependencies()

        assert scenario.steps["step2"].depends_on == ["step1"]
        assert scenario.steps["step3"].depends_on == ["step1"]
        assert scenario.steps["step4"].depends_on == ["step1"]

    def test_topological_order_linear(self):
        """Test topological_order with linear flow."""
        scenario = Scenario(
            name="test",
            context=ScenarioContext(required_aspects=["coder"]),
            steps={
                "step1": StepDefinition(
                    id="step1",
                    prompt="First",
                    output=StepOutput(document_type="test"),
                    next=["step2"],
                    depends_on=[]
                ),
                "step2": StepDefinition(
                    id="step2",
                    prompt="Second",
                    output=StepOutput(document_type="test"),
                    next=["step3"],
                    depends_on=["step1"]
                ),
                "step3": StepDefinition(
                    id="step3",
                    prompt="Third",
                    output=StepOutput(document_type="test"),
                    next=[],
                    depends_on=["step2"]
                )
            }
        )

        order = scenario.topological_order()
        assert order == ["step1", "step2", "step3"]

    def test_topological_order_fan_out_fan_in(self):
        """Test topological_order with fan-out and fan-in pattern."""
        scenario = Scenario(
            name="test",
            context=ScenarioContext(required_aspects=["coder"]),
            steps={
                "step1": StepDefinition(
                    id="step1",
                    prompt="Root",
                    output=StepOutput(document_type="test"),
                    next=["step2", "step3"],
                    depends_on=[]
                ),
                "step2": StepDefinition(
                    id="step2",
                    prompt="Branch 1",
                    output=StepOutput(document_type="test"),
                    next=["step4"],
                    depends_on=["step1"]
                ),
                "step3": StepDefinition(
                    id="step3",
                    prompt="Branch 2",
                    output=StepOutput(document_type="test"),
                    next=["step4"],
                    depends_on=["step1"]
                ),
                "step4": StepDefinition(
                    id="step4",
                    prompt="Merge",
                    output=StepOutput(document_type="test"),
                    next=[],
                    depends_on=["step2", "step3"]
                )
            }
        )

        order = scenario.topological_order()
        assert order[0] == "step1"
        assert order[-1] == "step4"
        assert set(order[1:3]) == {"step2", "step3"}

    def test_topological_order_cycle_detection(self):
        """Test topological_order detects cycles."""
        scenario = Scenario(
            name="test",
            context=ScenarioContext(required_aspects=["coder"]),
            steps={
                "step1": StepDefinition(
                    id="step1",
                    prompt="First",
                    output=StepOutput(document_type="test"),
                    next=["step2"],
                    depends_on=["step3"]  # Creates cycle
                ),
                "step2": StepDefinition(
                    id="step2",
                    prompt="Second",
                    output=StepOutput(document_type="test"),
                    next=["step3"],
                    depends_on=["step1"]
                ),
                "step3": StepDefinition(
                    id="step3",
                    prompt="Third",
                    output=StepOutput(document_type="test"),
                    next=["step1"],
                    depends_on=["step2"]
                )
            }
        )

        with pytest.raises(ValueError) as exc_info:
            scenario.topological_order()
        assert "cycle" in str(exc_info.value).lower()


class TestStepResult:
    """Tests for StepResult model."""

    def test_step_result_creation(self):
        """Test StepResult with all fields."""
        now = datetime.now(timezone.utc)
        result = StepResult(
            step_id="step1",
            response="Test response",
            think="Internal thoughts",
            doc_id="doc123",
            document_type="analysis",
            document_weight=0.8,
            tokens_used=100,
            timestamp=now
        )
        assert result.step_id == "step1"
        assert result.response == "Test response"
        assert result.think == "Internal thoughts"
        assert result.doc_id == "doc123"
        assert result.document_type == "analysis"
        assert result.document_weight == 0.8
        assert result.tokens_used == 100
        assert result.timestamp == now

    def test_step_result_no_think(self):
        """Test StepResult without think field."""
        now = datetime.now(timezone.utc)
        result = StepResult(
            step_id="step1",
            response="Response",
            doc_id="doc123",
            document_type="test",
            document_weight=1.0,
            tokens_used=50,
            timestamp=now
        )
        assert result.think is None

    def test_step_result_serialization(self):
        """Test StepResult JSON serialization."""
        now = datetime.now(timezone.utc)
        result = StepResult(
            step_id="step1",
            response="Test",
            doc_id="doc123",
            document_type="test",
            document_weight=1.0,
            tokens_used=50,
            timestamp=now
        )
        # Should be able to dump to JSON
        json_data = result.model_dump(mode='json')
        assert json_data['step_id'] == "step1"
        assert 'timestamp' in json_data


class TestStepJob:
    """Tests for StepJob model."""

    def test_step_job_defaults(self):
        """Test StepJob with default values."""
        now = datetime.now(timezone.utc)
        job = StepJob(
            pipeline_id="pipe123",
            step_id="step1",
            enqueued_at=now
        )
        assert job.pipeline_id == "pipe123"
        assert job.step_id == "step1"
        assert job.attempt == 1
        assert job.max_attempts == 3
        assert job.enqueued_at == now
        assert job.priority == 0

    def test_step_job_custom_values(self):
        """Test StepJob with custom values."""
        now = datetime.now(timezone.utc)
        job = StepJob(
            pipeline_id="pipe123",
            step_id="step1",
            attempt=2,
            max_attempts=5,
            enqueued_at=now,
            priority=10
        )
        assert job.attempt == 2
        assert job.max_attempts == 5
        assert job.priority == 10

    def test_increment_attempt(self):
        """Test increment_attempt method."""
        now = datetime.now(timezone.utc)
        job = StepJob(
            pipeline_id="pipe123",
            step_id="step1",
            attempt=1,
            enqueued_at=now
        )

        new_job = job.increment_attempt()

        # Original job unchanged
        assert job.attempt == 1

        # New job has incremented attempt
        assert new_job.attempt == 2
        assert new_job.pipeline_id == "pipe123"
        assert new_job.step_id == "step1"
        assert new_job.enqueued_at == now

    def test_increment_attempt_multiple(self):
        """Test multiple increment_attempt calls."""
        now = datetime.now(timezone.utc)
        job = StepJob(
            pipeline_id="pipe123",
            step_id="step1",
            enqueued_at=now
        )

        job2 = job.increment_attempt()
        job3 = job2.increment_attempt()

        assert job.attempt == 1
        assert job2.attempt == 2
        assert job3.attempt == 3

    def test_step_job_serialization(self):
        """Test StepJob JSON serialization."""
        now = datetime.now(timezone.utc)
        job = StepJob(
            pipeline_id="pipe123",
            step_id="step1",
            enqueued_at=now
        )
        # Should be able to dump to JSON
        json_data = job.model_dump(mode='json')
        assert json_data['pipeline_id'] == "pipe123"
        assert 'enqueued_at' in json_data


class TestPipelineState:
    """Tests for PipelineState model."""

    def test_pipeline_state_minimal(self):
        """Test PipelineState with minimal required fields."""
        now = datetime.now(timezone.utc)
        state = PipelineState(
            pipeline_id="pipe123",
            scenario_name="test",
            conversation_id="conv123",
            persona_id="persona123",
            user_id="user123",
            model="gpt-4",
            branch=1,
            created_at=now,
            updated_at=now
        )
        assert state.pipeline_id == "pipe123"
        assert state.scenario_name == "test"
        assert state.conversation_id == "conv123"
        assert state.persona_id == "persona123"
        assert state.user_id == "user123"
        assert state.model == "gpt-4"
        assert state.branch == 1
        assert state.step_counter == 1
        assert state.extra == []
        assert state.completed_steps == []
        assert state.step_doc_ids == {}
        assert state.seed_doc_ids == {}

    def test_pipeline_state_complete(self):
        """Test PipelineState with all fields."""
        now = datetime.now(timezone.utc)

        state = PipelineState(
            pipeline_id="pipe123",
            scenario_name="analyst",
            conversation_id="conv123",
            persona_id="persona123",
            user_id="user123",
            model="gpt-4",
            thought_model="claude-3",
            codex_model="gpt-4-turbo",
            guidance="Focus on details",
            query_text="What happened?",
            persona_mood="thoughtful",
            branch=2,
            step_counter=5,
            extra=["Note 1", "Note 2"],
            completed_steps=["step1"],
            step_doc_ids={"step1": "doc123"},
            seed_doc_ids={"step1": ["seed_doc1", "seed_doc2"]},
            created_at=now,
            updated_at=now
        )

        assert state.thought_model == "claude-3"
        assert state.codex_model == "gpt-4-turbo"
        assert state.guidance == "Focus on details"
        assert state.query_text == "What happened?"
        assert state.persona_mood == "thoughtful"
        assert state.step_counter == 5
        assert len(state.extra) == 2
        assert state.completed_steps == ["step1"]
        assert "step1" in state.step_doc_ids
        assert state.step_doc_ids["step1"] == "doc123"
        assert "step1" in state.seed_doc_ids
        assert len(state.seed_doc_ids["step1"]) == 2

    def test_pipeline_state_serialization(self):
        """Test PipelineState JSON serialization."""
        now = datetime.now(timezone.utc)
        state = PipelineState(
            pipeline_id="pipe123",
            scenario_name="test",
            conversation_id="conv123",
            persona_id="persona123",
            user_id="user123",
            model="gpt-4",
            branch=1,
            created_at=now,
            updated_at=now
        )

        # Should be able to dump to JSON
        json_data = state.model_dump(mode='json')
        assert json_data['pipeline_id'] == "pipe123"
        assert 'created_at' in json_data
        assert 'updated_at' in json_data

    def test_pipeline_state_with_step_doc_ids(self):
        """Test PipelineState stores doc_id references for step outputs."""
        now = datetime.now(timezone.utc)

        state = PipelineState(
            pipeline_id="pipe123",
            scenario_name="test",
            conversation_id="conv123",
            persona_id="persona123",
            user_id="user123",
            model="gpt-4",
            branch=1,
            completed_steps=["step1", "step2"],
            step_doc_ids={"step1": "doc1", "step2": "doc2"},
            created_at=now,
            updated_at=now
        )

        assert len(state.step_doc_ids) == 2
        assert state.step_doc_ids["step1"] == "doc1"
        assert state.step_doc_ids["step2"] == "doc2"


class TestScenarioTool:
    """Tests for ScenarioTool model."""

    def test_scenario_tool_required_fields(self):
        """Test ScenarioTool with all required fields."""
        tool = ScenarioTool(
            name="select_topic",
            description="Select a topic to explore",
            parameters={
                "topic": {"type": "string", "description": "The topic"},
                "reasoning": {"type": "string", "description": "Why this topic"}
            }
        )
        assert tool.name == "select_topic"
        assert tool.description == "Select a topic to explore"
        assert "topic" in tool.parameters
        assert tool.required == []

    def test_scenario_tool_with_required_params(self):
        """Test ScenarioTool with required parameters."""
        tool = ScenarioTool(
            name="add_task",
            description="Add a task to the plan",
            parameters={
                "summary": {"type": "string"},
                "details": {"type": "string"}
            },
            required=["summary", "details"]
        )
        assert tool.required == ["summary", "details"]


class TestCondition:
    """Tests for Condition model."""

    def test_condition_with_conditional_fields(self):
        """Test Condition with source/condition/target/goto."""
        cond = Condition(
            source="tool_result.accept",
            condition="==",
            target="true",
            goto="next_step"
        )
        assert cond.source == "tool_result.accept"
        assert cond.condition == "=="
        assert cond.target == "true"
        assert cond.goto == "next_step"
        assert cond.default is None

    def test_condition_with_default(self):
        """Test Condition with default only."""
        cond = Condition(default="fallback_step")
        assert cond.default == "fallback_step"
        assert cond.source is None
        assert cond.condition is None

    def test_condition_with_collect_to(self):
        """Test Condition with collect_to."""
        cond = Condition(
            source="tool_name",
            condition="==",
            target="add_task",
            goto="verify_task",
            collect_to="tasks"
        )
        assert cond.collect_to == "tasks"

    def test_condition_in_operator(self):
        """Test Condition with 'in' operator."""
        cond = Condition(
            source="tool_result.status",
            condition="in",
            target=["pending", "active"],
            goto="process"
        )
        assert cond.condition == "in"
        assert cond.target == ["pending", "active"]

    def test_condition_not_in_operator(self):
        """Test Condition with 'not_in' operator."""
        cond = Condition(
            source="tool_result.status",
            condition="not_in",
            target=["failed", "aborted"],
            goto="continue"
        )
        assert cond.condition == "not_in"

    def test_condition_invalid_both_conditional_and_default(self):
        """Test Condition fails when both conditional and default specified."""
        with pytest.raises(ValidationError) as exc_info:
            Condition(
                source="tool_result.accept",
                condition="==",
                target="true",
                goto="next_step",
                default="fallback"  # Invalid: both conditional and default
            )
        assert "cannot have both" in str(exc_info.value).lower()

    def test_condition_invalid_neither_conditional_nor_default(self):
        """Test Condition fails when neither conditional nor default specified."""
        with pytest.raises(ValidationError) as exc_info:
            Condition(
                source="tool_result.accept",  # Missing condition and goto
            )
        assert "must have either" in str(exc_info.value).lower()

    def test_condition_invalid_operator(self):
        """Test Condition fails with invalid operator."""
        with pytest.raises(ValidationError) as exc_info:
            Condition(
                source="tool_result.accept",
                condition="contains",  # Invalid operator
                target="true",
                goto="next_step"
            )
        assert "invalid condition operator" in str(exc_info.value).lower()


class TestStepConfigExtensions:
    """Tests for StepConfig new fields."""

    def test_step_config_new_defaults(self):
        """Test new StepConfig fields have correct defaults."""
        config = StepConfig()
        assert config.max_iterations is None
        assert config.on_limit is None
        assert config.tool_retries == 3

    def test_step_config_iteration_settings(self):
        """Test StepConfig with iteration settings."""
        config = StepConfig(
            max_iterations=10,
            on_limit="finalize",
            tool_retries=5
        )
        assert config.max_iterations == 10
        assert config.on_limit == "finalize"
        assert config.tool_retries == 5


class TestStepResultExtensions:
    """Tests for StepResult new fields."""

    def test_step_result_with_tool_fields(self):
        """Test StepResult with tool_name and tool_result."""
        now = datetime.now(timezone.utc)
        result = StepResult(
            step_id="select_topic",
            response="",
            doc_id="doc123",
            document_type="step-output",
            document_weight=1.0,
            tokens_used=100,
            timestamp=now,
            tool_name="select_topic",
            tool_result={"topic": "AI consciousness", "reasoning": "Important topic"}
        )
        assert result.tool_name == "select_topic"
        assert result.tool_result == {"topic": "AI consciousness", "reasoning": "Important topic"}

    def test_step_result_tool_fields_default_none(self):
        """Test StepResult tool fields default to None."""
        now = datetime.now(timezone.utc)
        result = StepResult(
            step_id="step1",
            response="Test response",
            doc_id="doc123",
            document_type="test",
            document_weight=1.0,
            tokens_used=50,
            timestamp=now
        )
        assert result.tool_name is None
        assert result.tool_result is None


class TestNewStepDefinitions:
    """Tests for new step definition types."""

    def test_context_only_step_definition(self):
        """Test ContextOnlyStepDefinition."""
        step = ContextOnlyStepDefinition(
            id="gather_context",
            context=[
                MemoryAction(action="search_memories", query_text="recent", top_n=10)
            ],
            next=["select_topic"]
        )
        assert step.id == "gather_context"
        assert step.type == "context_only"
        assert len(step.context) == 1
        assert step.next == ["select_topic"]
        assert step.depends_on == []

    def test_standard_step_definition(self):
        """Test StandardStepDefinition."""
        step = StandardStepDefinition(
            id="generate_summary",
            prompt="Summarize the following context.",
            config=StepConfig(max_tokens=2048),
            output=StepOutput(document_type="summary"),
            next=["end"]
        )
        assert step.id == "generate_summary"
        assert step.type == "standard"
        assert step.prompt == "Summarize the following context."
        assert step.config.max_tokens == 2048
        assert step.output.document_type == "summary"
        assert step.next == ["end"]

    def test_tool_calling_step_definition(self):
        """Test ToolCallingStepDefinition."""
        step = ToolCallingStepDefinition(
            id="select_topic",
            prompt="Select a topic to explore.",
            tools=["select_topic"],
            next_conditions=[
                Condition(default="validate")
            ],
            config=StepConfig(model_role="tool")
        )
        assert step.id == "select_topic"
        assert step.type == "tool_calling"
        assert step.tools == ["select_topic"]
        assert len(step.next_conditions) == 1
        assert step.config.model_role == "tool"

    def test_rendering_step_definition(self):
        """Test RenderingStepDefinition."""
        step = RenderingStepDefinition(
            id="finalize",
            template="# Plan\n{% for task in collections.tasks %}{{ task.summary }}{% endfor %}",
            output=StepOutput(document_type="agent-plan"),
            next=["end"]
        )
        assert step.id == "finalize"
        assert step.type == "rendering"
        assert "{% for task" in step.template
        assert step.output.document_type == "agent-plan"
        assert step.next == ["end"]

    def test_base_step_definition_depends_on(self):
        """Test depends_on field on step definitions."""
        step = ContextOnlyStepDefinition(
            id="step2",
            depends_on=["step1"],
            next=["step3"]
        )
        assert step.depends_on == ["step1"]


class TestNewStepDefinitionDiscriminatedUnion:
    """Tests for NewStepDefinition discriminated union parsing."""

    def test_parse_context_only_from_dict(self):
        """Test parsing ContextOnlyStepDefinition from dict."""
        from pydantic import TypeAdapter

        data = {
            "id": "gather",
            "type": "context_only",
            "context": [{"action": "search_memories", "top_n": 10}],
            "next": ["process"]
        }
        adapter = TypeAdapter(NewStepDefinition)
        step = adapter.validate_python(data)
        assert step.type == "context_only"
        assert step.id == "gather"

    def test_parse_standard_from_dict(self):
        """Test parsing StandardStepDefinition from dict."""
        from pydantic import TypeAdapter

        data = {
            "id": "generate",
            "type": "standard",
            "prompt": "Generate content",
            "output": {"document_type": "output"},
            "next": ["end"]
        }
        adapter = TypeAdapter(NewStepDefinition)
        step = adapter.validate_python(data)
        assert step.type == "standard"
        assert step.prompt == "Generate content"

    def test_parse_tool_calling_from_dict(self):
        """Test parsing ToolCallingStepDefinition from dict."""
        from pydantic import TypeAdapter

        data = {
            "id": "decide",
            "type": "tool_calling",
            "prompt": "Make a decision",
            "tools": ["decide_tool"],
            "next_conditions": [{"default": "next"}]
        }
        adapter = TypeAdapter(NewStepDefinition)
        step = adapter.validate_python(data)
        assert step.type == "tool_calling"
        assert step.tools == ["decide_tool"]

    def test_parse_rendering_from_dict(self):
        """Test parsing RenderingStepDefinition from dict."""
        from pydantic import TypeAdapter

        data = {
            "id": "render",
            "type": "rendering",
            "template": "# Output\n{{ content }}",
            "output": {"document_type": "final"},
            "next": ["end"]
        }
        adapter = TypeAdapter(NewStepDefinition)
        step = adapter.validate_python(data)
        assert step.type == "rendering"
        assert "{{ content }}" in step.template
