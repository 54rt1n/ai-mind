# tests/unit/dreamer/test_framework.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for ScenarioFramework and ScenarioBuilder."""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
import yaml

from aim.dreamer.core.framework import ScenarioFramework
from aim.dreamer.core.builder import ScenarioBuilder, load_scenario_framework
from aim.dreamer.core.models import (
    ScenarioTool,
    Condition,
    ContextOnlyStepDefinition,
    StandardStepDefinition,
    ToolCallingStepDefinition,
    RenderingStepDefinition,
    StepConfig,
    StepOutput,
    MemoryAction,
)


class TestScenarioFramework:
    """Tests for ScenarioFramework model."""

    def test_framework_minimal(self):
        """Test ScenarioFramework with minimal required fields."""
        framework = ScenarioFramework(
            name="test",
            first_step="step1",
            steps={
                "step1": ContextOnlyStepDefinition(
                    id="step1",
                    next=["end"]
                )
            }
        )
        assert framework.name == "test"
        assert framework.version == 1
        assert framework.first_step == "step1"
        assert len(framework.steps) == 1
        assert len(framework.tools) == 0

    def test_framework_with_tools(self):
        """Test ScenarioFramework with tool definitions."""
        framework = ScenarioFramework(
            name="test",
            first_step="decide",
            steps={
                "decide": ToolCallingStepDefinition(
                    id="decide",
                    prompt="Make a choice",
                    tools=["select"],
                    next_conditions=[Condition(default="end")]
                )
            },
            tools={
                "select": ScenarioTool(
                    name="select",
                    description="Select an option",
                    parameters={"option": {"type": "string"}},
                    required=["option"]
                )
            }
        )
        assert len(framework.tools) == 1
        assert "select" in framework.tools
        assert framework.tools["select"].required == ["option"]

    def test_get_tools_success(self):
        """Test get_tools returns Tool objects."""
        framework = ScenarioFramework(
            name="test",
            first_step="decide",
            steps={
                "decide": ToolCallingStepDefinition(
                    id="decide",
                    prompt="Decide",
                    tools=["my_tool"],
                    next_conditions=[Condition(default="end")]
                )
            },
            tools={
                "my_tool": ScenarioTool(
                    name="my_tool",
                    description="A test tool",
                    parameters={"arg": {"type": "string"}},
                    required=["arg"]
                )
            }
        )

        tools = framework.get_tools(["my_tool"])
        assert len(tools) == 1
        assert tools[0].function.name == "my_tool"
        assert tools[0].function.description == "A test tool"

    def test_get_tools_not_found(self):
        """Test get_tools raises for undefined tool."""
        framework = ScenarioFramework(
            name="test",
            first_step="step1",
            steps={
                "step1": ContextOnlyStepDefinition(id="step1", next=["end"])
            }
        )

        with pytest.raises(ValueError) as exc_info:
            framework.get_tools(["missing_tool"])
        assert "not defined" in str(exc_info.value)

    def test_get_root_steps(self):
        """Test get_root_steps returns steps with no dependencies."""
        framework = ScenarioFramework(
            name="test",
            first_step="step1",
            steps={
                "step1": ContextOnlyStepDefinition(
                    id="step1",
                    next=["step2"]
                ),
                "step2": ContextOnlyStepDefinition(
                    id="step2",
                    depends_on=["step1"],
                    next=["end"]
                )
            }
        )

        roots = framework.get_root_steps()
        assert roots == ["step1"]

    def test_topological_order(self):
        """Test topological_order returns correct order."""
        framework = ScenarioFramework(
            name="test",
            first_step="a",
            steps={
                "a": ContextOnlyStepDefinition(id="a", next=["b"]),
                "b": ContextOnlyStepDefinition(id="b", depends_on=["a"], next=["c"]),
                "c": ContextOnlyStepDefinition(id="c", depends_on=["b"], next=["end"]),
            }
        )

        order = framework.topological_order()
        assert order == ["a", "b", "c"]

    def test_topological_order_cycle_detection(self):
        """Test topological_order raises on cycle."""
        framework = ScenarioFramework(
            name="test",
            first_step="a",
            steps={
                "a": ContextOnlyStepDefinition(id="a", depends_on=["c"], next=["b"]),
                "b": ContextOnlyStepDefinition(id="b", depends_on=["a"], next=["c"]),
                "c": ContextOnlyStepDefinition(id="c", depends_on=["b"], next=["a"]),
            }
        )

        with pytest.raises(ValueError) as exc_info:
            framework.topological_order()
        assert "cycle" in str(exc_info.value).lower()

    def test_validate_goto_targets_valid(self):
        """Test validate_goto_targets passes for valid targets."""
        framework = ScenarioFramework(
            name="test",
            first_step="step1",
            steps={
                "step1": ContextOnlyStepDefinition(id="step1", next=["step2"]),
                "step2": ContextOnlyStepDefinition(id="step2", next=["end"]),
            }
        )
        # Should not raise
        framework.validate_goto_targets()

    def test_validate_goto_targets_invalid_next(self):
        """Test validate_goto_targets fails for invalid next target."""
        framework = ScenarioFramework(
            name="test",
            first_step="step1",
            steps={
                "step1": ContextOnlyStepDefinition(id="step1", next=["nonexistent"]),
            }
        )

        with pytest.raises(ValueError) as exc_info:
            framework.validate_goto_targets()
        assert "invalid next target" in str(exc_info.value).lower()

    def test_validate_goto_targets_invalid_condition(self):
        """Test validate_goto_targets fails for invalid condition target."""
        framework = ScenarioFramework(
            name="test",
            first_step="step1",
            steps={
                "step1": ToolCallingStepDefinition(
                    id="step1",
                    prompt="Test",
                    tools=["t"],
                    next_conditions=[Condition(default="nonexistent")]
                )
            },
            tools={
                "t": ScenarioTool(name="t", description="Test", parameters={})
            }
        )

        with pytest.raises(ValueError) as exc_info:
            framework.validate_goto_targets()
        assert "invalid condition target" in str(exc_info.value).lower()

    def test_validate_tools_success(self):
        """Test validate_tools passes when all tools defined."""
        framework = ScenarioFramework(
            name="test",
            first_step="step1",
            steps={
                "step1": ToolCallingStepDefinition(
                    id="step1",
                    prompt="Test",
                    tools=["defined_tool"],
                    next_conditions=[Condition(default="end")]
                )
            },
            tools={
                "defined_tool": ScenarioTool(
                    name="defined_tool",
                    description="A defined tool",
                    parameters={}
                )
            }
        )
        # Should not raise
        framework.validate_tools()

    def test_validate_tools_undefined(self):
        """Test validate_tools fails for undefined tool reference."""
        framework = ScenarioFramework(
            name="test",
            first_step="step1",
            steps={
                "step1": ToolCallingStepDefinition(
                    id="step1",
                    prompt="Test",
                    tools=["undefined_tool"],
                    next_conditions=[Condition(default="end")]
                )
            }
        )

        with pytest.raises(ValueError) as exc_info:
            framework.validate_tools()
        assert "undefined tool" in str(exc_info.value).lower()


class TestScenarioBuilder:
    """Tests for ScenarioBuilder."""

    def test_builder_from_dict_minimal(self):
        """Test builder parses minimal scenario dict."""
        builder = ScenarioBuilder()

        data = {
            "name": "test_scenario",
            "first_step": "step1",
            "steps": {
                "step1": {
                    "type": "context_only",
                    "next": ["end"]
                }
            }
        }

        framework = builder.from_dict(data)
        assert framework.name == "test_scenario"
        assert framework.first_step == "step1"
        assert len(framework.steps) == 1
        assert framework.steps["step1"].type == "context_only"

    def test_builder_from_dict_with_tools(self):
        """Test builder parses tools section."""
        builder = ScenarioBuilder()

        data = {
            "name": "test",
            "first_step": "decide",
            "tools": {
                "select_topic": {
                    "description": "Select a topic",
                    "parameters": {
                        "topic": {"type": "string"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["topic"]
                }
            },
            "steps": {
                "decide": {
                    "type": "tool_calling",
                    "prompt": "Choose a topic",
                    "tools": ["select_topic"],
                    "next_conditions": [{"default": "end"}]
                }
            }
        }

        framework = builder.from_dict(data)
        assert "select_topic" in framework.tools
        assert framework.tools["select_topic"].required == ["topic"]

    def test_builder_injects_step_id(self):
        """Test builder injects step ID from dict key."""
        builder = ScenarioBuilder()

        data = {
            "name": "test",
            "first_step": "my_step",
            "steps": {
                "my_step": {
                    "type": "context_only",
                    "next": ["end"]
                }
            }
        }

        framework = builder.from_dict(data)
        assert framework.steps["my_step"].id == "my_step"

    def test_builder_infers_context_only_type(self):
        """Test builder infers context_only type."""
        builder = ScenarioBuilder()

        data = {
            "name": "test",
            "first_step": "gather",
            "steps": {
                "gather": {
                    # No type, no prompt, no output, has context
                    "context": [{"action": "search_memories", "top_n": 10}],
                    "next": ["process"]
                },
                "process": {
                    "type": "context_only",
                    "next": ["end"]
                }
            }
        }

        framework = builder.from_dict(data)
        assert framework.steps["gather"].type == "context_only"

    def test_builder_infers_standard_type(self):
        """Test builder infers standard type."""
        builder = ScenarioBuilder()

        data = {
            "name": "test",
            "first_step": "generate",
            "steps": {
                "generate": {
                    # No type, has prompt and output
                    "prompt": "Generate content",
                    "output": {"document_type": "content"},
                    "next": ["end"]
                }
            }
        }

        framework = builder.from_dict(data)
        assert framework.steps["generate"].type == "standard"

    def test_builder_infers_tool_calling_type(self):
        """Test builder infers tool_calling type."""
        builder = ScenarioBuilder()

        data = {
            "name": "test",
            "first_step": "decide",
            "tools": {
                "choose": {"description": "Make choice", "parameters": {}}
            },
            "steps": {
                "decide": {
                    # No type, has tools and next_conditions
                    "prompt": "Decide",
                    "tools": ["choose"],
                    "next_conditions": [{"default": "end"}]
                }
            }
        }

        framework = builder.from_dict(data)
        assert framework.steps["decide"].type == "tool_calling"

    def test_builder_infers_rendering_type(self):
        """Test builder infers rendering type."""
        builder = ScenarioBuilder()

        data = {
            "name": "test",
            "first_step": "render",
            "steps": {
                "render": {
                    # No type, has template
                    "template": "# Output\n{{ content }}",
                    "output": {"document_type": "final"},
                    "next": ["end"]
                }
            }
        }

        framework = builder.from_dict(data)
        assert framework.steps["render"].type == "rendering"

    def test_builder_parses_context_actions(self):
        """Test builder parses context MemoryAction list."""
        builder = ScenarioBuilder()

        data = {
            "name": "test",
            "first_step": "gather",
            "steps": {
                "gather": {
                    "type": "context_only",
                    "context": [
                        {"action": "search_memories", "query_text": "recent", "top_n": 10},
                        {"action": "sort", "by": "timestamp", "direction": "descending"}
                    ],
                    "next": ["end"]
                }
            }
        }

        framework = builder.from_dict(data)
        step = framework.steps["gather"]
        assert len(step.context) == 2
        assert step.context[0].action == "search_memories"
        assert step.context[0].top_n == 10
        assert step.context[1].action == "sort"

    def test_builder_parses_seed_actions(self):
        """Test builder parses scenario-level seed MemoryAction list."""
        builder = ScenarioBuilder()

        data = {
            "name": "test",
            "first_step": "step1",
            "seed": [
                {"action": "load_conversation", "target": "current"},
                {"action": "get_memory", "document_types": ["motd"], "top_n": 5},
            ],
            "steps": {
                "step1": {
                    "type": "context_only",
                    "next": ["end"]
                }
            }
        }

        framework = builder.from_dict(data)
        assert len(framework.seed) == 2
        assert framework.seed[0].action == "load_conversation"
        assert framework.seed[1].action == "get_memory"
        assert framework.seed[1].document_types == ["motd"]

    def test_builder_parses_step_config(self):
        """Test builder parses step config."""
        builder = ScenarioBuilder()

        data = {
            "name": "test",
            "first_step": "step1",
            "steps": {
                "step1": {
                    "type": "standard",
                    "prompt": "Generate",
                    "config": {
                        "max_tokens": 2048,
                        "max_iterations": 5,
                        "on_limit": "finish",
                        "tool_retries": 2
                    },
                    "output": {"document_type": "output"},
                    "next": ["end"]
                }
            }
        }

        framework = builder.from_dict(data)
        config = framework.steps["step1"].config
        assert config.max_tokens == 2048
        assert config.max_iterations == 5
        assert config.on_limit == "finish"
        assert config.tool_retries == 2

    def test_builder_parses_next_conditions(self):
        """Test builder parses next_conditions."""
        builder = ScenarioBuilder()

        data = {
            "name": "test",
            "first_step": "decide",
            "tools": {
                "choose": {"description": "Choose", "parameters": {}}
            },
            "steps": {
                "decide": {
                    "type": "tool_calling",
                    "prompt": "Decide",
                    "tools": ["choose"],
                    "next_conditions": [
                        {
                            "source": "tool_result.accept",
                            "condition": "==",
                            "target": "true",
                            "goto": "accepted"
                        },
                        {"default": "rejected"}
                    ]
                },
                "accepted": {"type": "context_only", "next": ["end"]},
                "rejected": {"type": "context_only", "next": ["end"]}
            }
        }

        framework = builder.from_dict(data)
        conditions = framework.steps["decide"].next_conditions
        assert len(conditions) == 2
        assert conditions[0].source == "tool_result.accept"
        assert conditions[0].condition == "=="
        assert conditions[0].goto == "accepted"
        assert conditions[1].default == "rejected"

    def test_builder_validation_fails_for_invalid_target(self):
        """Test builder validates goto targets at build time."""
        builder = ScenarioBuilder()

        data = {
            "name": "test",
            "first_step": "step1",
            "steps": {
                "step1": {
                    "type": "context_only",
                    "next": ["nonexistent_step"]
                }
            }
        }

        with pytest.raises(ValueError) as exc_info:
            builder.from_dict(data)
        assert "invalid" in str(exc_info.value).lower()

    def test_builder_defaults_first_step(self):
        """Test builder defaults first_step to first step in dict."""
        builder = ScenarioBuilder()

        data = {
            "name": "test",
            # No first_step
            "steps": {
                "the_step": {
                    "type": "context_only",
                    "next": ["end"]
                }
            }
        }

        framework = builder.from_dict(data)
        assert framework.first_step == "the_step"


class TestLoadScenarioFramework:
    """Tests for load_scenario_framework convenience function."""

    def test_load_from_file(self):
        """Test loading scenario from YAML file."""
        with TemporaryDirectory() as tmpdir:
            scenarios_dir = Path(tmpdir)

            # Create test YAML
            yaml_content = {
                "name": "test_file_load",
                "version": 1,
                "first_step": "step1",
                "steps": {
                    "step1": {
                        "type": "context_only",
                        "next": ["end"]
                    }
                }
            }

            yaml_path = scenarios_dir / "test_file_load.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_content, f)

            # Load
            framework = load_scenario_framework("test_file_load", scenarios_dir)

            assert framework.name == "test_file_load"
            assert framework.version == 1

    def test_load_file_not_found(self):
        """Test loading nonexistent file raises FileNotFoundError."""
        with TemporaryDirectory() as tmpdir:
            scenarios_dir = Path(tmpdir)

            with pytest.raises(FileNotFoundError):
                load_scenario_framework("nonexistent", scenarios_dir)


class TestComplexScenario:
    """Test building complex multi-step scenario."""

    def test_exploration_pattern(self):
        """Test exploration scenario pattern from DREAMER_EXTENSION_PLAN."""
        builder = ScenarioBuilder()

        data = {
            "name": "exploration_brainstorm",
            "version": 1,
            "description": "Brainstorm exploration - select topic and validate",
            "first_step": "gather_context",
            "tools": {
                "select_topic": {
                    "description": "Select exploration topic",
                    "parameters": {
                        "topic": {"type": "string"},
                        "approach": {"type": "string"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["topic", "approach", "reasoning"]
                },
                "validate_exploration": {
                    "description": "Validate the exploration decision",
                    "parameters": {
                        "accept": {"type": "boolean"},
                        "reasoning": {"type": "string"},
                        "guidance": {"type": "string"}
                    },
                    "required": ["accept", "reasoning"]
                }
            },
            "steps": {
                "gather_context": {
                    "type": "context_only",
                    "context": [
                        {"action": "search_memories", "query_text": "recent thoughts", "top_n": 30}
                    ],
                    "next": ["select_topic"]
                },
                "select_topic": {
                    "type": "tool_calling",
                    "prompt": "Select a topic to explore.",
                    "tools": ["select_topic"],
                    "next_conditions": [{"default": "validate"}]
                },
                "validate": {
                    "type": "tool_calling",
                    "prompt": "Validate the selection: {{ steps.select_topic.tool_result.topic }}",
                    "tools": ["validate_exploration"],
                    "next_conditions": [
                        {
                            "source": "tool_result.accept",
                            "condition": "==",
                            "target": "true",
                            "goto": "end"
                        },
                        {"default": "end"}
                    ]
                }
            }
        }

        framework = builder.from_dict(data)

        # Verify structure
        assert framework.name == "exploration_brainstorm"
        assert len(framework.steps) == 3
        assert len(framework.tools) == 2

        # Verify step types
        assert framework.steps["gather_context"].type == "context_only"
        assert framework.steps["select_topic"].type == "tool_calling"
        assert framework.steps["validate"].type == "tool_calling"

        # Verify tool references work
        tools = framework.get_tools(["select_topic"])
        assert len(tools) == 1
        assert tools[0].function.name == "select_topic"
