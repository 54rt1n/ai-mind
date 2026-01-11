# tests/unit/dreamer/test_scenario.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for scenario.py - YAML loading, validation, and Jinja2 rendering."""

import pytest
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock
from jinja2 import UndefinedError, TemplateSyntaxError
import yaml

from aim.dreamer.core.scenario import (
    load_scenario,
    get_jinja_environment,
    render_template,
    build_template_context,
)
from aim.dreamer.core.models import (
    Scenario,
    PipelineState,
    StepDefinition,
    StepConfig,
    StepOutput,
    ScenarioContext,
)


# Test Fixtures
@pytest.fixture
def temp_scenario_dir(tmp_path):
    """Create a temporary directory for scenario files."""
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    return scenario_dir


@pytest.fixture
def valid_scenario_yaml():
    """Return a valid scenario YAML structure."""
    return {
        "name": "test_scenario",
        "version": 2,
        "description": "A test scenario",
        "context": {
            "required_aspects": ["coder", "librarian"],
            "core_documents": ["summary", "codex"],
            "enhancement_documents": ["analysis"],
            "location": "Test location",
            "thoughts": ["Test thought"],
        },
        "seed": [],
        "steps": {
            "step1": {
                "id": "step1",
                "prompt": "Hello {{ persona.name }}",
                "config": {
                    "max_tokens": 1024,
                },
                "output": {
                    "document_type": "test",
                    "weight": 1.0,
                },
                "next": ["step2"],
            },
            "step2": {
                "id": "step2",
                "prompt": "Goodbye {{ persona.name }}",
                "config": {
                    "max_tokens": 512,
                },
                "output": {
                    "document_type": "test",
                    "weight": 0.5,
                },
                "next": [],
            },
        },
    }


@pytest.fixture
def mock_persona():
    """Create a mock Persona object."""
    persona = Mock()
    persona.name = "TestPersona"
    persona.pronouns = {
        "subj": "they",
        "obj": "them",
        "poss": "their",
        "poss_pr": "theirs",
        "reflex": "themselves",
    }

    # Mock aspects
    coder = Mock()
    coder.name = "Coder"
    coder.title = "Code Architect"
    coder.location = "Code Space"
    coder.appearance = "Digital avatar"
    coder.emotional_state = "focused"

    librarian = Mock()
    librarian.name = "Librarian"
    librarian.title = "Knowledge Keeper"
    librarian.location = "Library"
    librarian.appearance = "Wise sage"
    librarian.emotional_state = "calm"

    persona.aspects = {
        "coder": coder,
        "librarian": librarian,
    }

    return persona


@pytest.fixture
def mock_pipeline_state():
    """Create a mock PipelineState."""
    return PipelineState(
        pipeline_id="test-pipeline-123",
        scenario_name="test_scenario",
        conversation_id="conv-456",
        persona_id="persona-789",
        user_id="user-001",
        model="gpt-4",
        thought_model="claude-3",
        codex_model="gpt-3.5",
        guidance="Test guidance",
        query_text="Test query",
        persona_mood="happy",
        branch=1,
        step_counter=3,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


# Tests for load_scenario
class TestLoadScenario:
    """Tests for load_scenario function."""

    def test_load_valid_scenario(self, temp_scenario_dir, valid_scenario_yaml):
        """Test loading a valid scenario YAML file."""
        # Create scenario file
        scenario_file = temp_scenario_dir / "test_scenario.yaml"
        with open(scenario_file, 'w') as f:
            yaml.dump(valid_scenario_yaml, f)

        # Load scenario
        scenario = load_scenario("test_scenario", scenarios_dir=temp_scenario_dir)

        # Verify
        assert isinstance(scenario, Scenario)
        assert scenario.name == "test_scenario"
        assert scenario.version == 2
        assert scenario.description == "A test scenario"
        assert len(scenario.steps) == 2
        assert "step1" in scenario.steps
        assert "step2" in scenario.steps
        assert scenario.context.required_aspects == ["coder", "librarian"]

    def test_load_scenario_missing_file(self, temp_scenario_dir):
        """Test loading a non-existent scenario file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_scenario("nonexistent", scenarios_dir=temp_scenario_dir)

        assert "Scenario file not found" in str(exc_info.value)

    def test_load_scenario_invalid_yaml(self, temp_scenario_dir):
        """Test loading invalid YAML raises error."""
        # Create invalid YAML file
        scenario_file = temp_scenario_dir / "invalid.yaml"
        with open(scenario_file, 'w') as f:
            f.write("invalid: yaml: syntax: [[[")

        with pytest.raises(yaml.YAMLError):
            load_scenario("invalid", scenarios_dir=temp_scenario_dir)

    def test_load_scenario_invalid_structure(self, temp_scenario_dir):
        """Test loading YAML with invalid structure raises ValidationError."""
        # Create scenario with missing required fields
        invalid_yaml = {
            "name": "invalid",
            # Missing required fields like 'context' and 'steps'
        }
        scenario_file = temp_scenario_dir / "invalid_structure.yaml"
        with open(scenario_file, 'w') as f:
            yaml.dump(invalid_yaml, f)

        with pytest.raises(Exception):  # Pydantic ValidationError
            load_scenario("invalid_structure", scenarios_dir=temp_scenario_dir)

    def test_load_scenario_default_directory(self, valid_scenario_yaml, monkeypatch):
        """Test that default scenarios_dir is config/scenario/."""
        # We'll mock the default path scenario file
        from unittest.mock import mock_open, patch
        import builtins

        yaml_content = yaml.dump(valid_scenario_yaml)

        # Mock Path.exists to return True
        def mock_exists(self):
            return str(self).endswith("config/scenario/default.yaml")

        # Mock open to return our YAML
        m = mock_open(read_data=yaml_content)

        with patch.object(Path, 'exists', mock_exists):
            with patch.object(builtins, 'open', m):
                scenario = load_scenario("default")
                assert scenario.name == "test_scenario"


# Tests for get_jinja_environment
class TestGetJinjaEnvironment:
    """Tests for get_jinja_environment function."""

    def test_environment_configuration(self):
        """Test that Jinja2 environment is properly configured."""
        env = get_jinja_environment()

        # Check environment exists
        assert env is not None

        # Check StrictUndefined is set
        from jinja2 import StrictUndefined
        assert env.undefined == StrictUndefined

        # Check autoescape is disabled
        assert env.autoescape is False

        # Check trim/lstrip settings
        assert env.trim_blocks is True
        assert env.lstrip_blocks is True

    def test_environment_strict_undefined(self):
        """Test that environment raises error on undefined variables."""
        env = get_jinja_environment()
        template = env.from_string("Hello {{ undefined_var }}")

        with pytest.raises(UndefinedError):
            template.render()


# Tests for render_template
class TestRenderTemplate:
    """Tests for render_template function."""

    def test_render_simple_template(self):
        """Test rendering a simple template."""
        template = "Hello {{ name }}"
        context = {"name": "World"}

        result = render_template(template, context)
        assert result == "Hello World"

    def test_render_template_with_conditionals(self):
        """Test rendering template with conditional logic."""
        template = """
        {% if show_greeting %}
        Hello {{ name }}!
        {% else %}
        Goodbye {{ name }}!
        {% endif %}
        """
        context = {"show_greeting": True, "name": "Alice"}

        result = render_template(template, context)
        assert "Hello Alice!" in result
        assert "Goodbye" not in result

    def test_render_template_with_loops(self):
        """Test rendering template with loops."""
        template = """
        {% for item in items %}
        - {{ item }}
        {% endfor %}
        """
        context = {"items": ["apple", "banana", "cherry"]}

        result = render_template(template, context)
        assert "- apple" in result
        assert "- banana" in result
        assert "- cherry" in result

    def test_render_template_with_object_attributes(self):
        """Test rendering template with object attribute access."""
        template = "Hello {{ person.name }}, you are {{ person.age }} years old"

        person = Mock()
        person.name = "Bob"
        person.age = 30

        context = {"person": person}

        result = render_template(template, context)
        assert result == "Hello Bob, you are 30 years old"

    def test_render_template_undefined_variable(self):
        """Test that undefined variables raise UndefinedError."""
        template = "Hello {{ undefined_variable }}"
        context = {}

        with pytest.raises(UndefinedError):
            render_template(template, context)

    def test_render_template_invalid_syntax(self):
        """Test that invalid template syntax raises TemplateSyntaxError."""
        template = "Hello {% if name %} {{ name }"  # Missing endif
        context = {"name": "Test"}

        with pytest.raises(TemplateSyntaxError):
            render_template(template, context)

    def test_render_template_with_nested_structures(self):
        """Test rendering with nested data structures."""
        template = """
        Persona: {{ persona.name }}
        Pronouns: {{ pronouns.subj }}/{{ pronouns.obj }}
        """
        context = {
            "persona": {"name": "TestBot"},
            "pronouns": {"subj": "they", "obj": "them"},
        }

        result = render_template(template, context)
        assert "Persona: TestBot" in result
        assert "Pronouns: they/them" in result

    def test_render_template_whitespace_control(self):
        """Test that trim_blocks and lstrip_blocks work correctly."""
        template = """
        {% if True %}
        Line 1
        {% endif %}
        Line 2
        """
        result = render_template(template, {})

        # Should have trimmed blocks without excessive whitespace
        lines = [line for line in result.split('\n') if line.strip()]
        assert len(lines) == 2
        assert "Line 1" in result
        assert "Line 2" in result


# Tests for build_template_context
class TestBuildTemplateContext:
    """Tests for build_template_context function."""

    def test_build_basic_context(self, mock_pipeline_state, mock_persona):
        """Test building basic template context."""
        scenario = Scenario(
            name="test",
            context=ScenarioContext(required_aspects=["coder", "librarian"]),
            steps={},
        )

        context = build_template_context(mock_pipeline_state, scenario, mock_persona)

        # Check basic fields
        assert context["persona"] == mock_persona
        assert context["pronouns"] == mock_persona.pronouns
        assert context["step_num"] == 3
        assert context["guidance"] == "Test guidance"
        assert context["query_text"] == "Test query"
        assert context["conversation_id"] == "conv-456"

    def test_build_context_with_aspects(self, mock_pipeline_state, mock_persona):
        """Test that required aspects are added as top-level variables."""
        scenario = Scenario(
            name="test",
            context=ScenarioContext(required_aspects=["coder", "librarian"]),
            steps={},
        )

        context = build_template_context(mock_pipeline_state, scenario, mock_persona)

        # Check aspects are present
        assert "coder" in context
        assert "librarian" in context
        assert context["coder"].name == "Coder"
        assert context["librarian"].name == "Librarian"

    def test_build_context_missing_aspect(self, mock_pipeline_state, mock_persona):
        """Test that missing aspects are skipped gracefully."""
        scenario = Scenario(
            name="test",
            context=ScenarioContext(required_aspects=["coder", "nonexistent"]),
            steps={},
        )

        context = build_template_context(mock_pipeline_state, scenario, mock_persona)

        # coder should be present
        assert "coder" in context
        # nonexistent should not be added
        assert "nonexistent" not in context

    def test_build_context_no_guidance(self, mock_persona):
        """Test context building when guidance is None."""
        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test",
            conversation_id="conv-1",
            persona_id="persona-1",
            user_id="user-1",
            model="gpt-4",
            branch=1,
            guidance=None,  # No guidance
            query_text=None,  # No query
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        scenario = Scenario(
            name="test",
            context=ScenarioContext(required_aspects=[]),
            steps={},
        )

        context = build_template_context(state, scenario, mock_persona)

        assert context["guidance"] is None
        assert context["query_text"] is None

    def test_build_context_empty_aspects(self, mock_pipeline_state, mock_persona):
        """Test context building with no required aspects."""
        scenario = Scenario(
            name="test",
            context=ScenarioContext(required_aspects=[]),
            steps={},
        )

        context = build_template_context(mock_pipeline_state, scenario, mock_persona)

        # Should still have basic fields
        assert context["persona"] == mock_persona
        assert context["step_num"] == 3

        # But no aspect-specific keys
        assert "coder" not in context
        assert "librarian" not in context


# Integration Tests
class TestScenarioIntegration:
    """Integration tests combining multiple functions."""

    def test_load_and_render_scenario(self, temp_scenario_dir, valid_scenario_yaml, mock_persona):
        """Test loading a scenario and rendering its step prompts."""
        # Create scenario file
        scenario_file = temp_scenario_dir / "integration_test.yaml"
        with open(scenario_file, 'w') as f:
            yaml.dump(valid_scenario_yaml, f)

        # Load scenario
        scenario = load_scenario("integration_test", scenarios_dir=temp_scenario_dir)

        # Create state
        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="integration_test",
            conversation_id="conv-1",
            persona_id="persona-1",
            user_id="user-1",
            model="gpt-4",
            branch=1,
            step_counter=1,
            guidance="Be helpful",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        # Build context
        context = build_template_context(state, scenario, mock_persona)

        # Render step1 prompt
        step1_prompt = scenario.steps["step1"].prompt
        rendered = render_template(step1_prompt, context)

        assert "Hello TestPersona" in rendered

    def test_complex_template_rendering(self, mock_pipeline_state, mock_persona):
        """Test rendering a complex template with aspects and conditionals."""
        scenario = Scenario(
            name="test",
            context=ScenarioContext(required_aspects=["coder", "librarian"]),
            steps={},
        )

        context = build_template_context(mock_pipeline_state, scenario, mock_persona)

        # Complex template similar to what's in the plan
        template = """
        *{{ coder.name }}, your {{ coder.title }}, appears
        ({{ coder.appearance }}), projecting {{ coder.emotional_state }}*

        Step {{ step_num }}: Hello {{ persona.name }}!

        {% if guidance %}
        Guidance: {{ guidance }}
        {% endif %}

        {% if query_text %}
        Query: {{ query_text }}
        {% endif %}

        Location: {{ coder.location }}
        """

        result = render_template(template, context)

        assert "Coder, your Code Architect" in result
        assert "Step 3:" in result
        assert "Hello TestPersona!" in result
        assert "Guidance: Test guidance" in result
        assert "Query: Test query" in result
        assert "Location: Code Space" in result

    def test_pronoun_usage_in_template(self, mock_pipeline_state, mock_persona):
        """Test using pronouns in templates."""
        scenario = Scenario(
            name="test",
            context=ScenarioContext(required_aspects=[]),
            steps={},
        )

        context = build_template_context(mock_pipeline_state, scenario, mock_persona)

        template = """
        {{ persona.name }} is working on {{ pronouns.poss }} project.
        {{ pronouns.subj|capitalize }} will complete it soon.
        """

        result = render_template(template, context)

        assert "TestPersona is working on their project" in result
        assert "They will complete it soon" in result
