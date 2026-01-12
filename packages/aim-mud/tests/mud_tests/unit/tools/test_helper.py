# packages/aim-mud/tests/mud_tests/unit/tools/test_helper.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for ToolHelper."""

import pytest
from datetime import datetime, timezone

from andimud_worker.tools.helper import ToolHelper
from aim.tool.formatting import ToolUser
from aim.tool.dto import Tool, ToolFunction, ToolFunctionParameters
from aim_mud_types.plan import AgentPlan, PlanTask, PlanStatus, TaskStatus


@pytest.fixture
def sample_tool():
    """Create a sample tool for testing."""
    return Tool(
        type="test",
        function=ToolFunction(
            name="test_function",
            description="A test function",
            parameters=ToolFunctionParameters(
                type="object",
                properties={"arg1": {"type": "string"}},
                required=["arg1"],
            ),
        ),
    )


@pytest.fixture
def tool_user(sample_tool):
    """Create a ToolUser with a sample tool."""
    return ToolUser([sample_tool])


@pytest.fixture
def helper(tool_user):
    """Create a ToolHelper for testing."""
    return ToolHelper(tool_user)


@pytest.fixture
def sample_plan():
    """Create a sample plan for testing."""
    return AgentPlan(
        plan_id="test-plan-123",
        agent_id="andi",
        objective="Test objective",
        summary="Test plan",
        status=PlanStatus.ACTIVE,
        tasks=[
            PlanTask(
                id=0,
                description="Task 1",
                summary="First task",
                context="Context 1",
                status=TaskStatus.IN_PROGRESS,
            ),
        ],
        current_task_id=0,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


class TestToolHelper:
    """Tests for ToolHelper."""

    def test_init_stores_base_tools(self, helper, sample_tool):
        """Test that init stores base tools."""
        assert len(helper._base_tools) == 1
        assert helper._base_tools[0] == sample_tool

    def test_init_empty_plan_tools(self, helper):
        """Test that plan tools start empty."""
        assert helper._plan_tools == []

    def test_has_plan_tools_false_initially(self, helper):
        """Test has_plan_tools returns False initially."""
        assert helper.has_plan_tools() is False


class TestAddPlanTools:
    """Tests for add_plan_tools method."""

    def test_loads_plan_tools(self, helper, sample_plan, tmp_path):
        """Test loading plan tools from file."""
        # Create a mock plan.yaml
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text("""
type: plan
functions:
  - name: plan_update
    description: Update plan status
    parameters:
      type: object
      properties:
        status:
          type: string
        resolution:
          type: string
      required: [status, resolution]
""")

        helper.add_plan_tools(sample_plan, str(tmp_path))

        assert helper.has_plan_tools() is True
        assert len(helper._plan_tools) == 1
        # Tool user should have both base and plan tools
        assert len(helper._tool_user.tools) == 2

    def test_skips_if_already_loaded(self, helper, sample_plan, tmp_path):
        """Test that add_plan_tools skips if already loaded."""
        # Create a mock plan.yaml
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text("""
type: plan
functions:
  - name: plan_update
    description: Update plan status
    parameters:
      type: object
      properties:
        status:
          type: string
      required: [status]
""")

        helper.add_plan_tools(sample_plan, str(tmp_path))
        initial_count = len(helper._tool_user.tools)

        # Call again
        helper.add_plan_tools(sample_plan, str(tmp_path))

        # Should not double-add
        assert len(helper._tool_user.tools) == initial_count

    def test_handles_missing_file(self, helper, sample_plan, tmp_path):
        """Test graceful handling of missing plan.yaml."""
        # No plan.yaml exists
        helper.add_plan_tools(sample_plan, str(tmp_path))

        assert helper.has_plan_tools() is False

    def test_plan_tool_has_correct_function_name(self, helper, sample_plan, tmp_path):
        """Test that loaded plan tool has the correct function name."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text("""
type: plan
functions:
  - name: plan_update
    description: Update plan status
    parameters:
      type: object
      properties:
        status:
          type: string
          enum: [completed, blocked, skipped]
        resolution:
          type: string
      required: [status, resolution]
""")

        helper.add_plan_tools(sample_plan, str(tmp_path))

        plan_tool = helper._plan_tools[0]
        assert plan_tool.function.name == "plan_update"
        assert plan_tool.type == "plan"


class TestRemovePlanTools:
    """Tests for remove_plan_tools method."""

    def test_removes_plan_tools(self, helper, sample_plan, tmp_path):
        """Test removing plan tools."""
        # Create and load plan tools
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text("""
type: plan
functions:
  - name: plan_update
    description: Update plan status
    parameters:
      type: object
      properties:
        status:
          type: string
      required: [status]
""")
        helper.add_plan_tools(sample_plan, str(tmp_path))
        assert helper.has_plan_tools() is True

        helper.remove_plan_tools()

        assert helper.has_plan_tools() is False
        assert helper._plan_tools == []
        # Tool user should only have base tools
        assert len(helper._tool_user.tools) == 1

    def test_noop_if_no_plan_tools(self, helper):
        """Test remove_plan_tools is safe when no tools loaded."""
        helper.remove_plan_tools()  # Should not raise
        assert helper.has_plan_tools() is False

    def test_base_tools_preserved_after_remove(self, helper, sample_plan, sample_tool, tmp_path):
        """Test that base tools are preserved after removing plan tools."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text("""
type: plan
functions:
  - name: plan_update
    description: Update plan status
    parameters:
      type: object
      properties:
        status:
          type: string
      required: [status]
""")
        helper.add_plan_tools(sample_plan, str(tmp_path))
        helper.remove_plan_tools()

        # Base tool should still be there
        assert len(helper._tool_user.tools) == 1
        assert helper._tool_user.tools[0].function.name == "test_function"


class TestFromFile:
    """Tests for from_file class method."""

    def test_creates_helper_from_file(self, tmp_path):
        """Test creating helper from a tool config file."""
        tool_yaml = tmp_path / "agent.yaml"
        tool_yaml.write_text("""
type: agent
functions:
  - name: speak
    description: Say something
    parameters:
      type: object
      properties:
        text:
          type: string
      required: [text]
""")

        helper = ToolHelper.from_file(str(tool_yaml), str(tmp_path))

        assert len(helper._base_tools) == 1
        assert helper._base_tools[0].function.name == "speak"

    def test_from_file_initializes_empty_plan_tools(self, tmp_path):
        """Test that from_file initializes with empty plan tools."""
        tool_yaml = tmp_path / "agent.yaml"
        tool_yaml.write_text("""
type: agent
functions:
  - name: speak
    description: Say something
    parameters:
      type: object
      properties:
        text:
          type: string
      required: [text]
""")

        helper = ToolHelper.from_file(str(tool_yaml), str(tmp_path))

        assert helper._plan_tools == []
        assert helper.has_plan_tools() is False


class TestDecorateXml:
    """Tests for decorate_xml method."""

    def test_decorate_xml_includes_plan_tools_after_add(self, helper, sample_plan, tmp_path):
        """Test that decorate_xml includes plan tools after they are added."""
        from aim.utils.xml import XmlFormatter

        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text("""
type: plan
functions:
  - name: plan_update
    description: Update plan status
    parameters:
      type: object
      properties:
        status:
          type: string
      required: [status]
""")

        # Before adding plan tools
        xml_before = XmlFormatter()
        helper.decorate_xml(xml_before)
        xml_content_before = xml_before.render()

        helper.add_plan_tools(sample_plan, str(tmp_path))

        # After adding plan tools
        xml_after = XmlFormatter()
        helper.decorate_xml(xml_after)
        xml_content_after = xml_after.render()

        # The XML after should contain the plan_update tool
        assert "plan_update" in xml_content_after
        # The XML before should not contain plan_update
        assert "plan_update" not in xml_content_before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
