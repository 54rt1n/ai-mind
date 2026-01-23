# tests/mud_tests/unit/conversation/test_plan_integration.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for plan integration in memory strategies."""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

from andimud_worker.conversation.memory.decision import MUDDecisionStrategy
from aim.utils.xml import XmlFormatter


@pytest.fixture
def mock_chat():
    chat = MagicMock()
    chat.config = MagicMock()
    chat.config.system_message = ""
    chat.current_location = ""
    return chat


@pytest.fixture
def strategy(mock_chat):
    return MUDDecisionStrategy(mock_chat)


@pytest.fixture
def sample_plan():
    from aim_mud_types.models.plan import AgentPlan, PlanTask, PlanStatus, TaskStatus
    return AgentPlan(
        plan_id="test-plan-123", agent_id="andi",
        objective="Test the plan integration", summary="Test integration plan",
        status=PlanStatus.ACTIVE,
        tasks=[
            PlanTask(id=0, description="First task - do something important",
                     summary="First task", context="Sets the foundation",
                     status=TaskStatus.IN_PROGRESS),
            PlanTask(id=1, description="Second task - follow up",
                     summary="Second task", context="Builds on first",
                     status=TaskStatus.PENDING),
        ],
        current_task_id=0,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


class TestGetConsciousnessHead:
    def test_no_active_plan(self, strategy):
        formatter = XmlFormatter()
        result = strategy.get_consciousness_head(formatter)
        assert "Active Plan" not in result.render()

    def test_with_active_plan(self, strategy, sample_plan):
        strategy._active_plan = sample_plan
        formatter = XmlFormatter()
        result = strategy.get_consciousness_head(formatter)
        rendered = result.render()
        assert "Active Plan" in rendered
        assert "Test the plan integration" in rendered
        assert "First task" in rendered

    def test_task_markers(self, strategy, sample_plan):
        strategy._active_plan = sample_plan
        formatter = XmlFormatter()
        rendered = strategy.get_consciousness_head(formatter).render()
        assert "[>]" in rendered  # in_progress
        assert "[ ]" in rendered  # pending

    def test_plan_summary_and_status(self, strategy, sample_plan):
        strategy._active_plan = sample_plan
        formatter = XmlFormatter()
        rendered = strategy.get_consciousness_head(formatter).render()
        assert "Test integration plan" in rendered
        assert "active" in rendered

    def test_current_task_details(self, strategy, sample_plan):
        strategy._active_plan = sample_plan
        formatter = XmlFormatter()
        rendered = strategy.get_consciousness_head(formatter).render()
        assert "First task - do something important" in rendered
        assert "Sets the foundation" in rendered

    def test_all_task_status_markers(self, strategy, sample_plan):
        from aim_mud_types.models.plan import TaskStatus
        # Modify tasks to have all status types
        sample_plan.tasks[0].status = TaskStatus.COMPLETED
        sample_plan.tasks[1].status = TaskStatus.BLOCKED
        strategy._active_plan = sample_plan
        formatter = XmlFormatter()
        rendered = strategy.get_consciousness_head(formatter).render()
        assert "[x]" in rendered  # completed
        assert "[!]" in rendered  # blocked

    def test_task_numbering(self, strategy, sample_plan):
        strategy._active_plan = sample_plan
        formatter = XmlFormatter()
        rendered = strategy.get_consciousness_head(formatter).render()
        assert "1." in rendered
        assert "2." in rendered


class TestGetPlanGuidance:
    def test_no_active_plan(self, strategy):
        assert strategy.get_plan_guidance() == ""

    def test_with_active_plan(self, strategy, sample_plan):
        strategy._active_plan = sample_plan
        result = strategy.get_plan_guidance()
        assert "Test integration plan" in result
        assert "First task" in result

    def test_all_tasks_complete(self, strategy, sample_plan):
        sample_plan.current_task_id = 2
        strategy._active_plan = sample_plan
        result = strategy.get_plan_guidance()
        assert "All tasks complete" in result

    def test_guidance_format(self, strategy, sample_plan):
        strategy._active_plan = sample_plan
        result = strategy.get_plan_guidance()
        assert "Executing plan:" in result
        assert "Current task:" in result


class TestPlanAttributeInitialization:
    def test_active_plan_initially_none(self, strategy):
        assert strategy._active_plan is None

    def test_active_plan_can_be_set(self, strategy, sample_plan):
        strategy._active_plan = sample_plan
        assert strategy._active_plan is sample_plan
        assert strategy._active_plan.plan_id == "test-plan-123"

    def test_redis_context_initially_none(self, strategy):
        """Test that Redis context attributes are None on init."""
        assert strategy._redis_client is None
        assert strategy._agent_id is None
        assert strategy._plan_tool_impl is None


class TestSetContext:
    """Tests for set_context() method."""

    def test_sets_redis_client(self, strategy):
        """Test that set_context stores the Redis client."""
        mock_redis = MagicMock()
        strategy.set_context(mock_redis, "test-agent")
        assert strategy._redis_client is mock_redis

    def test_sets_agent_id(self, strategy):
        """Test that set_context stores the agent ID."""
        mock_redis = MagicMock()
        strategy.set_context(mock_redis, "test-agent")
        assert strategy._agent_id == "test-agent"


class TestGetPlanToolImpl:
    """Tests for get_plan_tool_impl() method."""

    def test_returns_none_when_not_set(self, strategy):
        """Test that get_plan_tool_impl returns None initially."""
        assert strategy.get_plan_tool_impl() is None

    def test_returns_tool_when_set(self, strategy):
        """Test that get_plan_tool_impl returns the tool when set."""
        mock_tool = MagicMock()
        strategy._plan_tool_impl = mock_tool
        assert strategy.get_plan_tool_impl() is mock_tool


class TestInitToolsWithPlan:
    """Tests for conditional plan tool loading in init_tools()."""

    def test_loads_plan_tools_when_plan_active(self, strategy, sample_plan, tmp_path):
        """Test that plan tools are loaded when a plan is active."""
        # Create a mock tools directory with required files
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()

        # Create base tool file
        base_tool = tools_dir / "mud_decision.yaml"
        base_tool.write_text("""
type: decision
functions:
  - name: speak
    description: Say something
    parameters:
      type: object
      properties: {}
""")

        # Create plan tool file
        plan_tool = tools_dir / "plan.yaml"
        plan_tool.write_text("""
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

        # Set up context for plan tool loading
        mock_redis = MagicMock()
        strategy.set_context(mock_redis, "test-agent")
        strategy._active_plan = sample_plan

        # Initialize tools
        strategy.init_tools(str(base_tool), str(tools_dir))

        # Should have created plan tool impl
        assert strategy._plan_tool_impl is not None
        # Tool user should have both tools
        tool_names = [t.function.name for t in strategy.tool_user.tools]
        assert "speak" in tool_names
        assert "plan_update" in tool_names

    def test_skips_plan_tools_when_no_plan(self, strategy, tmp_path):
        """Test that plan tools are not loaded when no plan is active."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()

        # Create base tool file
        base_tool = tools_dir / "mud_decision.yaml"
        base_tool.write_text("""
type: decision
functions:
  - name: speak
    description: Say something
    parameters:
      type: object
      properties: {}
""")

        # Create plan tool file (should NOT be loaded)
        plan_tool = tools_dir / "plan.yaml"
        plan_tool.write_text("""
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

        # No plan set
        strategy._active_plan = None

        # Initialize tools
        strategy.init_tools(str(base_tool), str(tools_dir))

        # Should NOT have created plan tool impl
        assert strategy._plan_tool_impl is None
        # Tool user should only have base tools
        tool_names = [t.function.name for t in strategy.tool_user.tools]
        assert "speak" in tool_names
        assert "plan_update" not in tool_names

    def test_skips_plan_tools_when_no_context(self, strategy, sample_plan, tmp_path):
        """Test that plan tools are not loaded when Redis context is not set."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()

        base_tool = tools_dir / "mud_decision.yaml"
        base_tool.write_text("""
type: decision
functions:
  - name: speak
    description: Say something
    parameters:
      type: object
      properties: {}
""")

        plan_tool = tools_dir / "plan.yaml"
        plan_tool.write_text("""
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

        # Plan is active but no context set
        strategy._active_plan = sample_plan
        # Don't call set_context()

        strategy.init_tools(str(base_tool), str(tools_dir))

        # Should NOT have created plan tool impl
        assert strategy._plan_tool_impl is None

    def test_plan_tool_impl_receives_context(self, strategy, sample_plan, tmp_path):
        """Test that PlanExecutionTool receives Redis context."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()

        base_tool = tools_dir / "mud_decision.yaml"
        base_tool.write_text("""
type: decision
functions:
  - name: speak
    description: Say something
    parameters:
      type: object
      properties: {}
""")

        plan_tool = tools_dir / "plan.yaml"
        plan_tool.write_text("""
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

        mock_redis = MagicMock()
        strategy.set_context(mock_redis, "test-agent-xyz")
        strategy._active_plan = sample_plan

        strategy.init_tools(str(base_tool), str(tools_dir))

        # Plan tool impl should have received context
        assert strategy._plan_tool_impl._redis_client is mock_redis
        assert strategy._plan_tool_impl._agent_id == "test-agent-xyz"
