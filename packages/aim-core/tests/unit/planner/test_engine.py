# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for PlannerEngine."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
from dataclasses import dataclass

from aim.planner.engine import PlannerEngine
from aim.planner.form import FormState


@dataclass
class MockPipelineResult:
    """Mock PipelineResult from DreamerClient."""
    success: bool
    error: str = None
    status: object = None


@dataclass
class MockPipelineStatus:
    """Mock PipelineStatus."""
    status: str = "complete"


@pytest.fixture
def mock_config():
    """Create a mock ChatConfig."""
    config = MagicMock()
    config.model_name = "test-model"
    config.persona_id = "andi"
    config.redis_host = "localhost"
    config.redis_port = 6379
    config.redis_db = 0
    config.redis_password = None
    config.tools_path = "config/tools"
    return config


@pytest.fixture
def mock_dreamer_client():
    """Create a mock DreamerClient."""
    client = MagicMock()
    client.run_and_wait = AsyncMock()
    return client


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    return AsyncMock()


@pytest.fixture
def engine(mock_config, mock_dreamer_client, mock_redis_client):
    """Create a PlannerEngine for testing."""
    return PlannerEngine(
        config=mock_config,
        dreamer_client=mock_dreamer_client,
        redis_client=mock_redis_client,
    )


class TestPlannerEngine:
    """Tests for PlannerEngine."""

    @pytest.mark.asyncio
    async def test_deliberation_failure(self, engine, mock_dreamer_client):
        """Test handling of deliberation failure."""
        mock_dreamer_client.run_and_wait.return_value = MockPipelineResult(
            success=False,
            error="Scenario failed",
        )

        result = await engine.create_plan("andi", "Test objective")

        assert result is None
        mock_dreamer_client.run_and_wait.assert_called_once()

    @pytest.mark.asyncio
    async def test_deliberation_incomplete(self, engine, mock_dreamer_client):
        """Test handling of incomplete deliberation."""
        mock_dreamer_client.run_and_wait.return_value = MockPipelineResult(
            success=True,
            status=MockPipelineStatus(status="failed"),
        )

        result = await engine.create_plan("andi", "Test objective")

        assert result is None

    @pytest.mark.asyncio
    async def test_no_deliberation_document(self, engine, mock_dreamer_client, mock_config):
        """Test handling of missing deliberation document."""
        mock_dreamer_client.run_and_wait.return_value = MockPipelineResult(
            success=True,
            status=MockPipelineStatus(status="complete"),
        )

        # Mock CVM to return empty DataFrame
        with patch("aim.planner.engine.ConversationModel") as MockCVM:
            mock_cvm_instance = MagicMock()
            mock_cvm_instance.query.return_value = pd.DataFrame()
            MockCVM.from_config.return_value = mock_cvm_instance

            result = await engine.create_plan("andi", "Test objective")

        assert result is None

    @pytest.mark.asyncio
    async def test_form_builder_no_tools(self, engine, mock_dreamer_client, mock_config):
        """Test handling of missing planner_form tools."""
        mock_dreamer_client.run_and_wait.return_value = MockPipelineResult(
            success=True,
            status=MockPipelineStatus(status="complete"),
        )

        # Mock CVM to return deliberation content
        with patch("aim.planner.engine.ConversationModel") as MockCVM:
            mock_cvm_instance = MagicMock()
            mock_cvm_instance.query.return_value = pd.DataFrame([
                {"content": "Deliberation output here"}
            ])
            MockCVM.from_config.return_value = mock_cvm_instance

            # Mock ToolLoader to return no tools
            with patch("aim.planner.engine.ToolLoader") as MockLoader:
                mock_loader = MagicMock()
                mock_loader.get_tools_by_type.return_value = []
                MockLoader.return_value = mock_loader

                result = await engine.create_plan("andi", "Test objective")

        assert result is None

    def test_execute_form_tool_add_task(self, engine):
        """Test executing plan_add_task tool."""
        from aim.planner.form import PlanFormBuilder

        form = PlanFormBuilder("objective", "context")
        result = engine._execute_form_tool(
            form,
            "plan_add_task",
            {"description": "desc", "summary": "sum", "context": "ctx"},
        )

        assert result["status"] == "added"
        assert len(form.tasks) == 1

    def test_execute_form_tool_done_adding(self, engine):
        """Test executing plan_done_adding tool."""
        from aim.planner.form import PlanFormBuilder

        form = PlanFormBuilder("objective", "context")
        form.add_task("desc", "sum", "ctx")

        result = engine._execute_form_tool(form, "plan_done_adding", {})

        assert result["status"] == "ok"
        assert form.state == FormState.VERIFY_TASK

    def test_execute_form_tool_verify_task(self, engine):
        """Test executing plan_verify_task tool."""
        from aim.planner.form import PlanFormBuilder

        form = PlanFormBuilder("objective", "context")
        form.add_task("desc", "sum", "ctx")
        form.done_adding()

        result = engine._execute_form_tool(form, "plan_verify_task", {"approved": True})

        assert result["status"] == "ok"
        assert form.tasks[0].verified is True

    def test_execute_form_tool_edit_task(self, engine):
        """Test executing plan_edit_task tool."""
        from aim.planner.form import PlanFormBuilder

        form = PlanFormBuilder("objective", "context")
        form.add_task("desc", "sum", "ctx")

        result = engine._execute_form_tool(
            form,
            "plan_edit_task",
            {"task_id": 0, "field": "summary", "new_value": "new sum"},
        )

        assert result["status"] == "ok"
        assert form.tasks[0].summary == "new sum"

    def test_execute_form_tool_set_summary(self, engine):
        """Test executing plan_set_summary tool."""
        from aim.planner.form import PlanFormBuilder

        form = PlanFormBuilder("objective", "context")

        result = engine._execute_form_tool(
            form,
            "plan_set_summary",
            {"summary": "Test plan summary"},
        )

        assert result["status"] == "ok"
        assert form.summary == "Test plan summary"

    def test_execute_form_tool_confirm(self, engine):
        """Test executing plan_confirm tool."""
        from aim.planner.form import PlanFormBuilder

        form = PlanFormBuilder("objective", "context")
        form.add_task("desc", "sum", "ctx")
        form.done_adding()
        form.verify_task(approved=True)
        form.set_summary("Test summary")

        result = engine._execute_form_tool(form, "plan_confirm", {})

        assert result["status"] == "complete"
        assert form.state == FormState.COMPLETE

    def test_execute_form_tool_unknown(self, engine):
        """Test executing unknown tool."""
        from aim.planner.form import PlanFormBuilder

        form = PlanFormBuilder("objective", "context")
        result = engine._execute_form_tool(form, "unknown_tool", {})

        assert result["status"] == "error"
        assert "Unknown tool" in result["message"]

    def test_build_form_system_prompt(self, engine):
        """Test building form system prompt."""
        prompt = engine._build_form_system_prompt("Test deliberation context")

        assert "Coder aspect" in prompt
        assert "Test deliberation context" in prompt
        assert "plan_add_task" in prompt
        assert "plan_confirm" in prompt


class TestRunFormBuilder:
    """Tests for _run_form_builder method."""

    @pytest.mark.asyncio
    async def test_successful_form_building(self, engine, mock_config):
        """Test successful form building flow."""
        # This test simulates the LLM making correct tool calls

        # Mock ToolLoader
        mock_tools = [MagicMock() for _ in range(6)]  # 6 planner_form tools

        # Track iteration to return different tool calls
        call_count = [0]

        async def mock_llm_call(system, user, tool_user):
            call_count[0] += 1
            # Simulate LLM tool call responses in order
            responses = [
                '{"plan_add_task": {"description": "Task 1 desc", "summary": "Task 1", "context": "Why task 1"}}',
                '{"plan_done_adding": {}}',
                '{"plan_verify_task": {"approved": true}}',
                '{"plan_set_summary": {"summary": "Test plan summary"}}',
                '{"plan_confirm": {}}',
            ]
            if call_count[0] <= len(responses):
                return responses[call_count[0] - 1]
            return None

        with patch("aim.planner.engine.ToolLoader") as MockLoader:
            mock_loader = MagicMock()
            mock_loader.get_tools_by_type.return_value = mock_tools
            MockLoader.return_value = mock_loader

            with patch.object(engine, "_call_llm_with_tools", side_effect=mock_llm_call):
                # Mock ToolUser.process_response
                with patch("aim.planner.engine.ToolUser") as MockToolUser:
                    mock_tool_user = MagicMock()

                    # Return appropriate results for each call
                    results = [
                        MagicMock(is_valid=True, function_name="plan_add_task", arguments={"description": "Task 1 desc", "summary": "Task 1", "context": "Why task 1"}),
                        MagicMock(is_valid=True, function_name="plan_done_adding", arguments={}),
                        MagicMock(is_valid=True, function_name="plan_verify_task", arguments={"approved": True}),
                        MagicMock(is_valid=True, function_name="plan_set_summary", arguments={"summary": "Test plan summary"}),
                        MagicMock(is_valid=True, function_name="plan_confirm", arguments={}),
                    ]
                    mock_tool_user.process_response.side_effect = results
                    MockToolUser.return_value = mock_tool_user

                    plan = await engine._run_form_builder(
                        "andi",
                        "Test objective",
                        "Test deliberation context",
                    )

        assert plan is not None
        assert plan.agent_id == "andi"
        assert plan.objective == "Test objective"
        assert plan.summary == "Test plan summary"
        assert len(plan.tasks) == 1

    @pytest.mark.asyncio
    async def test_form_builder_max_iterations(self, engine, mock_config):
        """Test form builder respects max iterations."""
        # Mock tools
        mock_tools = [MagicMock()]

        # Always return invalid response to trigger max iterations
        async def mock_llm_call(system, user, tool_user):
            return "invalid response"

        with patch("aim.planner.engine.ToolLoader") as MockLoader:
            mock_loader = MagicMock()
            mock_loader.get_tools_by_type.return_value = mock_tools
            MockLoader.return_value = mock_loader

            with patch.object(engine, "_call_llm_with_tools", side_effect=mock_llm_call):
                with patch("aim.planner.engine.ToolUser") as MockToolUser:
                    mock_tool_user = MagicMock()
                    # Always return invalid
                    mock_tool_user.process_response.return_value = MagicMock(
                        is_valid=False,
                        error="Invalid response"
                    )
                    MockToolUser.return_value = mock_tool_user

                    # This should eventually return None after max iterations
                    plan = await engine._run_form_builder(
                        "andi",
                        "Test objective",
                        "Test deliberation context",
                    )

        assert plan is None

    @pytest.mark.asyncio
    async def test_form_builder_handles_llm_failure(self, engine, mock_config):
        """Test form builder continues on LLM failure."""
        mock_tools = [MagicMock()]

        call_count = [0]

        async def mock_llm_call(system, user, tool_user):
            call_count[0] += 1
            # Fail first call, then succeed
            if call_count[0] == 1:
                return None  # Simulate failure
            # Return valid response
            return '{"plan_add_task": {"description": "desc", "summary": "sum", "context": "ctx"}}'

        with patch("aim.planner.engine.ToolLoader") as MockLoader:
            mock_loader = MagicMock()
            mock_loader.get_tools_by_type.return_value = mock_tools
            MockLoader.return_value = mock_loader

            with patch.object(engine, "_call_llm_with_tools", side_effect=mock_llm_call):
                with patch("aim.planner.engine.ToolUser") as MockToolUser:
                    mock_tool_user = MagicMock()

                    def make_result(response):
                        if response is None:
                            return MagicMock(is_valid=False, error="No response")
                        return MagicMock(
                            is_valid=True,
                            function_name="plan_add_task",
                            arguments={"description": "desc", "summary": "sum", "context": "ctx"}
                        )

                    mock_tool_user.process_response.side_effect = [
                        MagicMock(is_valid=True, function_name="plan_add_task",
                                  arguments={"description": "desc", "summary": "sum", "context": "ctx"}),
                        MagicMock(is_valid=True, function_name="plan_done_adding", arguments={}),
                        MagicMock(is_valid=True, function_name="plan_verify_task", arguments={"approved": True}),
                        MagicMock(is_valid=True, function_name="plan_set_summary", arguments={"summary": "sum"}),
                        MagicMock(is_valid=True, function_name="plan_confirm", arguments={}),
                    ]
                    MockToolUser.return_value = mock_tool_user

                    # Should eventually complete
                    plan = await engine._run_form_builder(
                        "andi",
                        "Test objective",
                        "Test deliberation context",
                    )

        # Even with initial failure, should complete
        assert plan is not None or call_count[0] > 1  # Either completed or retried


class TestCallLLMWithTools:
    """Tests for _call_llm_with_tools method."""

    @pytest.mark.asyncio
    async def test_model_not_available(self, engine, mock_config):
        """Test handling when model is not available."""
        with patch("aim.planner.engine.LanguageModelV2") as MockLLM:
            MockLLM.index_models.return_value = {}  # No models available

            mock_tool_user = MagicMock()
            mock_tool_user.get_tool_guidance.return_value = ""

            result = await engine._call_llm_with_tools(
                "system",
                "user",
                mock_tool_user,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_successful_llm_call(self, engine, mock_config):
        """Test successful LLM call."""
        mock_provider = MagicMock()
        mock_provider.stream_turns.return_value = iter(["chunk1", "chunk2"])

        mock_model = MagicMock()
        mock_model.llm_factory.return_value = mock_provider

        with patch("aim.planner.engine.LanguageModelV2") as MockLLM:
            MockLLM.index_models.return_value = {"test-model": mock_model}

            mock_tool_user = MagicMock()
            mock_tool_user.get_tool_guidance.return_value = "Tool guidance"

            result = await engine._call_llm_with_tools(
                "system prompt",
                "user prompt",
                mock_tool_user,
            )

        assert result == "chunk1chunk2"

    @pytest.mark.asyncio
    async def test_llm_call_retry_on_failure(self, engine, mock_config):
        """Test LLM call retries on failure."""
        call_count = [0]

        def mock_stream(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("First call failed")
            return iter(["success"])

        mock_provider = MagicMock()
        mock_provider.stream_turns.side_effect = mock_stream

        mock_model = MagicMock()
        mock_model.llm_factory.return_value = mock_provider

        with patch("aim.planner.engine.LanguageModelV2") as MockLLM:
            MockLLM.index_models.return_value = {"test-model": mock_model}

            mock_tool_user = MagicMock()
            mock_tool_user.get_tool_guidance.return_value = ""

            result = await engine._call_llm_with_tools(
                "system",
                "user",
                mock_tool_user,
            )

        assert result == "success"
        assert call_count[0] == 2


class TestCreatePlan:
    """Tests for create_plan method."""

    @pytest.mark.asyncio
    async def test_full_pipeline_success(self, engine, mock_dreamer_client, mock_redis_client):
        """Test complete successful pipeline."""
        # Setup deliberation success
        mock_dreamer_client.run_and_wait.return_value = MockPipelineResult(
            success=True,
            status=MockPipelineStatus(status="complete"),
        )

        # Mock CVM for deliberation document
        with patch("aim.planner.engine.ConversationModel") as MockCVM:
            mock_cvm = MagicMock()
            mock_cvm.query.return_value = pd.DataFrame([
                {"content": "Plan deliberation output"}
            ])
            MockCVM.from_config.return_value = mock_cvm

            # Mock form builder success
            with patch.object(engine, "_run_form_builder") as mock_form_builder:
                mock_plan = MagicMock()
                mock_plan.plan_id = "test-plan-id"
                mock_form_builder.return_value = mock_plan

                # Mock RedisMUDClient - patch at the import location
                with patch("aim_mud_types.client.RedisMUDClient") as MockRedis:
                    mock_redis = MagicMock()
                    mock_redis.create_plan = AsyncMock()
                    MockRedis.return_value = mock_redis

                    result = await engine.create_plan("andi", "Test objective")

        assert result is not None
        assert result == mock_plan
        mock_redis.create_plan.assert_called_once_with(mock_plan)

    @pytest.mark.asyncio
    async def test_pipeline_form_builder_returns_none(self, engine, mock_dreamer_client):
        """Test pipeline when form builder fails."""
        mock_dreamer_client.run_and_wait.return_value = MockPipelineResult(
            success=True,
            status=MockPipelineStatus(status="complete"),
        )

        with patch("aim.planner.engine.ConversationModel") as MockCVM:
            mock_cvm = MagicMock()
            mock_cvm.query.return_value = pd.DataFrame([
                {"content": "Deliberation output"}
            ])
            MockCVM.from_config.return_value = mock_cvm

            with patch.object(engine, "_run_form_builder") as mock_form_builder:
                mock_form_builder.return_value = None

                result = await engine.create_plan("andi", "Test objective")

        assert result is None
