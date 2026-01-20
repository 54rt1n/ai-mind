# packages/aim-mud/tests/mud_tests/unit/worker/test_dream_planner.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for DreamCommand planner pipeline handling."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from andimud_worker.commands.dream import DreamCommand
from aim_mud_types import TurnRequestStatus


@pytest.fixture
def mock_worker():
    """Create a mock worker."""
    from andimud_worker.config import MUDConfig

    worker = MagicMock()
    worker.chat_config = MagicMock()
    worker.redis = AsyncMock()
    worker.config = MUDConfig(agent_id="andi", persona_id="andi")
    worker.cvm = MagicMock()
    worker.process_dream_turn = AsyncMock()
    worker._update_conversation_report = AsyncMock()
    return worker


@pytest.fixture
def command():
    """Create a DreamCommand instance."""
    return DreamCommand()


class TestPlannerPipeline:
    """Tests for planner pipeline handling."""

    @pytest.mark.asyncio
    async def test_missing_objective(self, command, mock_worker):
        """Test error when objective is missing."""
        result = await command.execute(
            mock_worker,
            turn_id="test-turn",
            metadata={"pipeline": "planner"},
        )

        assert result.complete is True
        assert result.status == TurnRequestStatus.FAIL
        assert "missing objective" in result.message

    @pytest.mark.asyncio
    async def test_planner_success(self, command, mock_worker):
        """Test successful plan creation.

        Uses patch.object on the command's _execute_planner to avoid
        importing external modules that have missing dependencies.
        """
        mock_plan = MagicMock()
        mock_plan.plan_id = "plan-123"
        mock_plan.summary = "Test plan summary"

        # Create successful result that _execute_planner would return
        success_result = MagicMock()
        success_result.complete = True
        success_result.status = TurnRequestStatus.DONE
        success_result.message = f"Plan created: {mock_plan.summary}"

        with patch.object(
            command, "_execute_planner", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = success_result

            result = await command.execute(
                mock_worker,
                turn_id="test-turn",
                metadata={"pipeline": "planner", "objective": "Test objective"},
            )

        assert result.complete is True
        assert result.status == TurnRequestStatus.DONE
        assert "Test plan summary" in result.message
        mock_execute.assert_called_once_with(
            mock_worker,
            {"pipeline": "planner", "objective": "Test objective"},
            "test-turn",
        )

    @pytest.mark.asyncio
    async def test_planner_failure(self, command, mock_worker):
        """Test plan creation failure.

        Uses patch.object on the command's _execute_planner to avoid
        importing external modules that have missing dependencies.
        """
        # Create failure result that _execute_planner would return
        fail_result = MagicMock()
        fail_result.complete = True
        fail_result.status = TurnRequestStatus.FAIL
        fail_result.message = "Plan creation failed"

        with patch.object(
            command, "_execute_planner", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = fail_result

            result = await command.execute(
                mock_worker,
                turn_id="test-turn",
                metadata={"pipeline": "planner", "objective": "Test objective"},
            )

        assert result.complete is True
        assert result.status == TurnRequestStatus.FAIL
        assert "failed" in result.message.lower()


class TestDreamPipelineFallback:
    """Tests for dream pipeline (non-planner) handling."""

    @pytest.mark.asyncio
    async def test_dream_without_pipeline_key(self, command, mock_worker):
        """Test dream execution when no pipeline key."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.pipeline_id = "dream-123"
        mock_result.duration_seconds = 5.0
        mock_worker.process_dream_turn.return_value = mock_result

        result = await command.execute(
            mock_worker,
            turn_id="test-turn",
            metadata={"scenario": "analysis_dialogue"},
        )

        assert result.complete is True
        assert result.status == TurnRequestStatus.DONE
        mock_worker.process_dream_turn.assert_called_once()

    @pytest.mark.asyncio
    async def test_dream_missing_scenario(self, command, mock_worker):
        """Test error when scenario is missing."""
        result = await command.execute(
            mock_worker,
            turn_id="test-turn",
            metadata={},
        )

        assert result.complete is True
        assert result.status == TurnRequestStatus.FAIL
        assert "missing scenario" in result.message

    @pytest.mark.asyncio
    async def test_dream_failure(self, command, mock_worker):
        """Test dream failure handling."""
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Dream error"
        mock_worker.process_dream_turn.return_value = mock_result

        result = await command.execute(
            mock_worker,
            turn_id="test-turn",
            metadata={"scenario": "analysis_dialogue"},
        )

        assert result.complete is True
        assert result.status == TurnRequestStatus.FAIL
        assert "Dream error" in result.message

    @pytest.mark.asyncio
    async def test_dream_with_all_metadata(self, command, mock_worker):
        """Test dream execution with all metadata fields."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.pipeline_id = "dream-456"
        mock_result.duration_seconds = 3.2
        mock_worker.process_dream_turn.return_value = mock_result

        result = await command.execute(
            mock_worker,
            turn_id="test-turn",
            metadata={
                "scenario": "journaler_dialogue",
                "query": "What happened today?",
                "guidance": "Focus on emotions",
                "conversation_id": "conv-789",
            },
        )

        assert result.complete is True
        assert result.status == TurnRequestStatus.DONE

        # Verify process_dream_turn was called with correct args
        mock_worker.process_dream_turn.assert_called_once_with(
            scenario="journaler_dialogue",
            query="What happened today?",
            guidance="Focus on emotions",
            triggered_by="manual",
            target_conversation_id="conv-789",
        )

        # Verify post-success actions
        mock_worker.cvm.refresh.assert_called_once()
        mock_worker._update_conversation_report.assert_called_once()


class TestPipelineRouting:
    """Tests for pipeline routing logic."""

    @pytest.mark.asyncio
    async def test_routes_to_planner_when_pipeline_is_planner(self, command, mock_worker):
        """Test that pipeline=planner routes to planner handler."""
        # Without objective, should fail with planner-specific error
        result = await command.execute(
            mock_worker,
            turn_id="test-turn",
            metadata={"pipeline": "planner"},
        )

        # The error message proves it went to planner path
        assert "missing objective" in result.message

    @pytest.mark.asyncio
    async def test_routes_to_dream_when_pipeline_not_planner(self, command, mock_worker):
        """Test that other pipeline values route to dream handler."""
        # Without scenario, should fail with dream-specific error
        result = await command.execute(
            mock_worker,
            turn_id="test-turn",
            metadata={"pipeline": "other"},
        )

        # The error message proves it went to dream path
        assert "missing scenario" in result.message

    @pytest.mark.asyncio
    async def test_routes_to_dream_when_no_pipeline_key(self, command, mock_worker):
        """Test that missing pipeline key routes to dream handler."""
        result = await command.execute(
            mock_worker,
            turn_id="test-turn",
            metadata={},
        )

        # The error message proves it went to dream path
        assert "missing scenario" in result.message
