# packages/aim-mud/tests/mud_tests/unit/worker/test_scenario_step.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for execute_scenario_step() and initialize_scenario_dream()."""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aim_mud_types.models.coordination import DreamingState, DreamStatus


@pytest.fixture
def mock_worker():
    """Create a mock worker with necessary attributes."""
    from andimud_worker.config import MUDConfig

    worker = MagicMock()
    worker.config = MUDConfig(agent_id="andi", persona_id="andi")
    worker.config.agent_id = "andi"
    worker.config.persona_id = "andi"
    worker.chat_config = MagicMock()
    worker.cvm = MagicMock()
    worker.roster = MagicMock()
    worker.redis = AsyncMock()

    # Mock the mixin methods
    worker.load_dreaming_state = AsyncMock()
    worker.save_dreaming_state = AsyncMock()
    worker.archive_dreaming_state = AsyncMock()
    worker.delete_dreaming_state = AsyncMock()
    worker.update_dreaming_heartbeat = AsyncMock()
    worker.atomic_heartbeat_update = AsyncMock(return_value=1)

    return worker


@pytest.fixture
def sample_framework_dict():
    """Create sample ScenarioFramework as dict."""
    from aim.dreamer.core.framework import ScenarioFramework
    from aim.dreamer.core.models import ContextOnlyStepDefinition, MemoryAction

    framework = ScenarioFramework(
        name="test_scenario",
        first_step="gather",
        steps={
            "gather": ContextOnlyStepDefinition(
                id="gather",
                context=[MemoryAction(action="search_memories", top_n=5)],
                next=["end"]
            )
        }
    )
    return framework.model_dump()


@pytest.fixture
def sample_state_dict():
    """Create sample ScenarioState as dict."""
    from aim.dreamer.core.state import ScenarioState

    state = ScenarioState.initial(
        first_step="gather",
        conversation_id="conv123",
        guidance="Test guidance",
        query_text="Test query"
    )
    return state.model_dump()


@pytest.fixture
def sample_dreaming_state(sample_framework_dict, sample_state_dict):
    """Create a DreamingState with framework and state dicts."""
    return DreamingState(
        pipeline_id="pipeline-123",
        agent_id="andi",
        status=DreamStatus.RUNNING,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        scenario_name="test_scenario",
        execution_order=[],
        conversation_id="conv123",
        base_model="claude-3-5-sonnet-20241022",
        framework=sample_framework_dict,
        state=sample_state_dict,
    )


class TestExecuteScenarioStep:
    """Tests for execute_scenario_step()."""

    @pytest.mark.asyncio
    async def test_returns_true_when_no_dreaming_state(self, mock_worker):
        """Test returns True when no dreaming state exists."""
        mock_worker.load_dreaming_state.return_value = None

        from andimud_worker.mixins.dreamer import DreamerMixin

        result = await DreamerMixin.execute_scenario_step(mock_worker, "pipeline-123")

        assert result is True
        mock_worker.load_dreaming_state.assert_called_once_with("andi")

    @pytest.mark.asyncio
    async def test_returns_true_when_pipeline_id_mismatch(
        self, mock_worker, sample_dreaming_state
    ):
        """Test returns True when pipeline ID doesn't match."""
        mock_worker.load_dreaming_state.return_value = sample_dreaming_state

        from andimud_worker.mixins.dreamer import DreamerMixin

        result = await DreamerMixin.execute_scenario_step(mock_worker, "wrong-id")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_when_already_complete(
        self, mock_worker, sample_dreaming_state
    ):
        """Test returns True when dream is already complete."""
        sample_dreaming_state.status = DreamStatus.COMPLETE
        mock_worker.load_dreaming_state.return_value = sample_dreaming_state

        from andimud_worker.mixins.dreamer import DreamerMixin

        result = await DreamerMixin.execute_scenario_step(mock_worker, "pipeline-123")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_when_missing_framework(self, mock_worker):
        """Test returns True and fails dream when framework is missing."""
        dreaming_state = DreamingState(
            pipeline_id="pipeline-123",
            agent_id="andi",
            status=DreamStatus.RUNNING,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            scenario_name="test_scenario",
            execution_order=[],
            conversation_id="conv123",
            base_model="claude-3-5-sonnet-20241022",
            framework=None,  # Missing!
            state=None,  # Missing!
        )
        mock_worker.load_dreaming_state.return_value = dreaming_state

        from andimud_worker.mixins.dreamer import DreamerMixin

        result = await DreamerMixin.execute_scenario_step(mock_worker, "pipeline-123")

        assert result is True
        # Verify dream was marked as failed
        mock_worker.save_dreaming_state.assert_called()
        saved_state = mock_worker.save_dreaming_state.call_args[0][0]
        assert saved_state.status == DreamStatus.FAILED

    @pytest.mark.asyncio
    async def test_returns_true_when_scenario_already_at_end(
        self, mock_worker, sample_dreaming_state
    ):
        """Test returns True when scenario state is at 'end'."""
        from aim.dreamer.core.state import ScenarioState

        # Create a state that's already at end
        completed_state = ScenarioState(
            current_step="end",
            conversation_id="conv123"
        )
        sample_dreaming_state.state = completed_state.model_dump()
        mock_worker.load_dreaming_state.return_value = sample_dreaming_state

        from andimud_worker.mixins.dreamer import DreamerMixin

        result = await DreamerMixin.execute_scenario_step(mock_worker, "pipeline-123")

        assert result is True
        # Verify final status was COMPLETE (not ABORTED since step is "end")
        mock_worker.save_dreaming_state.assert_called()

    @pytest.mark.asyncio
    async def test_executes_step_and_updates_state(
        self, mock_worker, sample_dreaming_state
    ):
        """Test successful step execution updates state."""
        mock_worker.load_dreaming_state.return_value = sample_dreaming_state

        # Mock persona
        mock_persona = MagicMock()
        mock_persona.id = "test-persona"
        mock_persona.name = "Test"
        mock_persona.pronouns = {"subject": "they"}
        mock_worker.roster.get_persona.return_value = mock_persona

        # Mock the executor and strategy execution
        from aim.dreamer.core.strategy import ScenarioStepResult

        mock_result = ScenarioStepResult(
            success=True,
            next_step="end",
            state_changed=True,
            doc_created=False
        )

        with patch('aim.dreamer.core.strategy.StepFactory.create') as mock_factory:
            mock_strategy = MagicMock()
            mock_strategy.execute = AsyncMock(return_value=mock_result)
            mock_factory.return_value = mock_strategy

            # Mock memory_dsl to avoid actual CVM operations
            with patch('aim.dreamer.core.memory_dsl.execute_memory_actions') as mock_mem:
                mock_mem.return_value = []

                from andimud_worker.mixins.dreamer import DreamerMixin

                result = await DreamerMixin.execute_scenario_step(
                    mock_worker, "pipeline-123"
                )

        # Step executed successfully and scenario is complete
        assert result is True
        mock_worker.save_dreaming_state.assert_called()
        mock_worker.archive_dreaming_state.assert_called()
        mock_worker.delete_dreaming_state.assert_called()


class TestInitializeScenarioDream:
    """Tests for initialize_scenario_dream()."""

    @pytest.mark.asyncio
    async def test_creates_dreaming_state_with_serialized_framework(self, mock_worker):
        """Test creates DreamingState with serialized framework and state."""
        from aim.dreamer.core.framework import ScenarioFramework
        from aim.dreamer.core.models import ContextOnlyStepDefinition

        framework = ScenarioFramework(
            name="test_scenario",
            first_step="gather",
            steps={
                "gather": ContextOnlyStepDefinition(
                    id="gather",
                    next=["end"]
                )
            }
        )

        # Mock persona
        mock_persona = MagicMock()
        mock_persona.id = "test-persona"
        mock_persona.name = "Test"
        mock_worker.roster.get_persona.return_value = mock_persona

        # Mock ModelSet
        with patch('aim.llm.model_set.ModelSet.from_config') as mock_ms:
            mock_model_set = MagicMock()
            mock_model_set.default_model = "test-model"
            mock_ms.return_value = mock_model_set

            from andimud_worker.mixins.dreamer import DreamerMixin

            result = await DreamerMixin.initialize_scenario_dream(
                mock_worker,
                framework=framework,
                conversation_id="conv123",
                guidance="Test guidance",
                query_text="Test query"
            )

        # Verify DreamingState was created
        assert result.scenario_name == "test_scenario"
        assert result.status == DreamStatus.PENDING
        assert result.agent_id == "andi"
        assert result.guidance == "Test guidance"
        assert result.query == "Test query"
        assert result.conversation_id == "conv123"

        # Verify framework was set
        assert result.framework is not None
        assert result.framework["name"] == "test_scenario"
        assert result.framework["first_step"] == "gather"

        # Verify state was set
        assert result.state is not None
        assert result.state["current_step"] == "gather"
        assert result.state["guidance"] == "Test guidance"
        assert result.state["query_text"] == "Test query"

        # Verify state was saved
        mock_worker.save_dreaming_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_creates_unique_pipeline_id(self, mock_worker):
        """Test generates unique pipeline IDs."""
        from aim.dreamer.core.framework import ScenarioFramework
        from aim.dreamer.core.models import ContextOnlyStepDefinition

        framework = ScenarioFramework(
            name="test_scenario",
            first_step="step1",
            steps={
                "step1": ContextOnlyStepDefinition(id="step1", next=["end"])
            }
        )

        mock_persona = MagicMock()
        mock_persona.id = "test-persona"
        mock_worker.roster.get_persona.return_value = mock_persona

        with patch('aim.llm.model_set.ModelSet.from_config') as mock_ms:
            mock_model_set = MagicMock()
            mock_model_set.default_model = "test-model"
            mock_ms.return_value = mock_model_set

            from andimud_worker.mixins.dreamer import DreamerMixin

            result1 = await DreamerMixin.initialize_scenario_dream(
                mock_worker, framework=framework
            )
            result2 = await DreamerMixin.initialize_scenario_dream(
                mock_worker, framework=framework
            )

        assert result1.pipeline_id != result2.pipeline_id


class TestExecuteScenarioStepRetry:
    """Tests for retry handling in execute_scenario_step()."""

    @pytest.mark.asyncio
    async def test_records_error_on_strategy_failure(
        self, mock_worker, sample_dreaming_state
    ):
        """Test records error when strategy execution fails."""
        sample_dreaming_state.current_step_attempts = 0
        sample_dreaming_state.max_step_retries = 3
        mock_worker.load_dreaming_state.return_value = sample_dreaming_state

        # Mock persona
        mock_persona = MagicMock()
        mock_persona.id = "test-persona"
        mock_worker.roster.get_persona.return_value = mock_persona

        # Mock strategy that raises an error
        with patch('aim.dreamer.core.strategy.StepFactory.create') as mock_factory:
            mock_strategy = MagicMock()
            mock_strategy.execute = AsyncMock(side_effect=ValueError("Test error"))
            mock_factory.return_value = mock_strategy

            from andimud_worker.mixins.dreamer import DreamerMixin

            result = await DreamerMixin.execute_scenario_step(
                mock_worker, "pipeline-123"
            )

        # Should not be complete (will retry)
        assert result is False

        # Verify error was recorded
        mock_worker.save_dreaming_state.assert_called()
        saved_state = mock_worker.save_dreaming_state.call_args[0][0]
        assert saved_state.current_step_attempts == 1
        assert saved_state.last_error == "Test error"
        assert saved_state.next_retry_at is not None

    @pytest.mark.asyncio
    async def test_marks_failed_after_max_retries(
        self, mock_worker, sample_dreaming_state
    ):
        """Test marks dream as failed after max retries."""
        sample_dreaming_state.current_step_attempts = 2  # Will become 3
        sample_dreaming_state.max_step_retries = 3
        mock_worker.load_dreaming_state.return_value = sample_dreaming_state

        mock_persona = MagicMock()
        mock_persona.id = "test-persona"
        mock_worker.roster.get_persona.return_value = mock_persona

        with patch('aim.dreamer.core.strategy.StepFactory.create') as mock_factory:
            mock_strategy = MagicMock()
            mock_strategy.execute = AsyncMock(side_effect=ValueError("Final error"))
            mock_factory.return_value = mock_strategy

            from andimud_worker.mixins.dreamer import DreamerMixin

            result = await DreamerMixin.execute_scenario_step(
                mock_worker, "pipeline-123"
            )

        # Should be complete (failed permanently)
        assert result is True

        # Verify dream was marked as failed
        saved_state = mock_worker.save_dreaming_state.call_args[0][0]
        assert saved_state.status == DreamStatus.FAILED
        mock_worker.archive_dreaming_state.assert_called()
        mock_worker.delete_dreaming_state.assert_called()
