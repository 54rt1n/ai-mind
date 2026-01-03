# tests/unit/dreamer/test_dialogue_integration.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Integration tests for dialogue flow through worker and API."""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from aim.dreamer.worker import DreamerWorker
from aim.dreamer.api import start_pipeline
from aim.dreamer.state import StateStore
from aim.dreamer.scheduler import Scheduler
from aim.dreamer.models import StepJob, StepStatus
from aim.dreamer.dialogue.models import DialogueState, DialogueTurn
from aim.dreamer.dialogue.strategy import DialogueStrategy
from aim.dreamer.dialogue.scenario import DialogueScenario
from aim.config import ChatConfig


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock()
    redis.delete = AsyncMock()
    redis.hset = AsyncMock()
    redis.hget = AsyncMock(return_value=None)
    redis.hgetall = AsyncMock(return_value={})
    redis.hdel = AsyncMock()
    return redis


@pytest.fixture
def state_store(mock_redis):
    """Create StateStore with mocked Redis."""
    return StateStore(mock_redis)


@pytest.fixture
def mock_config():
    """Create mock ChatConfig."""
    config = Mock(spec=ChatConfig)
    config.default_model = "test-model"
    config.thought_model = None
    config.codex_model = None
    config.guidance = None
    config.persona_mood = None
    config.persona_id = "Andi"
    config.user_id = "user"
    config.temperature = 0.7
    return config


@pytest.fixture
def dialogue_state():
    """Create a DialogueState fixture."""
    return DialogueState(
        pipeline_id="test-pipeline-123",
        strategy_name="analysis_dialogue",
        conversation_id="conv-123",
        persona_id="Andi",
        user_id="user",
        model="test-model",
    )


class TestStateStoreDialogue:
    """Test StateStore dialogue methods."""

    @pytest.mark.asyncio
    async def test_get_state_type_returns_dialogue_for_dialogue_state(
        self, state_store, mock_redis, dialogue_state
    ):
        """get_state_type returns 'dialogue' when DialogueState is stored."""
        # Store dialogue state JSON
        state_json = dialogue_state.model_dump_json()
        mock_redis.get = AsyncMock(return_value=state_json)

        result = await state_store.get_state_type("test-pipeline-123")

        assert result == "dialogue"
        mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_state_type_returns_pipeline_for_pipeline_state(
        self, state_store, mock_redis
    ):
        """get_state_type returns 'pipeline' when PipelineState is stored."""
        # Create pipeline state JSON (has scenario_name, not strategy_name)
        pipeline_json = json.dumps({
            "pipeline_id": "test-123",
            "scenario_name": "analyst",
            "persona_id": "Andi",
            "user_id": "user",
            "model": "test-model",
            "branch": 0,
            "step_counter": 1,
            "completed_steps": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        mock_redis.get = AsyncMock(return_value=pipeline_json)

        result = await state_store.get_state_type("test-123")

        assert result == "pipeline"

    @pytest.mark.asyncio
    async def test_get_state_type_returns_none_when_not_found(
        self, state_store, mock_redis
    ):
        """get_state_type returns None when no state exists."""
        mock_redis.get = AsyncMock(return_value=None)

        result = await state_store.get_state_type("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_save_and_load_dialogue_state(self, state_store, mock_redis, dialogue_state):
        """DialogueState round-trips through save/load."""
        # Capture what gets saved
        saved_data = {}

        async def capture_set(key, value):
            saved_data[key] = value

        mock_redis.set = AsyncMock(side_effect=capture_set)

        # Save
        await state_store.save_dialogue_state(dialogue_state)

        # Verify something was saved
        assert len(saved_data) == 1
        saved_key = list(saved_data.keys())[0]
        assert "pipeline" in saved_key  # Uses same key pattern as pipeline state

        # Setup load to return saved data
        mock_redis.get = AsyncMock(return_value=saved_data[saved_key])

        # Load
        loaded = await state_store.load_dialogue_state("test-pipeline-123")

        assert loaded is not None
        assert loaded.pipeline_id == dialogue_state.pipeline_id
        assert loaded.strategy_name == dialogue_state.strategy_name
        assert loaded.persona_id == dialogue_state.persona_id


class TestWorkerDialogueRouting:
    """Test worker routes dialogue flows correctly."""

    @pytest.mark.asyncio
    async def test_worker_routes_to_dialogue_handler(self, mock_config, dialogue_state):
        """Worker detects dialogue state and routes to _process_dialogue_step."""
        mock_state_store = AsyncMock()
        mock_scheduler = AsyncMock()

        # Setup: lock acquired, state type is dialogue
        mock_state_store.acquire_lock = AsyncMock(return_value=True)
        mock_state_store.get_step_status = AsyncMock(return_value=StepStatus.PENDING)
        mock_state_store.set_step_status = AsyncMock()
        mock_state_store.get_state_type = AsyncMock(return_value="dialogue")
        mock_state_store.load_dialogue_state = AsyncMock(return_value=dialogue_state)
        mock_state_store.save_dialogue_state = AsyncMock()
        mock_state_store.release_lock = AsyncMock()

        mock_scheduler.mark_complete = AsyncMock()
        mock_scheduler.enqueue_step = AsyncMock()
        mock_scheduler.check_pipeline_complete = AsyncMock()
        mock_scheduler.mark_failed = AsyncMock()

        worker = DreamerWorker(
            config=mock_config,
            state_store=mock_state_store,
            scheduler=mock_scheduler,
        )

        # Setup roster with persona
        mock_persona = Mock()
        mock_persona.persona_id = "Andi"
        mock_persona.full_name = "Andi"
        mock_persona.pronouns = {"subj": "they", "obj": "them", "poss": "their"}
        mock_persona.aspects = {}
        mock_roster = Mock()
        mock_roster.personas = {"Andi": mock_persona}
        worker.roster = mock_roster

        # Mock CVM
        worker.cvm = Mock()

        job = StepJob(
            pipeline_id="test-pipeline-123",
            step_id="ner_request",
            attempt=1,
            max_attempts=3,
            enqueued_at=datetime.now(timezone.utc),
        )

        # Mock the dialogue execution path
        with patch.object(worker, "_process_dialogue_step", new_callable=AsyncMock) as mock_process:
            with patch("aim.dreamer.worker.load_scenario") as mock_load:
                mock_scenario = Mock()
                mock_scenario.name = "analysis_dialogue"
                mock_scenario.flow = "dialogue"
                mock_load.return_value = mock_scenario

                await worker.process_job(job)

                # Verify dialogue handler was called
                mock_process.assert_called_once()
                call_args = mock_process.call_args
                assert call_args[0][0] == job  # First arg is job
                assert call_args[0][1] == mock_scenario  # Second arg is scenario

    @pytest.mark.asyncio
    async def test_worker_routes_to_standard_handler_for_pipeline_state(self, mock_config):
        """Worker detects pipeline state and uses standard execute_step."""
        mock_state_store = AsyncMock()
        mock_scheduler = AsyncMock()

        # Create pipeline state
        from aim.dreamer.models import PipelineState, Scenario, ScenarioContext, StepDefinition, StepOutput

        pipeline_state = PipelineState(
            pipeline_id="test-pipeline-456",
            scenario_name="summarizer",
            conversation_id="conv-456",
            persona_id="Andi",
            user_id="user",
            model="test-model",
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        scenario = Scenario(
            name="summarizer",
            context=ScenarioContext(required_aspects=[]),
            steps={
                "summarize": StepDefinition(
                    id="summarize",
                    prompt="Summarize",
                    output=StepOutput(document_type="summary"),
                    next=[],
                )
            }
        )

        # Setup: lock acquired, state type is pipeline
        mock_state_store.acquire_lock = AsyncMock(return_value=True)
        mock_state_store.get_step_status = AsyncMock(return_value=StepStatus.PENDING)
        mock_state_store.set_step_status = AsyncMock()
        mock_state_store.get_state_type = AsyncMock(return_value="pipeline")
        mock_state_store.load_state = AsyncMock(return_value=pipeline_state)
        mock_state_store.save_state = AsyncMock()
        mock_state_store.release_lock = AsyncMock()

        mock_scheduler.all_deps_complete = AsyncMock(return_value=True)
        mock_scheduler.mark_complete = AsyncMock()
        mock_scheduler.enqueue_step = AsyncMock()
        mock_scheduler.check_pipeline_complete = AsyncMock()

        worker = DreamerWorker(
            config=mock_config,
            state_store=mock_state_store,
            scheduler=mock_scheduler,
        )

        # Setup roster and CVM
        mock_persona = Mock()
        mock_roster = Mock()
        mock_roster.personas = {"Andi": mock_persona}
        worker.roster = mock_roster
        worker.cvm = Mock()
        worker.cvm.insert = Mock()

        job = StepJob(
            pipeline_id="test-pipeline-456",
            step_id="summarize",
            attempt=1,
            max_attempts=3,
            enqueued_at=datetime.now(timezone.utc),
        )

        # Mock execute_step (standard path)
        from aim.dreamer.models import StepResult
        step_result = StepResult(
            step_id="summarize",
            response="Summary text",
            doc_id="doc-123",
            document_type="summary",
            document_weight=1.0,
            tokens_used=100,
            timestamp=datetime.now(timezone.utc),
        )

        with patch("aim.dreamer.worker.load_scenario", return_value=scenario):
            with patch("aim.dreamer.worker.execute_step", new_callable=AsyncMock) as mock_execute:
                with patch("aim.dreamer.worker.create_message") as mock_create:
                    mock_execute.return_value = (step_result, [], False)
                    mock_create.return_value = Mock()

                    await worker.process_job(job)

                    # Verify standard execute_step was called
                    mock_execute.assert_called_once()


class TestStartPipelineDialogue:
    """Test start_pipeline creates correct state for dialogue flows."""

    @pytest.mark.asyncio
    async def test_start_pipeline_creates_dialogue_state(self, mock_config):
        """start_pipeline creates DialogueState for dialogue scenarios."""
        mock_state_store = AsyncMock()
        mock_scheduler = AsyncMock()

        mock_state_store.save_dialogue_state = AsyncMock()
        mock_state_store.init_dag = AsyncMock()
        mock_scheduler.enqueue_step = AsyncMock()

        # Mock CVM and roster
        mock_cvm = Mock()
        mock_roster = Mock()
        mock_persona = Mock()
        mock_persona.persona_id = "Andi"
        mock_roster.personas = {"Andi": mock_persona}

        with patch("aim.dreamer.api.load_scenario") as mock_load_scenario:
            with patch("aim.dreamer.api.ConversationModel") as mock_cvm_class:
                with patch("aim.dreamer.api.Roster") as mock_roster_class:
                    with patch("aim.dreamer.api.LanguageModelV2") as mock_lm:
                        with patch("aim.dreamer.api.DialogueStrategy") as mock_strategy_class:
                            # Setup scenario with flow=dialogue
                            mock_scenario = Mock()
                            mock_scenario.name = "analysis_dialogue"
                            mock_scenario.flow = "dialogue"
                            mock_scenario.requires_conversation = True
                            mock_load_scenario.return_value = mock_scenario

                            mock_cvm_class.from_config.return_value = mock_cvm
                            mock_cvm.index = Mock()
                            mock_cvm.index.search.return_value = Mock(empty=False)
                            mock_cvm.index.search.return_value.iloc = {0: {"persona_id": "Andi"}}

                            import pandas as pd
                            mock_cvm.index.search.return_value = pd.DataFrame([{"persona_id": "Andi"}])
                            mock_cvm.get_next_branch.return_value = 1

                            mock_roster_class.from_config.return_value = mock_roster
                            mock_lm.index_models.return_value = {"test-model": Mock()}

                            # Setup strategy mock
                            mock_strategy = Mock()
                            mock_strategy.get_execution_order.return_value = ["ner_request"]
                            mock_strategy_class.load.return_value = mock_strategy

                            pipeline_id = await start_pipeline(
                                scenario_name="analysis_dialogue",
                                config=mock_config,
                                model_name="test-model",
                                state_store=mock_state_store,
                                scheduler=mock_scheduler,
                                conversation_id="conv-123",
                                persona_id="Andi",
                            )

                            # Verify DialogueState was saved (not PipelineState)
                            mock_state_store.save_dialogue_state.assert_called_once()
                            saved_state = mock_state_store.save_dialogue_state.call_args[0][0]
                            assert isinstance(saved_state, DialogueState)
                            assert saved_state.strategy_name == "analysis_dialogue"

                            # Verify first step was enqueued
                            mock_scheduler.enqueue_step.assert_called_once()
                            assert mock_scheduler.enqueue_step.call_args[0][1] == "ner_request"
