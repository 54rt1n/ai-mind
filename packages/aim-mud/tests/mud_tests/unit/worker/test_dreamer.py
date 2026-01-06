# packages/aim-mud/tests/mud_tests/unit/worker/test_dreamer.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for DreamerRunner and DreamerMixin."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from andimud_worker.dreamer.runner import (
    DreamerRunner,
    DreamRequest,
    DreamResult,
    CONVERSATION_ANALYSIS_SCENARIOS,
)
from andimud_worker.mixins.dreamer import DreamerMixin
from andimud_worker.worker import MUDAgentWorker
from andimud_worker.config import MUDConfig
from aim_mud_types import MUDSession
from aim.config import ChatConfig
from aim.dreamer.models import StepJob, StepStatus
from aim_mud_types import RedisKeys


@pytest.fixture
def chat_config():
    """Create a test ChatConfig."""
    config = ChatConfig()
    config.llm_provider = "anthropic"
    config.model_name = "claude-opus-4-5-20251101"
    config.anthropic_api_key = "test-key"
    return config


@pytest.fixture
def mud_config():
    """Create a test MUD configuration."""
    return MUDConfig(
        agent_id="test_agent",
        persona_id="test_persona",
        redis_url="redis://localhost:6379",
    )


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    redis = AsyncMock()
    redis.hgetall = AsyncMock(return_value={})
    redis.hget = AsyncMock(return_value=None)
    redis.hset = AsyncMock(return_value=1)
    redis.get = AsyncMock(return_value=None)
    redis.eval = AsyncMock(return_value=1)
    redis.expire = AsyncMock(return_value=True)
    redis.aclose = AsyncMock()
    return redis


@pytest.fixture
def mock_cvm():
    """Create a mock ConversationModel."""
    cvm = Mock()
    cvm.insert = Mock()
    return cvm


@pytest.fixture
def mock_roster():
    """Create a mock Roster with a test persona."""
    roster = Mock()
    persona = Mock()
    persona.name = "Test Persona"
    roster.personas = {"test_persona": persona}
    return roster


@pytest.fixture
def mock_state_store():
    """Create a mock StateStore."""
    store = AsyncMock()
    store.get_step_status = AsyncMock(return_value=StepStatus.PENDING)
    store.get_state_type = AsyncMock(return_value="standard")
    store.load_state = AsyncMock()
    store.save_state = AsyncMock()
    store.set_step_status = AsyncMock()
    store.acquire_lock = AsyncMock(return_value=True)
    store.release_lock = AsyncMock()
    return store


@pytest.fixture
def mock_scheduler():
    """Create a mock Scheduler."""
    scheduler = AsyncMock()
    scheduler.pop_step_job = AsyncMock(return_value=None)
    scheduler.process_delayed_jobs = AsyncMock(return_value=0)
    scheduler.requeue_step = AsyncMock()
    scheduler.mark_complete = AsyncMock()
    scheduler.mark_failed = AsyncMock()
    scheduler.enqueue_step = AsyncMock()
    scheduler.all_deps_complete = AsyncMock(return_value=True)
    scheduler.check_pipeline_complete = AsyncMock()
    scheduler.queue_key = "queue:steps"
    scheduler.delayed_key = "queue:steps:delayed"
    return scheduler


class TestConversationAnalysisScenarios:
    """Test the CONVERSATION_ANALYSIS_SCENARIOS constant."""

    def test_expected_scenarios_present(self):
        """Test all expected scenarios are in the set."""
        expected = {"analysis_dialogue", "summarizer"}
        assert CONVERSATION_ANALYSIS_SCENARIOS == expected


class TestDreamRequest:
    """Test DreamRequest dataclass."""

    def test_create_with_all_fields(self):
        """Test creating DreamRequest with all fields."""
        request = DreamRequest(
            scenario="journaler_dialogue",
            query="What happened today?",
            guidance="Focus on emotions",
            triggered_by="manual",
        )
        assert request.scenario == "journaler_dialogue"
        assert request.query == "What happened today?"
        assert request.guidance == "Focus on emotions"
        assert request.triggered_by == "manual"

    def test_create_with_defaults(self):
        """Test creating DreamRequest with default values."""
        request = DreamRequest(scenario="analysis_dialogue")
        assert request.scenario == "analysis_dialogue"
        assert request.query is None
        assert request.guidance is None
        assert request.triggered_by == "manual"

    def test_auto_triggered_request(self):
        """Test creating auto-triggered request."""
        request = DreamRequest(
            scenario="journaler_dialogue",
            triggered_by="auto",
        )
        assert request.triggered_by == "auto"


class TestDreamResult:
    """Test DreamResult dataclass."""

    def test_create_success_result(self):
        """Test creating successful DreamResult."""
        result = DreamResult(
            success=True,
            pipeline_id="pipeline_123",
            scenario="journaler_dialogue",
            duration_seconds=45.2,
        )
        assert result.success is True
        assert result.pipeline_id == "pipeline_123"
        assert result.scenario == "journaler_dialogue"
        assert result.error is None
        assert result.duration_seconds == 45.2

    def test_create_failure_result(self):
        """Test creating failed DreamResult."""
        result = DreamResult(
            success=False,
            scenario="analysis_dialogue",
            error="Pipeline failed",
            duration_seconds=10.5,
        )
        assert result.success is False
        assert result.pipeline_id is None
        assert result.scenario == "analysis_dialogue"
        assert result.error == "Pipeline failed"
        assert result.duration_seconds == 10.5


class TestDreamerRunnerInit:
    """Test DreamerRunner initialization."""

    def test_initialization(self, chat_config, mock_cvm, mock_roster, mock_redis):
        """Test DreamerRunner initializes correctly."""
        with patch("andimud_worker.dreamer.runner.StateStore") as mock_store_class:
            with patch("andimud_worker.dreamer.runner.Scheduler") as mock_scheduler_class:
                mock_store_class.return_value = AsyncMock()
                mock_scheduler_class.return_value = AsyncMock()

                runner = DreamerRunner(
                    config=chat_config,
                    cvm=mock_cvm,
                    roster=mock_roster,
                    redis_client=mock_redis,
                    agent_id="test_agent",
                    persona_id="test_persona",
                )

                assert runner.config == chat_config
                assert runner.cvm == mock_cvm
                assert runner.roster == mock_roster
                assert runner.redis == mock_redis
                assert runner.agent_id == "test_agent"
                assert runner.persona_id == "test_persona"
                assert runner.state_store is not None
                assert runner.scheduler is not None

                # Verify StateStore was created with correct prefix
                mock_store_class.assert_called_once_with(
                    mock_redis,
                    key_prefix="mud:dreamer:test_agent"
                )

    def test_creates_agent_specific_keys(self, chat_config, mock_cvm, mock_roster, mock_redis):
        """Test that StateStore/Scheduler use agent-specific key prefix."""
        with patch("andimud_worker.dreamer.runner.StateStore") as mock_store_class:
            with patch("andimud_worker.dreamer.runner.Scheduler") as mock_scheduler_class:
                mock_store = AsyncMock()
                mock_scheduler = AsyncMock()
                mock_scheduler.queue_key = ""
                mock_scheduler.delayed_key = ""

                mock_store_class.return_value = mock_store
                mock_scheduler_class.return_value = mock_scheduler

                runner = DreamerRunner(
                    config=chat_config,
                    cvm=mock_cvm,
                    roster=mock_roster,
                    redis_client=mock_redis,
                    agent_id="andi",
                    persona_id="andi",
                )

                # Verify key prefix is set in __init__
                assert "mud:dreamer:andi" in runner.scheduler.queue_key
                assert "mud:dreamer:andi" in runner.scheduler.delayed_key


class TestDreamerRunnerGetConversationId:
    """Test DreamerRunner._get_conversation_id method."""

    def test_returns_base_id_for_analysis_scenarios(
        self, chat_config, mock_cvm, mock_roster, mock_redis
    ):
        """Test returns base_conversation_id for analysis_dialogue."""
        with patch("andimud_worker.dreamer.runner.StateStore"):
            with patch("andimud_worker.dreamer.runner.Scheduler"):
                runner = DreamerRunner(
                    config=chat_config,
                    cvm=mock_cvm,
                    roster=mock_roster,
                    redis_client=mock_redis,
                    agent_id="andi",
                    persona_id="andi",
                )

                conversation_id = runner._get_conversation_id(
                    "analysis_dialogue", "andimud_123_abc"
                )
                assert conversation_id == "andimud_123_abc"

    def test_returns_base_id_for_summarizer(
        self, chat_config, mock_cvm, mock_roster, mock_redis
    ):
        """Test returns base_conversation_id for summarizer."""
        with patch("andimud_worker.dreamer.runner.StateStore"):
            with patch("andimud_worker.dreamer.runner.Scheduler"):
                runner = DreamerRunner(
                    config=chat_config,
                    cvm=mock_cvm,
                    roster=mock_roster,
                    redis_client=mock_redis,
                    agent_id="andi",
                    persona_id="andi",
                )

                conversation_id = runner._get_conversation_id(
                    "summarizer", "andimud_123_abc"
                )
                assert conversation_id == "andimud_123_abc"

    def test_creates_standalone_id_for_journaler(
        self, chat_config, mock_cvm, mock_roster, mock_redis
    ):
        """Test creates standalone conversation ID for journaler_dialogue."""
        with patch("andimud_worker.dreamer.runner.StateStore"):
            with patch("andimud_worker.dreamer.runner.Scheduler"):
                runner = DreamerRunner(
                    config=chat_config,
                    cvm=mock_cvm,
                    roster=mock_roster,
                    redis_client=mock_redis,
                    agent_id="andi",
                    persona_id="andi",
                )

                conversation_id = runner._get_conversation_id(
                    "journaler_dialogue", "andimud_123_abc"
                )
                assert conversation_id == "mud_dream_andi_journaler_dialogue"
                assert "andimud_123_abc" not in conversation_id

    def test_creates_standalone_id_for_daydream(
        self, chat_config, mock_cvm, mock_roster, mock_redis
    ):
        """Test creates standalone conversation ID for daydream_dialogue."""
        with patch("andimud_worker.dreamer.runner.StateStore"):
            with patch("andimud_worker.dreamer.runner.Scheduler"):
                runner = DreamerRunner(
                    config=chat_config,
                    cvm=mock_cvm,
                    roster=mock_roster,
                    redis_client=mock_redis,
                    agent_id="val",
                    persona_id="val",
                )

                conversation_id = runner._get_conversation_id(
                    "daydream_dialogue", "andimud_456_def"
                )
                assert conversation_id == "mud_dream_val_daydream_dialogue"


class TestDreamerRunnerRunDream:
    """Test DreamerRunner.run_dream method."""

    @pytest.mark.asyncio
    async def test_successful_dream_execution(
        self, chat_config, mock_cvm, mock_roster, mock_redis
    ):
        """Test successful dream pipeline execution."""
        with patch("andimud_worker.dreamer.runner.StateStore"):
            with patch("andimud_worker.dreamer.runner.Scheduler"):
                runner = DreamerRunner(
                    config=chat_config,
                    cvm=mock_cvm,
                    roster=mock_roster,
                    redis_client=mock_redis,
                    agent_id="andi",
                    persona_id="andi",
                )

                # Mock start_pipeline to return a pipeline ID
                with patch("andimud_worker.dreamer.runner.start_pipeline") as mock_start:
                    mock_start.return_value = "pipeline_123"

                    # Mock _execute_pipeline to complete successfully
                    with patch.object(runner, "_execute_pipeline") as mock_execute:
                        mock_execute.return_value = None

                        request = DreamRequest(scenario="journaler_dialogue")
                        result = await runner.run_dream(request, "andimud_123_abc")

                assert result.success is True
                assert result.pipeline_id == "pipeline_123"
                assert result.scenario == "journaler_dialogue"
                assert result.error is None
                assert result.duration_seconds > 0

                # Verify start_pipeline was called correctly
                mock_start.assert_called_once()
                call_kwargs = mock_start.call_args[1]
                assert call_kwargs["scenario_name"] == "journaler_dialogue"
                assert call_kwargs["persona_id"] == "andi"
                assert call_kwargs["cvm"] == mock_cvm

    @pytest.mark.asyncio
    async def test_dream_execution_with_query_and_guidance(
        self, chat_config, mock_cvm, mock_roster, mock_redis
    ):
        """Test dream execution passes query and guidance to start_pipeline."""
        with patch("andimud_worker.dreamer.runner.StateStore"):
            with patch("andimud_worker.dreamer.runner.Scheduler"):
                runner = DreamerRunner(
                    config=chat_config,
                    cvm=mock_cvm,
                    roster=mock_roster,
                    redis_client=mock_redis,
                    agent_id="andi",
                    persona_id="andi",
                )

                with patch("andimud_worker.dreamer.runner.start_pipeline") as mock_start:
                    mock_start.return_value = "pipeline_456"

                    with patch.object(runner, "_execute_pipeline"):
                        request = DreamRequest(
                            scenario="journaler_dialogue",
                            query="What happened today?",
                            guidance="Focus on emotions",
                        )
                        await runner.run_dream(request, "andimud_123_abc")

                call_kwargs = mock_start.call_args[1]
                assert call_kwargs["query_text"] == "What happened today?"
                assert call_kwargs["guidance"] == "Focus on emotions"

    @pytest.mark.asyncio
    async def test_dream_execution_handles_pipeline_failure(
        self, chat_config, mock_cvm, mock_roster, mock_redis
    ):
        """Test dream execution handles pipeline failure gracefully."""
        with patch("andimud_worker.dreamer.runner.StateStore"):
            with patch("andimud_worker.dreamer.runner.Scheduler"):
                runner = DreamerRunner(
                    config=chat_config,
                    cvm=mock_cvm,
                    roster=mock_roster,
                    redis_client=mock_redis,
                    agent_id="andi",
                    persona_id="andi",
                )

                with patch("andimud_worker.dreamer.runner.start_pipeline") as mock_start:
                    mock_start.side_effect = Exception("Pipeline startup failed")

                    request = DreamRequest(scenario="analysis_dialogue")
                    result = await runner.run_dream(request, "andimud_123_abc")

                assert result.success is False
                assert result.pipeline_id is None
                assert result.scenario == "analysis_dialogue"
                assert "Pipeline startup failed" in result.error
                assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_dream_execution_invokes_heartbeat_callback(
        self, chat_config, mock_cvm, mock_roster, mock_redis
    ):
        """Test dream execution invokes heartbeat callback during execution."""
        with patch("andimud_worker.dreamer.runner.StateStore"):
            with patch("andimud_worker.dreamer.runner.Scheduler"):
                runner = DreamerRunner(
                    config=chat_config,
                    cvm=mock_cvm,
                    roster=mock_roster,
                    redis_client=mock_redis,
                    agent_id="andi",
                    persona_id="andi",
                )

                heartbeat_calls = []

                async def heartbeat_callback():
                    heartbeat_calls.append(datetime.now(timezone.utc))

                with patch("andimud_worker.dreamer.runner.start_pipeline") as mock_start:
                    mock_start.return_value = "pipeline_789"

                    with patch.object(runner, "_execute_pipeline") as mock_execute:
                        # Simulate heartbeat being called during execution
                        async def execute_with_heartbeat(pipeline_id, callback):
                            if callback:
                                await callback()

                        mock_execute.side_effect = execute_with_heartbeat

                        request = DreamRequest(scenario="journaler_dialogue")
                        await runner.run_dream(
                            request, "andimud_123_abc", heartbeat_callback=heartbeat_callback
                        )

                # Verify heartbeat was called
                assert len(heartbeat_calls) > 0

    @pytest.mark.asyncio
    async def test_uses_base_conversation_id_for_analysis(
        self, chat_config, mock_cvm, mock_roster, mock_redis
    ):
        """Test analysis_dialogue uses base conversation_id."""
        with patch("andimud_worker.dreamer.runner.StateStore"):
            with patch("andimud_worker.dreamer.runner.Scheduler"):
                runner = DreamerRunner(
                    config=chat_config,
                    cvm=mock_cvm,
                    roster=mock_roster,
                    redis_client=mock_redis,
                    agent_id="andi",
                    persona_id="andi",
                )

                with patch("andimud_worker.dreamer.runner.start_pipeline") as mock_start:
                    mock_start.return_value = "pipeline_abc"

                    with patch.object(runner, "_execute_pipeline"):
                        request = DreamRequest(scenario="analysis_dialogue")
                        await runner.run_dream(request, "andimud_123_abc")

                call_kwargs = mock_start.call_args[1]
                assert call_kwargs["conversation_id"] == "andimud_123_abc"

    @pytest.mark.asyncio
    async def test_uses_standalone_conversation_id_for_journaler(
        self, chat_config, mock_cvm, mock_roster, mock_redis
    ):
        """Test journaler_dialogue uses standalone conversation_id."""
        with patch("andimud_worker.dreamer.runner.StateStore"):
            with patch("andimud_worker.dreamer.runner.Scheduler"):
                runner = DreamerRunner(
                    config=chat_config,
                    cvm=mock_cvm,
                    roster=mock_roster,
                    redis_client=mock_redis,
                    agent_id="andi",
                    persona_id="andi",
                )

                with patch("andimud_worker.dreamer.runner.start_pipeline") as mock_start:
                    mock_start.return_value = "pipeline_def"

                    with patch.object(runner, "_execute_pipeline"):
                        request = DreamRequest(scenario="journaler_dialogue")
                        await runner.run_dream(request, "andimud_123_abc")

                call_kwargs = mock_start.call_args[1]
                assert call_kwargs["conversation_id"] == "mud_dream_andi_journaler_dialogue"


class TestDreamerMixinInit:
    """Test DreamerMixin._init_dreamer method."""

    @pytest.mark.asyncio
    async def test_init_dreamer_creates_runner(self, mud_config, mock_redis, chat_config):
        """Test _init_dreamer creates DreamerRunner correctly."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        # Setup required attributes
        worker.chat_config = chat_config
        worker.cvm = Mock()
        worker.roster = Mock()

        with patch("andimud_worker.mixins.dreamer.DreamerRunner") as mock_runner_class:
            mock_runner_class.return_value = Mock()
            worker._init_dreamer()

            assert worker._dreamer_runner is not None
            mock_runner_class.assert_called_once_with(
                config=chat_config,
                cvm=worker.cvm,
                roster=worker.roster,
                redis_client=mock_redis,
                agent_id="test_agent",
                persona_id="test_persona",
            )


class TestDreamerMixinProcessDreamTurn:
    """Test DreamerMixin.process_dream_turn method."""

    @pytest.mark.asyncio
    async def test_process_dream_turn_initializes_runner_if_needed(
        self, mud_config, mock_redis, chat_config
    ):
        """Test process_dream_turn initializes runner on first call."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.chat_config = chat_config
        worker.cvm = Mock()
        worker.roster = Mock()
        worker.conversation_manager = Mock()
        worker.conversation_manager.conversation_id = "andimud_123_abc"

        assert worker._dreamer_runner is None

        with patch("andimud_worker.mixins.dreamer.DreamerRunner") as mock_runner_class:
            mock_runner = Mock()
            mock_runner.run_dream = AsyncMock(return_value=DreamResult(
                success=True,
                pipeline_id="pipeline_123",
                scenario="journaler_dialogue",
            ))
            mock_runner_class.return_value = mock_runner

            await worker.process_dream_turn(scenario="journaler_dialogue")

            assert worker._dreamer_runner is not None

    @pytest.mark.asyncio
    async def test_process_dream_turn_calls_runner_with_correct_params(
        self, mud_config, mock_redis, chat_config
    ):
        """Test process_dream_turn calls DreamerRunner.run_dream with correct params."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.chat_config = chat_config
        worker.cvm = Mock()
        worker.roster = Mock()
        worker.conversation_manager = Mock()
        worker.conversation_manager.conversation_id = "andimud_123_abc"

        with patch("andimud_worker.mixins.dreamer.DreamerRunner") as mock_runner_class:
            mock_runner = Mock()
            mock_runner.run_dream = AsyncMock(return_value=DreamResult(
                success=True,
                pipeline_id="pipeline_456",
                scenario="journaler_dialogue",
            ))
            mock_runner_class.return_value = mock_runner

            worker._init_dreamer()

            result = await worker.process_dream_turn(
                scenario="journaler_dialogue",
                query="What happened?",
                guidance="Focus on emotions",
                triggered_by="manual",
            )

            mock_runner.run_dream.assert_called_once()
            call_args = mock_runner.run_dream.call_args[0]

            # Check request parameter
            request = call_args[0]
            assert request.scenario == "journaler_dialogue"
            assert request.query == "What happened?"
            assert request.guidance == "Focus on emotions"
            assert request.triggered_by == "manual"

            # Check conversation_id parameter
            assert call_args[1] == "andimud_123_abc"

            # Check heartbeat callback is provided
            assert callable(call_args[2])

    @pytest.mark.asyncio
    async def test_process_dream_turn_updates_state_on_success(
        self, mud_config, mock_redis, chat_config
    ):
        """Test process_dream_turn updates dreamer state on success."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.chat_config = chat_config
        worker.cvm = Mock()
        worker.roster = Mock()
        worker.conversation_manager = Mock()
        worker.conversation_manager.conversation_id = "andimud_123_abc"

        with patch("andimud_worker.mixins.dreamer.DreamerRunner") as mock_runner_class:
            mock_runner = Mock()
            mock_runner.run_dream = AsyncMock(return_value=DreamResult(
                success=True,
                pipeline_id="pipeline_789",
                scenario="analysis_dialogue",
            ))
            mock_runner_class.return_value = mock_runner

            worker._init_dreamer()

            await worker.process_dream_turn(scenario="analysis_dialogue")

            # Verify state was updated
            mock_redis.hset.assert_called()
            call_args = mock_redis.hset.call_args
            assert call_args[0][0] == RedisKeys.agent_dreamer("test_agent")
            mapping = call_args[1]["mapping"]
            assert "last_dream_at" in mapping
            assert mapping["last_dream_scenario"] == "analysis_dialogue"
            assert mapping["pending_pipeline_id"] == ""

    @pytest.mark.asyncio
    async def test_process_dream_turn_heartbeat_refreshes_ttl(
        self, mud_config, mock_redis, chat_config
    ):
        """Test process_dream_turn heartbeat callback refreshes turn request TTL."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.chat_config = chat_config
        worker.cvm = Mock()
        worker.roster = Mock()
        worker.conversation_manager = Mock()
        worker.conversation_manager.conversation_id = "andimud_123_abc"

        heartbeat_callback = None

        async def capture_heartbeat(request, conversation_id, callback):
            nonlocal heartbeat_callback
            heartbeat_callback = callback
            # Call the heartbeat to test it
            if callback:
                await callback()
            return DreamResult(
                success=True,
                pipeline_id="pipeline_abc",
                scenario="journaler_dialogue",
            )

        with patch("andimud_worker.mixins.dreamer.DreamerRunner") as mock_runner_class:
            mock_runner = Mock()
            mock_runner.run_dream = AsyncMock(side_effect=capture_heartbeat)
            mock_runner_class.return_value = mock_runner

            worker._init_dreamer()

            await worker.process_dream_turn(scenario="journaler_dialogue")

            # Verify heartbeat callback was provided and called
            assert heartbeat_callback is not None

            # Verify Redis expire was called
            mock_redis.expire.assert_called()
            expire_call = mock_redis.expire.call_args
            assert "turn_request" in expire_call[0][0]
            assert expire_call[0][1] == mud_config.turn_request_ttl_seconds


class TestDreamerMixinUpdateDreamerState:
    """Test DreamerMixin._update_dreamer_state method."""

    @pytest.mark.asyncio
    async def test_update_state_on_success(self, mud_config, mock_redis):
        """Test _update_dreamer_state updates correctly on success."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        result = DreamResult(
            success=True,
            pipeline_id="pipeline_123",
            scenario="journaler_dialogue",
            duration_seconds=45.2,
        )

        await worker._update_dreamer_state(result)

        mock_redis.hset.assert_called_once()
        call_args = mock_redis.hset.call_args
        assert call_args[0][0] == RedisKeys.agent_dreamer("test_agent")
        mapping = call_args[1]["mapping"]
        assert "last_dream_at" in mapping
        assert mapping["last_dream_scenario"] == "journaler_dialogue"
        assert mapping["pending_pipeline_id"] == ""

    @pytest.mark.asyncio
    async def test_update_state_on_failure(self, mud_config, mock_redis):
        """Test _update_dreamer_state clears pending_pipeline_id on failure."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        result = DreamResult(
            success=False,
            scenario="analysis_dialogue",
            error="Pipeline failed",
            duration_seconds=10.5,
        )

        await worker._update_dreamer_state(result)

        mock_redis.hset.assert_called_once()
        call_args = mock_redis.hset.call_args
        mapping = call_args[1]["mapping"]
        # On failure, only pending_pipeline_id is cleared
        assert mapping["pending_pipeline_id"] == ""
        assert "last_dream_at" not in mapping
        assert "last_dream_scenario" not in mapping


class TestDreamerMixinCheckAutoTriggers:
    """Test DreamerMixin.check_auto_dream_triggers method."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_state(self, mud_config, mock_redis):
        """Test returns None when dreamer state doesn't exist."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        mock_redis.hgetall = AsyncMock(return_value={})

        result = await worker.check_auto_dream_triggers()

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self, mud_config, mock_redis):
        """Test returns None when dreamer is disabled."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        mock_redis.hgetall = AsyncMock(return_value={
            b"enabled": b"false",
            b"idle_threshold_seconds": b"3600",
            b"token_threshold": b"10000",
        })

        result = await worker.check_auto_dream_triggers()

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_idle_threshold_not_met(
        self, mud_config, mock_redis
    ):
        """Test returns None when not enough time has passed since last dream."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        # Set last_dream_at to very recent (1 minute ago)
        recent_time = datetime.now(timezone.utc)
        mock_redis.hgetall = AsyncMock(return_value={
            b"enabled": b"true",
            b"idle_threshold_seconds": b"3600",  # 1 hour required
            b"last_dream_at": recent_time.isoformat().encode(),
            b"token_threshold": b"10000",
        })

        result = await worker.check_auto_dream_triggers()

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_token_threshold_not_met(
        self, mud_config, mock_redis
    ):
        """Test returns None when token threshold not met."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.conversation_manager = Mock()
        worker.conversation_manager.get_total_tokens = AsyncMock(return_value=5000)

        # Set last_dream_at to long ago (2 hours)
        old_time = datetime.now(timezone.utc).replace(hour=datetime.now(timezone.utc).hour - 2)
        mock_redis.hgetall = AsyncMock(return_value={
            b"enabled": b"true",
            b"idle_threshold_seconds": b"3600",  # 1 hour (passed)
            b"last_dream_at": old_time.isoformat().encode(),
            b"token_threshold": b"10000",  # Need 10000 tokens
        })

        result = await worker.check_auto_dream_triggers()

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_request_when_triggers_met(
        self, mud_config, mock_redis
    ):
        """Test returns DreamRequest when all triggers are met."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.conversation_manager = Mock()
        worker.conversation_manager.get_total_tokens = AsyncMock(return_value=15000)

        # Set last_dream_at to long ago (2 hours)
        old_time = datetime.now(timezone.utc).replace(hour=datetime.now(timezone.utc).hour - 2)
        mock_redis.hgetall = AsyncMock(return_value={
            b"enabled": b"true",
            b"idle_threshold_seconds": b"3600",  # 1 hour (passed)
            b"last_dream_at": old_time.isoformat().encode(),
            b"token_threshold": b"10000",  # 10000 tokens (passed)
        })

        result = await worker.check_auto_dream_triggers()

        assert result is not None
        assert isinstance(result, DreamRequest)
        assert result.triggered_by == "auto"

    @pytest.mark.asyncio
    async def test_triggers_on_first_run_with_no_last_dream(
        self, mud_config, mock_redis
    ):
        """Test triggers when last_dream_at is not set (first run)."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.conversation_manager = Mock()
        worker.conversation_manager.get_total_tokens = AsyncMock(return_value=15000)

        mock_redis.hgetall = AsyncMock(return_value={
            b"enabled": b"true",
            b"idle_threshold_seconds": b"3600",
            b"token_threshold": b"10000",
            # No last_dream_at - first run
        })

        result = await worker.check_auto_dream_triggers()

        assert result is not None
        assert result.triggered_by == "auto"


class TestDreamerMixinSelectAutoScenario:
    """Test DreamerMixin._select_auto_dream_scenario method."""

    @pytest.mark.asyncio
    async def test_selects_analysis_for_high_token_count(
        self, mud_config, mock_redis
    ):
        """Test selects analysis_dialogue for high token count."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.conversation_manager = Mock()
        worker.conversation_manager.get_total_tokens = AsyncMock(return_value=25000)

        scenario = await worker._select_auto_dream_scenario()

        assert scenario == "analysis_dialogue"

    @pytest.mark.asyncio
    async def test_selects_journaler_for_low_token_count(
        self, mud_config, mock_redis
    ):
        """Test selects journaler_dialogue for low token count."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.conversation_manager = Mock()
        worker.conversation_manager.get_total_tokens = AsyncMock(return_value=15000)

        scenario = await worker._select_auto_dream_scenario()

        assert scenario == "journaler_dialogue"

    @pytest.mark.asyncio
    async def test_selects_journaler_when_no_conversation_manager(
        self, mud_config, mock_redis
    ):
        """Test selects journaler_dialogue when conversation_manager is None."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.conversation_manager = None

        scenario = await worker._select_auto_dream_scenario()

        assert scenario == "journaler_dialogue"

    @pytest.mark.asyncio
    async def test_threshold_exactly_at_20000(
        self, mud_config, mock_redis
    ):
        """Test behavior at exactly 20000 tokens (boundary condition)."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.conversation_manager = Mock()
        worker.conversation_manager.get_total_tokens = AsyncMock(return_value=20000)

        scenario = await worker._select_auto_dream_scenario()

        # At exactly 20000, should select journaler (not > 20000)
        assert scenario == "journaler_dialogue"

    @pytest.mark.asyncio
    async def test_threshold_just_over_20000(
        self, mud_config, mock_redis
    ):
        """Test behavior at 20001 tokens (just over threshold)."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.conversation_manager = Mock()
        worker.conversation_manager.get_total_tokens = AsyncMock(return_value=20001)

        scenario = await worker._select_auto_dream_scenario()

        # Just over 20000 should select analysis
        assert scenario == "analysis_dialogue"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
