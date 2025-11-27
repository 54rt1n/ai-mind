# tests/unit/dreamer/test_worker.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for aim/dreamer/worker.py"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio

from aim.dreamer.worker import DreamerWorker, run_worker
from aim.dreamer.executor import RetryableError
from aim.dreamer.models import (
    StepJob,
    PipelineState,
    StepDefinition,
    StepConfig,
    StepOutput,
    StepResult,
    Scenario,
    ScenarioContext,
)
from aim.config import ChatConfig


class TestDreamerWorkerInit:
    """Test DreamerWorker initialization."""

    def test_init_sets_attributes(self):
        """Test that __init__ properly sets all attributes."""
        mock_config = Mock(spec=ChatConfig)
        mock_state_store = Mock()
        mock_scheduler = Mock()

        worker = DreamerWorker(
            config=mock_config,
            state_store=mock_state_store,
            scheduler=mock_scheduler,
        )

        assert worker.config == mock_config
        assert worker.state_store == mock_state_store
        assert worker.scheduler == mock_scheduler
        assert worker.running is False
        assert worker.cvm is None
        assert worker.roster is None
        assert worker.encoder is None


class TestDreamerWorkerStop:
    """Test DreamerWorker stop method."""

    @pytest.mark.asyncio
    async def test_stop_sets_running_to_false(self):
        """Test that stop() sets running flag to False."""
        mock_config = Mock(spec=ChatConfig)
        mock_state_store = Mock()
        mock_scheduler = Mock()

        worker = DreamerWorker(
            config=mock_config,
            state_store=mock_state_store,
            scheduler=mock_scheduler,
        )

        # Set running to True to simulate a running worker
        worker.running = True

        await worker.stop()

        assert worker.running is False


class TestDreamerWorkerProcessJob:
    """Test DreamerWorker process_job method."""

    @pytest.mark.asyncio
    async def test_process_job_skip_when_lock_not_acquired(self):
        """Test that process_job skips when lock cannot be acquired."""
        mock_config = Mock(spec=ChatConfig)
        mock_state_store = AsyncMock()
        mock_scheduler = AsyncMock()

        # Lock acquisition fails
        mock_state_store.acquire_lock = AsyncMock(return_value=False)

        worker = DreamerWorker(
            config=mock_config,
            state_store=mock_state_store,
            scheduler=mock_scheduler,
        )

        job = StepJob(
            pipeline_id="test-123",
            step_id="step-1",
            attempt=1,
            max_attempts=3,
            enqueued_at=datetime.utcnow(),
            priority=0,
        )

        await worker.process_job(job)

        # Verify lock was attempted but state not loaded
        mock_state_store.acquire_lock.assert_called_once_with("test-123", "step-1", ttl=300)
        mock_state_store.load_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_job_skip_when_state_not_found(self):
        """Test that process_job skips when pipeline state not found."""
        mock_config = Mock(spec=ChatConfig)
        mock_state_store = AsyncMock()
        mock_scheduler = AsyncMock()

        # Lock acquisition succeeds but state not found
        mock_state_store.acquire_lock = AsyncMock(return_value=True)
        mock_state_store.load_state = AsyncMock(return_value=None)
        mock_state_store.release_lock = AsyncMock()

        worker = DreamerWorker(
            config=mock_config,
            state_store=mock_state_store,
            scheduler=mock_scheduler,
        )

        job = StepJob(
            pipeline_id="test-123",
            step_id="step-1",
            attempt=1,
            max_attempts=3,
            enqueued_at=datetime.utcnow(),
            priority=0,
        )

        await worker.process_job(job)

        # Verify state was loaded but scenario not loaded
        mock_state_store.load_state.assert_called_once_with("test-123")
        # Lock should be released in finally
        mock_state_store.release_lock.assert_called_once_with("test-123", "step-1")

    @pytest.mark.asyncio
    async def test_process_job_happy_path(self):
        """Test successful job processing through the full pipeline."""
        mock_config = Mock(spec=ChatConfig)
        mock_state_store = AsyncMock()
        mock_scheduler = AsyncMock()

        # Setup state
        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test_scenario",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="gpt-4",
            branch=0,
            step_counter=1,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        # Setup scenario and step
        scenario = Scenario(
            name="test_scenario",
            description="Test scenario",
            context=ScenarioContext(required_aspects=[]),
            steps={
                "step-1": StepDefinition(
                    id="step-1",
                    prompt="Test prompt",
                    config=StepConfig(),
                    output=StepOutput(document_type="test", weight=1.0),
                    next=["step-2"],
                ),
                "step-2": StepDefinition(
                    id="step-2",
                    prompt="Next step",
                    config=StepConfig(),
                    output=StepOutput(document_type="test", weight=1.0),
                    depends_on=["step-1"],
                    next=[],
                ),
            },
        )

        # Setup mocks
        mock_state_store.acquire_lock = AsyncMock(return_value=True)
        mock_state_store.load_state = AsyncMock(return_value=state)
        mock_state_store.save_state = AsyncMock()
        mock_state_store.release_lock = AsyncMock()

        mock_scheduler.all_deps_complete = AsyncMock(return_value=True)
        mock_scheduler.mark_complete = AsyncMock()
        mock_scheduler.enqueue_step = AsyncMock()
        mock_scheduler.check_pipeline_complete = AsyncMock()

        # Setup worker with mocked resources
        worker = DreamerWorker(
            config=mock_config,
            state_store=mock_state_store,
            scheduler=mock_scheduler,
        )

        mock_cvm = Mock()
        mock_cvm.insert = Mock()
        worker.cvm = mock_cvm

        mock_roster = Mock()
        mock_persona = Mock()
        mock_roster.personas = {"assistant": mock_persona}
        worker.roster = mock_roster

        import tiktoken
        worker.encoder = tiktoken.get_encoding("cl100k_base")

        # Mock execute_step and create_message
        step_result = StepResult(
            step_id="step-1",
            response="Test response",
            think=None,
            doc_id="doc-123",
            document_type="test",
            document_weight=1.0,
            tokens_used=50,
            timestamp=datetime.utcnow(),
        )

        mock_message = Mock()

        with patch("aim.dreamer.worker.load_scenario") as mock_load_scenario, \
             patch("aim.dreamer.worker.execute_step") as mock_execute_step, \
             patch("aim.dreamer.worker.create_message") as mock_create_message:

            mock_load_scenario.return_value = scenario
            # execute_step returns tuple: (StepResult, context_doc_ids, is_initial_context)
            mock_execute_step.return_value = (step_result, [], False)
            mock_create_message.return_value = mock_message

            job = StepJob(
                pipeline_id="test-123",
                step_id="step-1",
                attempt=1,
                max_attempts=3,
                enqueued_at=datetime.utcnow(),
                priority=0,
            )

            await worker.process_job(job)

        # Verify the full flow
        mock_state_store.acquire_lock.assert_called_once()
        mock_state_store.load_state.assert_called_once_with("test-123")
        mock_load_scenario.assert_called_once_with("test_scenario")
        mock_scheduler.all_deps_complete.assert_called()
        mock_execute_step.assert_called_once()
        mock_cvm.insert.assert_called_once_with(mock_message)
        mock_state_store.save_state.assert_called_once()
        mock_scheduler.mark_complete.assert_called_once_with("test-123", "step-1")
        mock_scheduler.enqueue_step.assert_called()
        mock_state_store.release_lock.assert_called_once_with("test-123", "step-1")

    @pytest.mark.asyncio
    async def test_process_job_requeue_when_deps_not_complete(self):
        """Test that job is requeued when dependencies not complete."""
        mock_config = Mock(spec=ChatConfig)
        mock_state_store = AsyncMock()
        mock_scheduler = AsyncMock()

        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test_scenario",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="gpt-4",
            branch=0,
            step_counter=1,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        scenario = Scenario(
            name="test_scenario",
            description="Test scenario",
            context=ScenarioContext(required_aspects=[]),
            steps={
                "step-1": StepDefinition(
                    id="step-1",
                    prompt="Test prompt",
                    config=StepConfig(),
                    output=StepOutput(document_type="test", weight=1.0),
                    depends_on=["step-0"],
                ),
            },
        )

        mock_state_store.acquire_lock = AsyncMock(return_value=True)
        mock_state_store.load_state = AsyncMock(return_value=state)
        mock_state_store.release_lock = AsyncMock()

        # Dependencies not complete
        mock_scheduler.all_deps_complete = AsyncMock(return_value=False)
        mock_scheduler.requeue_step = AsyncMock()

        worker = DreamerWorker(
            config=mock_config,
            state_store=mock_state_store,
            scheduler=mock_scheduler,
        )

        with patch("aim.dreamer.worker.load_scenario") as mock_load_scenario:
            mock_load_scenario.return_value = scenario

            job = StepJob(
                pipeline_id="test-123",
                step_id="step-1",
                attempt=1,
                max_attempts=3,
                enqueued_at=datetime.utcnow(),
                priority=0,
            )

            await worker.process_job(job)

        # Verify job was requeued
        mock_scheduler.all_deps_complete.assert_called_once()
        mock_scheduler.requeue_step.assert_called_once()
        call_args = mock_scheduler.requeue_step.call_args
        assert call_args[0][0] == job  # First positional arg is the job
        assert call_args[1]["delay"] == 5  # Keyword arg delay=5

    @pytest.mark.asyncio
    async def test_process_job_with_retryable_error(self):
        """Test that RetryableError causes job to be requeued with backoff."""
        mock_config = Mock(spec=ChatConfig)
        mock_state_store = AsyncMock()
        mock_scheduler = AsyncMock()

        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test_scenario",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="gpt-4",
            branch=0,
            step_counter=1,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        scenario = Scenario(
            name="test_scenario",
            description="Test scenario",
            context=ScenarioContext(required_aspects=[]),
            steps={
                "step-1": StepDefinition(
                    id="step-1",
                    prompt="Test prompt",
                    config=StepConfig(),
                    output=StepOutput(document_type="test", weight=1.0),
                ),
            },
        )

        mock_state_store.acquire_lock = AsyncMock(return_value=True)
        mock_state_store.load_state = AsyncMock(return_value=state)
        mock_state_store.release_lock = AsyncMock()

        mock_scheduler.all_deps_complete = AsyncMock(return_value=True)
        mock_scheduler.requeue_step = AsyncMock()

        worker = DreamerWorker(
            config=mock_config,
            state_store=mock_state_store,
            scheduler=mock_scheduler,
        )

        mock_roster = Mock()
        mock_persona = Mock()
        mock_roster.personas = {"assistant": mock_persona}
        worker.roster = mock_roster
        worker.cvm = Mock()

        import tiktoken
        worker.encoder = tiktoken.get_encoding("cl100k_base")

        with patch("aim.dreamer.worker.load_scenario") as mock_load_scenario, \
             patch("aim.dreamer.worker.execute_step") as mock_execute_step:

            mock_load_scenario.return_value = scenario
            # Raise RetryableError
            mock_execute_step.side_effect = RetryableError("Model unavailable")

            job = StepJob(
                pipeline_id="test-123",
                step_id="step-1",
                attempt=2,  # Second attempt
                max_attempts=3,
                enqueued_at=datetime.utcnow(),
                priority=0,
            )

            await worker.process_job(job)

        # Verify job was requeued with exponential backoff
        mock_scheduler.requeue_step.assert_called_once()
        call_args = mock_scheduler.requeue_step.call_args
        requeued_job = call_args[0][0]
        assert requeued_job.attempt == 3  # Incremented
        assert call_args[1]["delay"] == 60  # 30 * attempt (30 * 2)
        mock_state_store.release_lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_job_mark_failed_on_max_retries(self):
        """Test that job is marked failed after max retries exceeded."""
        mock_config = Mock(spec=ChatConfig)
        mock_state_store = AsyncMock()
        mock_scheduler = AsyncMock()

        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test_scenario",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="gpt-4",
            branch=0,
            step_counter=1,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        scenario = Scenario(
            name="test_scenario",
            description="Test scenario",
            context=ScenarioContext(required_aspects=[]),
            steps={
                "step-1": StepDefinition(
                    id="step-1",
                    prompt="Test prompt",
                    config=StepConfig(),
                    output=StepOutput(document_type="test", weight=1.0),
                ),
            },
        )

        mock_state_store.acquire_lock = AsyncMock(return_value=True)
        mock_state_store.load_state = AsyncMock(return_value=state)
        mock_state_store.release_lock = AsyncMock()

        mock_scheduler.all_deps_complete = AsyncMock(return_value=True)
        mock_scheduler.mark_failed = AsyncMock()

        worker = DreamerWorker(
            config=mock_config,
            state_store=mock_state_store,
            scheduler=mock_scheduler,
        )

        mock_roster = Mock()
        mock_persona = Mock()
        mock_roster.personas = {"assistant": mock_persona}
        worker.roster = mock_roster
        worker.cvm = Mock()

        import tiktoken
        worker.encoder = tiktoken.get_encoding("cl100k_base")

        with patch("aim.dreamer.worker.load_scenario") as mock_load_scenario, \
             patch("aim.dreamer.worker.execute_step") as mock_execute_step:

            mock_load_scenario.return_value = scenario
            mock_execute_step.side_effect = RetryableError("Model unavailable")

            job = StepJob(
                pipeline_id="test-123",
                step_id="step-1",
                attempt=3,  # Max attempts reached
                max_attempts=3,
                enqueued_at=datetime.utcnow(),
                priority=0,
            )

            await worker.process_job(job)

        # Verify job was marked failed
        mock_scheduler.mark_failed.assert_called_once()
        call_args = mock_scheduler.mark_failed.call_args
        assert call_args[0][0] == "test-123"
        assert call_args[0][1] == "step-1"
        assert "Max retries exceeded" in call_args[0][2]

    @pytest.mark.asyncio
    async def test_process_job_mark_failed_on_non_retryable_error(self):
        """Test that non-retryable errors immediately mark job as failed."""
        mock_config = Mock(spec=ChatConfig)
        mock_state_store = AsyncMock()
        mock_scheduler = AsyncMock()

        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test_scenario",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="gpt-4",
            branch=0,
            step_counter=1,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        scenario = Scenario(
            name="test_scenario",
            description="Test scenario",
            context=ScenarioContext(required_aspects=[]),
            steps={
                "step-1": StepDefinition(
                    id="step-1",
                    prompt="Test prompt",
                    config=StepConfig(),
                    output=StepOutput(document_type="test", weight=1.0),
                ),
            },
        )

        mock_state_store.acquire_lock = AsyncMock(return_value=True)
        mock_state_store.load_state = AsyncMock(return_value=state)
        mock_state_store.release_lock = AsyncMock()

        mock_scheduler.all_deps_complete = AsyncMock(return_value=True)
        mock_scheduler.mark_failed = AsyncMock()

        worker = DreamerWorker(
            config=mock_config,
            state_store=mock_state_store,
            scheduler=mock_scheduler,
        )

        mock_roster = Mock()
        mock_persona = Mock()
        mock_roster.personas = {"assistant": mock_persona}
        worker.roster = mock_roster
        worker.cvm = Mock()

        import tiktoken
        worker.encoder = tiktoken.get_encoding("cl100k_base")

        with patch("aim.dreamer.worker.load_scenario") as mock_load_scenario, \
             patch("aim.dreamer.worker.execute_step") as mock_execute_step:

            mock_load_scenario.return_value = scenario
            # Raise a non-retryable error
            mock_execute_step.side_effect = ValueError("Invalid configuration")

            job = StepJob(
                pipeline_id="test-123",
                step_id="step-1",
                attempt=1,
                max_attempts=3,
                enqueued_at=datetime.utcnow(),
                priority=0,
            )

            await worker.process_job(job)

        # Verify job was marked failed immediately
        mock_scheduler.mark_failed.assert_called_once()
        call_args = mock_scheduler.mark_failed.call_args
        assert call_args[0][0] == "test-123"
        assert call_args[0][1] == "step-1"
        assert "Execution error" in call_args[0][2]

    @pytest.mark.asyncio
    async def test_process_job_releases_lock_on_error(self):
        """Test that lock is always released even when error occurs."""
        mock_config = Mock(spec=ChatConfig)
        mock_state_store = AsyncMock()
        mock_scheduler = AsyncMock()

        mock_state_store.acquire_lock = AsyncMock(return_value=True)
        # Simulate error during load_state
        mock_state_store.load_state = AsyncMock(side_effect=Exception("Redis error"))
        mock_state_store.release_lock = AsyncMock()

        worker = DreamerWorker(
            config=mock_config,
            state_store=mock_state_store,
            scheduler=mock_scheduler,
        )

        job = StepJob(
            pipeline_id="test-123",
            step_id="step-1",
            attempt=1,
            max_attempts=3,
            enqueued_at=datetime.utcnow(),
            priority=0,
        )

        # Should not raise, error should be handled
        await worker.process_job(job)

        # Verify lock was released despite error
        mock_state_store.release_lock.assert_called_once_with("test-123", "step-1")


class TestDreamerWorkerStart:
    """Test DreamerWorker start method."""

    @pytest.mark.asyncio
    async def test_start_initializes_resources(self):
        """Test that start() initializes CVM, Roster, and encoder."""
        mock_config = Mock(spec=ChatConfig)
        mock_state_store = AsyncMock()
        mock_scheduler = AsyncMock()

        # Mock pop_step_job to return None (timeout) then raise to break loop
        call_count = [0]

        async def pop_side_effect(timeout=0):
            call_count[0] += 1
            if call_count[0] >= 2:
                # Stop worker after one iteration
                worker.running = False
            return None

        mock_scheduler.pop_step_job = AsyncMock(side_effect=pop_side_effect)
        mock_scheduler.process_delayed_jobs = AsyncMock(return_value=0)

        worker = DreamerWorker(
            config=mock_config,
            state_store=mock_state_store,
            scheduler=mock_scheduler,
        )

        with patch("aim.dreamer.worker.ConversationModel") as mock_cvm_class, \
             patch("aim.dreamer.worker.Roster") as mock_roster_class, \
             patch("aim.dreamer.worker.tiktoken") as mock_tiktoken:

            mock_cvm = Mock()
            mock_cvm_class.from_config.return_value = mock_cvm

            mock_roster = Mock()
            mock_roster_class.from_config.return_value = mock_roster

            mock_encoder = Mock()
            mock_tiktoken.get_encoding.return_value = mock_encoder

            await worker.start()

            # Verify resources were initialized
            assert worker.cvm == mock_cvm
            assert worker.roster == mock_roster
            assert worker.encoder == mock_encoder
            assert worker.running is False  # Stopped by our mock


class TestRunWorker:
    """Test run_worker entry point."""

    @pytest.mark.asyncio
    async def test_run_worker_creates_worker_and_starts(self):
        """Test that run_worker creates Redis client and starts worker."""
        mock_config = Mock(spec=ChatConfig)
        mock_config.redis_host = "localhost"
        mock_config.redis_port = 6379
        mock_config.redis_db = 0
        mock_config.redis_password = None

        # Patch redis.asyncio which is imported locally in run_worker
        with patch("redis.asyncio.Redis") as mock_redis_class, \
             patch("aim.dreamer.worker.StateStore") as mock_state_store_class, \
             patch("aim.dreamer.worker.Scheduler") as mock_scheduler_class, \
             patch("aim.dreamer.worker.DreamerWorker") as mock_worker_class:

            mock_redis_client = AsyncMock()
            mock_redis_class.return_value = mock_redis_client

            mock_state_store = Mock()
            mock_state_store_class.return_value = mock_state_store

            mock_scheduler = Mock()
            mock_scheduler_class.return_value = mock_scheduler

            mock_worker = AsyncMock()
            mock_worker.start = AsyncMock()
            mock_worker_class.return_value = mock_worker

            await run_worker(mock_config)

            # Verify Redis client was created
            mock_redis_class.assert_called_once_with(
                host="localhost",
                port=6379,
                db=0,
                password=None,
                decode_responses=False,
            )

            # Verify StateStore and Scheduler were created
            mock_state_store_class.assert_called_once_with(mock_redis_client)
            mock_scheduler_class.assert_called_once_with(mock_redis_client, mock_state_store)

            # Verify worker was created and started
            mock_worker_class.assert_called_once_with(mock_config, mock_state_store, mock_scheduler)
            mock_worker.start.assert_called_once()

            # Verify Redis client was closed
            mock_redis_client.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
