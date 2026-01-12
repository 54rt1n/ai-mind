# packages/aim-mud/tests/mud_tests/unit/worker/test_dreamer.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for DreamerRunner and DreamerMixin.

NOTE: DreamerRunner was simplified on 2026-01-09 to use the inline scheduler
instead of distributed infrastructure (StateStore, Scheduler, start_pipeline).
Many integration tests below need updating to work with the new inline approach.

The inline scheduler executes pipelines synchronously in-process without Redis
queues or state stores. Tests should mock execute_pipeline_inline instead of
the old distributed components.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone, timedelta

from andimud_worker.dreamer.runner import (
    DreamerRunner,
    DreamRequest,
    DreamResult,
    CONVERSATION_ANALYSIS_SCENARIOS,
)
from andimud_worker.mixins.dreamer import DreamerMixin
from andimud_worker.worker import MUDAgentWorker
from andimud_worker.config import MUDConfig
from andimud_worker.conversation import MUDConversationManager
from aim_mud_types import MUDSession
from aim.config import ChatConfig
from aim_mud_types import RedisKeys


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

    def test_initialization(self, test_config, test_cvm, test_roster):
        """Test DreamerRunner initializes correctly with simplified inline approach."""
        runner = DreamerRunner(
            config=test_config,
            cvm=test_cvm,
            roster=test_roster,
            persona_id="test_persona",
        )

        assert runner.config == test_config
        assert runner.cvm == test_cvm
        assert runner.roster == test_roster
        assert runner.persona_id == "test_persona"
        # No StateStore or Scheduler - uses inline execution


class TestDreamerRunnerGetConversationId:
    """Test DreamerRunner._get_conversation_id method."""

    def test_returns_base_id_for_analysis_scenarios(
        self, test_config, test_cvm, test_roster
    ):
        """Test returns base_conversation_id for analysis_dialogue."""
        runner = DreamerRunner(
            config=test_config,
            cvm=test_cvm,
            roster=test_roster,
            persona_id="andi",
        )

        conversation_id = runner._get_conversation_id(
            "analysis_dialogue", "andimud_123_abc"
        )
        assert conversation_id == "andimud_123_abc"

    def test_returns_base_id_for_summarizer(
        self, test_config, test_cvm, test_roster
    ):
        """Test returns base_conversation_id for summarizer."""
        runner = DreamerRunner(
            config=test_config,
            cvm=test_cvm,
            roster=test_roster,
            persona_id="andi",
        )

        conversation_id = runner._get_conversation_id(
            "summarizer", "andimud_123_abc"
        )
        assert conversation_id == "andimud_123_abc"

    def test_creates_standalone_id_for_journaler(
        self, test_config, test_cvm, test_roster
    ):
        """Test creates unique conversation ID for journaler_dialogue."""
        runner = DreamerRunner(
            config=test_config,
            cvm=test_cvm,
            roster=test_roster,
            persona_id="andi",
        )

        conversation_id = runner._get_conversation_id(
            "journaler_dialogue", "andimud_123_abc"
        )

        # Should match pattern: journaler_{timestamp}_{random}
        assert conversation_id.startswith("journaler_")
        assert "andimud_123_abc" not in conversation_id

        # Validate format: prefix_timestamp_random
        parts = conversation_id.split("_")
        assert len(parts) == 3
        assert parts[0] == "journaler"
        assert parts[1].isdigit()  # timestamp
        assert len(parts[2]) == 9  # random suffix (9 hex chars)

    def test_creates_standalone_id_for_daydream(
        self, test_config, test_cvm, test_roster
    ):
        """Test creates unique conversation ID for daydream_dialogue."""
        runner = DreamerRunner(
            config=test_config,
            cvm=test_cvm,
            roster=test_roster,
            persona_id="val",
        )

        conversation_id = runner._get_conversation_id(
            "daydream_dialogue", "andimud_456_def"
        )

        # Should match pattern: daydream_{timestamp}_{random}
        assert conversation_id.startswith("daydream_")

        # Validate format
        parts = conversation_id.split("_")
        assert len(parts) == 3
        assert parts[0] == "daydream"
        assert parts[1].isdigit()
        assert len(parts[2]) == 9

    def test_generates_unique_ids_per_call(
        self, test_config, test_cvm, test_roster
    ):
        """Test that multiple calls generate different IDs."""
        runner = DreamerRunner(
            config=test_config,
            cvm=test_cvm,
            roster=test_roster,
            persona_id="andi",
        )

        id1 = runner._get_conversation_id("journaler_dialogue", "andimud_123_abc")
        id2 = runner._get_conversation_id("journaler_dialogue", "andimud_123_abc")

        assert id1 != id2
        assert id1.startswith("journaler_")
        assert id2.startswith("journaler_")


class TestDreamerRunnerRunDream:
    """Test DreamerRunner.run_dream method."""

    @pytest.mark.asyncio
    async def test_successful_dream_execution(
        self, test_config, test_cvm, test_roster
    ):
        """Test successful dream pipeline execution using inline scheduler."""
        runner = DreamerRunner(
            config=test_config,
            cvm=test_cvm,
            roster=test_roster,
            persona_id="andi",
        )

        # Mock execute_pipeline_inline to return a pipeline ID
        with patch("andimud_worker.dreamer.runner.execute_pipeline_inline") as mock_execute:
            mock_execute.return_value = "pipeline_123"

            request = DreamRequest(scenario="test_standard")
            result = await runner.run_dream(request, "andimud_123_abc")

        assert result.success is True
        assert result.pipeline_id == "pipeline_123"
        assert result.scenario == "test_standard"
        assert result.error is None
        assert result.duration_seconds > 0

        # Verify execute_pipeline_inline was called correctly
        mock_execute.assert_called_once()
        call_kwargs = mock_execute.call_args[1]
        assert call_kwargs["scenario_name"] == "test_standard"
        assert call_kwargs["persona_id"] == "andi"
        assert call_kwargs["cvm"] == test_cvm

    @pytest.mark.asyncio
    async def test_dream_execution_with_query_and_guidance(
        self, test_config, test_cvm, test_roster
    ):
        """Test dream execution passes query and guidance to inline scheduler."""
        runner = DreamerRunner(
            config=test_config,
            cvm=test_cvm,
            roster=test_roster,
            persona_id="andi",
        )

        with patch("andimud_worker.dreamer.runner.execute_pipeline_inline") as mock_execute:
            mock_execute.return_value = "pipeline_456"

            request = DreamRequest(
                scenario="test_standard",
                query="What happened today?",
                guidance="Focus on emotions",
            )
            await runner.run_dream(request, "andimud_123_abc")

        call_kwargs = mock_execute.call_args[1]
        assert call_kwargs["query_text"] == "What happened today?"
        assert call_kwargs["guidance"] == "Focus on emotions"

    @pytest.mark.asyncio
    async def test_dream_execution_handles_pipeline_failure(
        self, test_config, test_cvm, test_roster
    ):
        """Test dream execution handles pipeline failure gracefully."""
        runner = DreamerRunner(
            config=test_config,
            cvm=test_cvm,
            roster=test_roster,
            persona_id="andi",
        )

        with patch("andimud_worker.dreamer.runner.execute_pipeline_inline") as mock_execute:
            mock_execute.side_effect = Exception("Pipeline execution failed")

            request = DreamRequest(scenario="analysis_dialogue")
            result = await runner.run_dream(request, "andimud_123_abc")

        assert result.success is False
        assert result.pipeline_id is None
        assert result.scenario == "analysis_dialogue"
        assert "Pipeline execution failed" in result.error
        assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_heartbeat_callback_receives_pipeline_and_step_ids(
        self, test_config, test_cvm, test_roster
    ):
        """Test heartbeat callback receives pipeline_id and step_id arguments.

        This test would have caught the production bug where the wrapped_heartbeat
        function in runner.py calls heartbeat_callback() with no arguments, but
        the callback expects (pipeline_id: str, step_id: str).

        CURRENT BUG (line 198 in runner.py):
            async def wrapped_heartbeat(pipeline_id: str, step_id: str) -> None:
                await heartbeat_callback()  # BUG: Missing arguments

        CORRECT CODE:
            async def wrapped_heartbeat(pipeline_id: str, step_id: str) -> None:
                await heartbeat_callback(pipeline_id, step_id)
        """
        runner = DreamerRunner(
            config=test_config,
            cvm=test_cvm,
            roster=test_roster,
            persona_id="andi",
        )

        # Track heartbeat callback invocations
        heartbeat_calls = []

        async def mock_heartbeat(pipeline_id: str, step_id: str) -> None:
            """Mock heartbeat that expects pipeline_id and step_id."""
            heartbeat_calls.append({
                "pipeline_id": pipeline_id,
                "step_id": step_id,
            })

        # Mock execute_pipeline_inline to simulate step execution
        async def mock_execute_pipeline(**kwargs):
            # Extract the wrapped heartbeat callback
            callback = kwargs.get("heartbeat_callback")
            if callback:
                # Simulate inline scheduler calling the callback with both arguments
                # This is what happens at line 344 in inline/scheduler.py
                await callback("pipeline_123", "step_1")
            return "pipeline_123"

        with patch("andimud_worker.dreamer.runner.execute_pipeline_inline", mock_execute_pipeline):
            request = DreamRequest(scenario="test_standard")
            result = await runner.run_dream(
                request,
                "andimud_123_abc",
                heartbeat_callback=mock_heartbeat
            )

        # Verify: Dream completed successfully
        assert result.success is True
        assert result.pipeline_id == "pipeline_123"

        # Verify: Heartbeat callback was invoked with both arguments
        assert len(heartbeat_calls) == 1, (
            "Heartbeat callback should have been called once by the inline scheduler"
        )

        # Verify: Callback received the correct arguments
        call = heartbeat_calls[0]
        assert call["pipeline_id"] == "pipeline_123", (
            "Heartbeat callback should receive pipeline_id from inline scheduler"
        )
        assert call["step_id"] == "step_1", (
            "Heartbeat callback should receive step_id from inline scheduler"
        )

    @pytest.mark.asyncio
    async def test_uses_base_conversation_id_for_analysis(
        self, test_config, test_cvm, test_roster
    ):
        """Test analysis_dialogue uses base conversation_id."""
        runner = DreamerRunner(
            config=test_config,
            cvm=test_cvm,
            roster=test_roster,
            persona_id="andi",
        )

        with patch("andimud_worker.dreamer.runner.execute_pipeline_inline") as mock_execute:
            mock_execute.return_value = "pipeline_abc"

            request = DreamRequest(scenario="analysis_dialogue")
            await runner.run_dream(request, "andimud_123_abc")

        call_kwargs = mock_execute.call_args[1]
        assert call_kwargs["conversation_id"] == "andimud_123_abc"

    @pytest.mark.asyncio
    async def test_uses_standalone_conversation_id_for_journaler(
        self, test_config, test_cvm, test_roster
    ):
        """Test journaler_dialogue uses standalone conversation_id."""
        runner = DreamerRunner(
            config=test_config,
            cvm=test_cvm,
            roster=test_roster,
            persona_id="andi",
        )

        with patch("andimud_worker.dreamer.runner.execute_pipeline_inline") as mock_execute:
            mock_execute.return_value = "pipeline_def"

            request = DreamRequest(scenario="journaler_dialogue")
            await runner.run_dream(request, "andimud_123_abc")

        call_kwargs = mock_execute.call_args[1]
        conversation_id = call_kwargs["conversation_id"]

        # Should be unique per execution (journaler_{timestamp}_{random})
        assert conversation_id.startswith("journaler_")
        parts = conversation_id.split("_")
        assert len(parts) == 3


class TestDreamerMixinInit:
    """Test DreamerMixin._init_dreamer method."""

    @pytest.mark.asyncio
    async def test_init_dreamer_creates_runner(self, test_mud_config, mock_redis, test_config):
        """Test _init_dreamer creates DreamerRunner correctly."""
        worker = MUDAgentWorker(config=test_mud_config, redis_client=mock_redis)

        # Setup required attributes
        worker.chat_config = test_config
        worker.cvm = Mock()
        worker.roster = Mock()

        with patch("andimud_worker.mixins.dreamer.DreamerRunner") as mock_runner_class:
            mock_runner_class.return_value = Mock()
            worker._init_dreamer()

            assert worker._dreamer_runner is not None
            mock_runner_class.assert_called_once_with(
                config=test_config,
                cvm=worker.cvm,
                roster=worker.roster,
                persona_id="test_persona",
            )




def test_chatconfig_fixture_has_no_fake_attributes(test_config):
    """Verify our test fixtures don't add nonexistent attributes to ChatConfig.

    This test exists because we previously corrupted ChatConfig with fake
    attributes (model_name) that don't exist on the real class, causing
    production crashes. NEVER AGAIN.

    Production crashed on 2026-01-08 because:
    - Test fixture added config.model_name (FAKE attribute)
    - Tests passed with fake data
    - Production code used config.model_name at runner.py:224
    - Production crashed: 'ChatConfig' object has no attribute 'model_name'

    The correct attribute is default_model, not model_name.
    """
    # These are REAL attributes that should exist on ChatConfig
    base_config = ChatConfig()
    assert hasattr(base_config, 'default_model'), "ChatConfig should have default_model"
    assert hasattr(base_config, 'thought_model'), "ChatConfig should have thought_model"
    assert hasattr(base_config, 'codex_model'), "ChatConfig should have codex_model"

    # This is a FAKE attribute that should NOT exist
    assert not hasattr(base_config, 'model_name'), \
        "ChatConfig should NOT have model_name - use default_model instead"

    # Verify our fixture doesn't corrupt the object
    # Fixture should only set attributes that exist on real ChatConfig
    # Get all non-dunder attributes from both objects
    real_attrs = {attr for attr in dir(base_config) if not attr.startswith('_')}
    fixture_attrs = {attr for attr in dir(test_config) if not attr.startswith('_')}

    # Find attributes that exist on fixture but not on real class
    fake_attrs = fixture_attrs - real_attrs
    assert fake_attrs == set(), \
        f"Fixture adds fake attributes: {fake_attrs}. Use only real ChatConfig attributes."


@pytest.mark.skip(reason="Integration tests need updating for inline scheduler (2026-01-09)")
class TestDreamerRunnerIntegration:
    """Integration tests for DreamerRunner pipeline execution.

    TODO: These tests need updating to work with the inline scheduler.
    The old tests used StateStore, Scheduler, and distributed infrastructure.
    The new inline approach executes pipelines synchronously in-process.

    See packages/aim-core/src/aim/dreamer/inline/ for the new implementation.
    """

    @pytest.mark.asyncio
    async def test_full_pipeline_execution_completes(
        self,
        test_config,
        test_cvm,
        test_roster,
        mock_redis,
        test_state_store,
        test_scheduler,
        minimal_standard_scenario,
        mock_execute_step,
        mock_load_scenario,
    ):
        """Test that full pipeline execution completes successfully.

        This test verifies:
        1. Pipeline starts and enqueues initial jobs
        2. _execute_pipeline() processes all jobs
        3. Steps are marked COMPLETE in state store
        4. The while loop breaks correctly when no more jobs
        5. run_dream() returns success=True

        This would catch the production bug where the pipeline hangs or fails
        to complete, preventing _update_conversation_report() from being called.
        """
        # Create runner with real StateStore and Scheduler
        runner = DreamerRunner(
            config=test_config,
            cvm=test_cvm,
            roster=test_roster,
            redis_client=mock_redis,
            agent_id="andi",
            persona_id="andi",
        )

        # Replace with test infrastructure
        runner.state_store = test_state_store
        runner.scheduler = test_scheduler

        # Mock start_pipeline to properly initialize pipeline state
        with patch("andimud_worker.dreamer.runner.start_pipeline") as mock_start:
            pipeline_id = "test_pipeline_integration_123"
            mock_start.return_value = pipeline_id

            # Mock execute_step (LLM call - external service)
            with patch("andimud_worker.dreamer.runner.execute_step", mock_execute_step):
                # Mock load_scenario (file I/O - external)
                with patch("andimud_worker.dreamer.runner.load_scenario", mock_load_scenario):
                    # Mock create_message to return proper ConversationMessage
                    def mock_create_message(state, step_def, result):
                        from aim.conversation.message import ConversationMessage
                        return ConversationMessage(
                            doc_id=result.doc_id,
                            document_type=result.document_type,
                            user_id=state.user_id,
                            persona_id=state.persona_id,
                            conversation_id=state.conversation_id or "test",
                            branch=state.branch,
                            sequence_no=state.step_counter,
                            speaker_id=state.persona_id,
                            listener_id="self",
                            role="assistant",
                            content=result.response,
                            timestamp=int(result.timestamp.timestamp()),
                        )

                    with patch("andimud_worker.dreamer.runner.create_message", mock_create_message):
                        # Setup: Initialize pipeline state in StateStore
                        from aim.dreamer.models import PipelineState, StepStatus

                        state = PipelineState(
                            pipeline_id=pipeline_id,
                            scenario_name="test_standard",
                            conversation_id="test_conversation",
                            persona_id="andi",
                            user_id="test_user",
                            model=test_config.default_model,
                            thought_model=test_config.thought_model,
                            codex_model=test_config.codex_model,
                            guidance=None,
                            query_text="Test query",
                            persona_mood=None,
                            branch=0,
                            step_counter=0,
                            completed_steps=[],
                            step_doc_ids={},
                            seed_doc_ids={},
                            context_doc_ids=[],
                            context_documents=None,
                            created_at=datetime.now(timezone.utc),
                            updated_at=datetime.now(timezone.utc),
                        )
                        await test_state_store.save_state(state)

                        # Mark step1 as PENDING (will be processed first)
                        await test_state_store.set_step_status(
                            pipeline_id, "step1", StepStatus.PENDING
                        )
                        await test_state_store.set_step_status(
                            pipeline_id, "step2", StepStatus.PENDING
                        )

                        # Enqueue initial job (step1) - this is what start_pipeline would do
                        await test_scheduler.enqueue_step(pipeline_id, "step1")

                        # Execute: Run the dream (uses REAL _execute_pipeline)
                        request = DreamRequest(scenario="test_standard")
                        result = await runner.run_dream(request, "andimud_123_abc")

        # Verify: Pipeline completed successfully
        assert result.success is True, f"Pipeline should complete successfully, got: {result.error}"
        assert result.pipeline_id == pipeline_id
        assert result.duration_seconds > 0

        # Verify: All steps marked COMPLETE
        step1_status = await test_state_store.get_step_status(pipeline_id, "step1")
        step2_status = await test_state_store.get_step_status(pipeline_id, "step2")

        assert step1_status == StepStatus.COMPLETE, "step1 should be marked COMPLETE"
        assert step2_status == StepStatus.COMPLETE, "step2 should be marked COMPLETE"

        # Verify: State updated correctly
        final_state = await test_state_store.load_state(pipeline_id)
        assert "step1" in final_state.completed_steps
        assert "step2" in final_state.completed_steps
        assert "step1" in final_state.step_doc_ids
        assert "step2" in final_state.step_doc_ids

    @pytest.mark.asyncio
    async def test_dream_command_updates_conversation_report(
        self,
        test_mud_config,
        mock_redis,
        test_config,
        test_cvm,
        test_roster,
        test_state_store,
        test_scheduler,
        test_conversation_manager,
        minimal_standard_scenario,
        mock_execute_step,
        mock_load_scenario,
    ):
        """Test that DreamCommand calls _update_conversation_report on success.

        This is the END-TO-END integration test that catches the production bug:
        If run_dream() doesn't return success=True, then DreamCommand won't call
        _update_conversation_report().

        This test verifies:
        1. DreamCommand.execute() runs successfully
        2. _update_conversation_report() is called
        3. Result status is "done"
        """
        from andimud_worker.commands.dream import DreamCommand
        from andimud_worker.worker import MUDAgentWorker

        # Create worker
        worker = MUDAgentWorker(config=test_mud_config, redis_client=mock_redis)
        worker.chat_config = test_config
        worker.cvm = test_cvm
        worker.roster = test_roster
        worker.conversation_manager = test_conversation_manager

        # Initialize dreamer with test infrastructure
        worker._init_dreamer()
        worker._dreamer_runner.state_store = test_state_store
        worker._dreamer_runner.scheduler = test_scheduler

        # Mock _update_conversation_report to track if it's called
        update_report_called = []

        async def mock_update_report():
            update_report_called.append(True)

        worker._update_conversation_report = mock_update_report

        # Setup: Initialize pipeline state
        pipeline_id = "test_pipeline_dream_cmd_456"

        with patch("andimud_worker.dreamer.runner.start_pipeline") as mock_start:
            mock_start.return_value = pipeline_id

            with patch("andimud_worker.dreamer.runner.execute_step", mock_execute_step):
                with patch("andimud_worker.dreamer.runner.load_scenario", mock_load_scenario):
                    # Mock create_message to return proper ConversationMessage
                    def mock_create_message(state, step_def, result):
                        from aim.conversation.message import ConversationMessage
                        return ConversationMessage(
                            doc_id=result.doc_id,
                            document_type=result.document_type,
                            user_id=state.user_id,
                            persona_id=state.persona_id,
                            conversation_id=state.conversation_id or "test",
                            branch=state.branch,
                            sequence_no=state.step_counter,
                            speaker_id=state.persona_id,
                            listener_id="self",
                            role="assistant",
                            content=result.response,
                            timestamp=int(result.timestamp.timestamp()),
                        )

                    with patch("andimud_worker.dreamer.runner.create_message", mock_create_message):
                        # Setup pipeline state
                        from aim.dreamer.models import PipelineState, StepStatus

                        state = PipelineState(
                            pipeline_id=pipeline_id,
                            scenario_name="test_standard",
                            conversation_id="test_conversation",
                            persona_id="andi",
                            user_id="test_user",
                            model=test_config.default_model,
                            thought_model=test_config.thought_model,
                            codex_model=test_config.codex_model,
                            guidance=None,
                            query_text="Test query",
                            persona_mood=None,
                            branch=0,
                            step_counter=0,
                            completed_steps=[],
                            step_doc_ids={},
                            seed_doc_ids={},
                            context_doc_ids=[],
                            context_documents=None,
                            created_at=datetime.now(timezone.utc),
                            updated_at=datetime.now(timezone.utc),
                        )
                        await test_state_store.save_state(state)
                        await test_state_store.set_step_status(pipeline_id, "step1", StepStatus.PENDING)
                        await test_state_store.set_step_status(pipeline_id, "step2", StepStatus.PENDING)

                        # Enqueue initial job
                        await test_scheduler.enqueue_step(pipeline_id, "step1")

                        # Execute: Run DreamCommand
                        command = DreamCommand()
                        result = await command.execute(
                            worker,
                            turn_id="test_turn_123",
                            scenario="test_standard",
                            query="Test query",
                            guidance=None,
                            conversation_id=None,
                        )

        # Verify: Command completed successfully
        assert result.status == "done", f"Command should succeed, got: {result.message}"
        assert result.complete is True

        # Verify: _update_conversation_report was called
        assert len(update_report_called) > 0, (
            "PRODUCTION BUG DETECTED: _update_conversation_report() was not called! "
            "This means run_dream() didn't return success=True, so the report never updated."
        )

    @pytest.mark.asyncio
    async def test_execute_pipeline_breaks_when_no_jobs(
        self,
        test_config,
        test_cvm,
        test_roster,
        mock_redis,
        test_state_store,
        test_scheduler,
    ):
        """Test _execute_pipeline breaks immediately when no jobs in queue.

        This test ensures the while loop doesn't hang forever when there are
        no jobs to process. This is a critical edge case that could cause the
        production bug (hanging pipeline = no completion = no report update).
        """
        # Create runner with real StateStore and Scheduler
        runner = DreamerRunner(
            config=test_config,
            cvm=test_cvm,
            roster=test_roster,
            redis_client=mock_redis,
            agent_id="andi",
            persona_id="andi",
        )

        runner.state_store = test_state_store
        runner.scheduler = test_scheduler

        # Setup: Empty queue (no jobs)
        pipeline_id = "test_pipeline_empty_789"

        # Execute: Call _execute_pipeline with empty queue
        import time
        start_time = time.time()

        await runner._execute_pipeline(pipeline_id, heartbeat_callback=None)

        elapsed = time.time() - start_time

        # Verify: Returns immediately (within 1 second)
        assert elapsed < 1.0, (
            f"_execute_pipeline should break immediately with no jobs, "
            f"but took {elapsed:.2f}s. This indicates a hang bug."
        )

    @pytest.mark.asyncio
    async def test_execute_pipeline_processes_jobs_in_order(
        self,
        test_config,
        test_cvm,
        test_roster,
        mock_redis,
        test_state_store,
        test_scheduler,
        minimal_standard_scenario,
        mock_execute_step,
        mock_load_scenario,
    ):
        """Test _execute_pipeline processes jobs in correct dependency order.

        This test verifies:
        1. step1 executes first (no dependencies)
        2. step2 executes after step1 completes (depends_on: [step1])
        3. Both steps are marked COMPLETE
        4. Pipeline completes cleanly
        """
        # Create runner
        runner = DreamerRunner(
            config=test_config,
            cvm=test_cvm,
            roster=test_roster,
            redis_client=mock_redis,
            agent_id="andi",
            persona_id="andi",
        )

        runner.state_store = test_state_store
        runner.scheduler = test_scheduler

        pipeline_id = "test_pipeline_order_999"

        # Track execution order
        execution_order = []

        async def tracking_execute_step(state, scenario, step_def, cvm, persona, config, model_set):
            """Track which steps execute."""
            execution_order.append(step_def.id)
            # Call original mock
            return await mock_execute_step(state, scenario, step_def, cvm, persona, config, model_set)

        with patch("andimud_worker.dreamer.runner.execute_step", tracking_execute_step):
            with patch("andimud_worker.dreamer.runner.load_scenario", mock_load_scenario):
                # Mock create_message to return proper ConversationMessage
                def mock_create_message(state, step_def, result):
                    from aim.conversation.message import ConversationMessage
                    return ConversationMessage(
                        doc_id=result.doc_id,
                        document_type=result.document_type,
                        user_id=state.user_id,
                        persona_id=state.persona_id,
                        conversation_id=state.conversation_id or "test",
                        branch=state.branch,
                        sequence_no=state.step_counter,
                        speaker_id=state.persona_id,
                        listener_id="self",
                        role="assistant",
                        content=result.response,
                        timestamp=int(result.timestamp.timestamp()),
                    )

                with patch("andimud_worker.dreamer.runner.create_message", mock_create_message):
                    # Setup pipeline state
                    from aim.dreamer.models import PipelineState, StepStatus

                    state = PipelineState(
                        pipeline_id=pipeline_id,
                        scenario_name="test_standard",
                        conversation_id="test_conversation",
                        persona_id="andi",
                        user_id="test_user",
                        model=test_config.default_model,
                        thought_model=test_config.thought_model,
                        codex_model=test_config.codex_model,
                        guidance=None,
                        query_text="Test query",
                        persona_mood=None,
                        branch=0,
                        step_counter=0,
                        completed_steps=[],
                        step_doc_ids={},
                        seed_doc_ids={},
                        context_doc_ids=[],
                        context_documents=None,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                    )
                    await test_state_store.save_state(state)
                    await test_state_store.set_step_status(pipeline_id, "step1", StepStatus.PENDING)
                    await test_state_store.set_step_status(pipeline_id, "step2", StepStatus.PENDING)

                    # Enqueue step1 (step2 will be enqueued after step1 completes)
                    await test_scheduler.enqueue_step(pipeline_id, "step1")

                    # Execute pipeline
                    await runner._execute_pipeline(pipeline_id, heartbeat_callback=None)

        # Verify: Both steps executed
        assert len(execution_order) == 2, f"Expected 2 steps executed, got {len(execution_order)}"

        # Verify: step1 executed before step2
        assert execution_order[0] == "step1", "step1 should execute first"
        assert execution_order[1] == "step2", "step2 should execute after step1"

        # Verify: Both marked COMPLETE
        step1_status = await test_state_store.get_step_status(pipeline_id, "step1")
        step2_status = await test_state_store.get_step_status(pipeline_id, "step2")

        assert step1_status == StepStatus.COMPLETE
        assert step2_status == StepStatus.COMPLETE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
