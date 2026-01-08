# andimud_worker/dreamer_runner.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Inline Dreamer pipeline execution for MUD workers.

This module provides DreamerRunner, which executes Dreamer pipelines
synchronously within the MUD worker process. Unlike the standalone
DreamerWorker which consumes from a Redis queue, DreamerRunner runs
pipelines inline to prevent concurrent LLM calls between regular
turns and dream processing.

Key design decisions:
1. Uses MUD worker's existing CVM (passed in constructor)
2. Creates its own StateStore/Scheduler with agent-specific key prefix
3. Inline step execution - no separate queue consumer needed
4. Heartbeat callback for long-running steps
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Callable, Awaitable
import logging
import time

from redis.asyncio import Redis

from aim.config import ChatConfig
from aim.conversation.model import ConversationModel
from aim.agents.roster import Roster
from aim.dreamer.api import start_pipeline
from aim.dreamer.state import StateStore
from aim.dreamer.scheduler import Scheduler
from aim.dreamer.models import StepJob, StepStatus
from aim.dreamer.scenario import load_scenario
from aim.dreamer.executor import execute_step, create_message, RetryableError
from aim.dreamer.dialogue.strategy import DialogueStrategy
from aim.dreamer.dialogue.scenario import DialogueScenario
from aim.llm.model_set import ModelSet

logger = logging.getLogger(__name__)


# Scenarios that operate on the MUD conversation history
CONVERSATION_ANALYSIS_SCENARIOS = {"analysis_dialogue", "summarizer"}


@dataclass
class DreamRequest:
    """Parameters for a dream execution.

    Attributes:
        scenario: Name of the scenario YAML to run (e.g., "analysis_dialogue")
        query: Optional query text for scenarios that use it (journaler, philosopher)
        guidance: Optional guidance text to influence generation
        triggered_by: How the dream was triggered ("manual" or "auto")
        target_conversation_id: Explicit conversation ID for analysis commands.
            If provided, overrides the default conversation selection logic.
    """
    scenario: str
    query: Optional[str] = None
    guidance: Optional[str] = None
    triggered_by: str = "manual"
    target_conversation_id: Optional[str] = None


@dataclass
class DreamResult:
    """Result of a dream execution.

    Attributes:
        success: Whether the pipeline completed successfully
        pipeline_id: The pipeline ID used for this dream
        scenario: The scenario that was run
        error: Error message if success is False
        duration_seconds: How long the pipeline took to execute
    """
    success: bool
    pipeline_id: Optional[str] = None
    scenario: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0


class DreamerRunner:
    """Runs Dreamer pipelines inline within the MUD worker.

    This class integrates with the existing Dreamer infrastructure
    but executes synchronously (blocking) within the worker's turn
    processing. This prevents concurrent LLM calls between regular
    turns and dream pipelines.

    The runner creates its own StateStore and Scheduler with an
    agent-specific key prefix to isolate state from other dreamers.
    It uses the MUD worker's existing CVM for memory operations.

    Example:
        runner = DreamerRunner(
            config=config,
            cvm=cvm,
            roster=roster,
            redis_client=redis,
            agent_id="andi",
            persona_id="andi",
        )

        # Analysis scenario - uses base_conversation_id
        result = await runner.run_dream(
            request=DreamRequest(scenario="analysis_dialogue"),
            base_conversation_id="andimud_123_abc",
            heartbeat_callback=refresh_heartbeat,
        )

        # With explicit target_conversation_id (overrides base)
        result = await runner.run_dream(
            request=DreamRequest(
                scenario="analysis_dialogue",
                target_conversation_id="specific_conv_123"
            ),
            base_conversation_id="andimud_123_abc",
            heartbeat_callback=refresh_heartbeat,
        )
    """

    def __init__(
        self,
        config: ChatConfig,
        cvm: ConversationModel,
        roster: Roster,
        redis_client: Redis,
        agent_id: str,
        persona_id: str,
    ):
        """Initialize the DreamerRunner.

        Args:
            config: ChatConfig with provider keys and settings
            cvm: ConversationModel for memory operations (shared with MUD worker)
            roster: Roster with persona definitions
            redis_client: Async Redis client
            agent_id: Unique identifier for this agent
            persona_id: Persona ID to use for dreams
        """
        self.config = config
        self.cvm = cvm
        self.roster = roster
        self.redis = redis_client
        self.agent_id = agent_id
        self.persona_id = persona_id

        # Initialize ModelSet for this persona
        persona = self.roster.get_persona(self.persona_id)
        self.model_set = ModelSet.from_config(config, persona)

        logger.info(
            f"DreamerRunner initialized for {persona_id}: "
            f"default={self.model_set.default_model}, "
            f"analysis={self.model_set.analysis_model}, "
            f"codex={self.model_set.codex_model}"
        )

        # Create StateStore/Scheduler with agent-specific key prefix
        # This isolates pipeline state from other agents and the main dreamer
        key_prefix = f"mud:dreamer:{agent_id}"
        self.state_store = StateStore(redis_client, key_prefix=key_prefix)
        self.scheduler = Scheduler(redis_client, self.state_store)

        # Override queue keys to use agent-specific queues
        self.scheduler.queue_key = f"{key_prefix}:queue:steps"
        self.scheduler.delayed_key = f"{key_prefix}:queue:steps:delayed"

    def _get_conversation_id(self, scenario: str, base_conversation_id: str) -> str:
        """Get conversation ID based on scenario type.

        Analysis/summarizer scenarios operate on the existing MUD conversation
        and should use the provided conversation_id. Other scenarios (journaler,
        philosopher, daydream) create standalone conversations.

        Args:
            scenario: Name of the scenario
            base_conversation_id: The MUD conversation ID

        Returns:
            Conversation ID to use for this pipeline
        """
        if scenario in CONVERSATION_ANALYSIS_SCENARIOS:
            return base_conversation_id
        else:
            # Standalone scenarios use a unique conversation per scenario
            return f"mud_dream_{self.agent_id}_{scenario}"

    async def run_dream(
        self,
        request: DreamRequest,
        base_conversation_id: str,
        heartbeat_callback: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> DreamResult:
        """Execute a complete dream pipeline.

        This method starts a pipeline and executes all steps inline,
        blocking until completion. It uses the existing Dreamer
        infrastructure but runs synchronously within the MUD worker.

        Args:
            request: DreamRequest with scenario and parameters. If
                target_conversation_id is set, it will be used directly.
                Otherwise, conversation selection follows scenario type:
                analysis scenarios use base_conversation_id, creative
                scenarios use a standalone conversation.
            base_conversation_id: The MUD conversation ID to use as fallback
                for analysis scenarios when no explicit target is provided.
            heartbeat_callback: Optional async callback to refresh heartbeat
                during long-running steps (prevents turn timeout)

        Returns:
            DreamResult with success status and metadata
        """
        start_time = time.time()

        try:
            # Use explicit target_conversation_id if provided, otherwise use scenario-aware logic
            if request.target_conversation_id:
                target_conversation_id = request.target_conversation_id
            else:
                target_conversation_id = self._get_conversation_id(
                    request.scenario, base_conversation_id
                )

            logger.info(
                f"Starting dream: scenario={request.scenario} "
                f"conversation={target_conversation_id} triggered_by={request.triggered_by}"
            )

            # Start the pipeline using existing Dreamer API
            pipeline_id = await start_pipeline(
                scenario_name=request.scenario,
                config=self.config,
                model_name=self.model_set.default_model,  # Use persona's default
                state_store=self.state_store,
                scheduler=self.scheduler,
                conversation_id=target_conversation_id,
                persona_id=self.persona_id,
                query_text=request.query,
                guidance=request.guidance,
                cvm=self.cvm,  # Pass our CVM for correct memory isolation
            )

            logger.info(f"Pipeline started: {pipeline_id}")

            # Execute pipeline steps inline
            await self._execute_pipeline(pipeline_id, heartbeat_callback)

            duration = time.time() - start_time
            logger.info(
                f"Dream completed: pipeline={pipeline_id} "
                f"duration={duration:.1f}s scenario={request.scenario}"
            )

            return DreamResult(
                success=True,
                pipeline_id=pipeline_id,
                scenario=request.scenario,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Dream failed: scenario={request.scenario} "
                f"error={str(e)} duration={duration:.1f}s"
            )
            return DreamResult(
                success=False,
                scenario=request.scenario,
                error=str(e),
                duration_seconds=duration,
            )

    async def _execute_pipeline(
        self,
        pipeline_id: str,
        heartbeat_callback: Optional[Callable[[], Awaitable[None]]],
    ) -> None:
        """Execute all steps of a pipeline inline.

        Adapts DreamerWorker's queue-based execution for inline use.
        Instead of blocking on a queue, we pop jobs non-blocking and
        execute them immediately until no more work remains.

        Args:
            pipeline_id: The pipeline to execute
            heartbeat_callback: Optional callback to refresh heartbeat
        """
        max_iterations = 100  # Safety limit to prevent infinite loops
        iterations = 0

        while iterations < max_iterations:
            iterations += 1

            # Pop job from queue with timeout=0 (non-blocking)
            job = await self.scheduler.pop_step_job(timeout=0)

            if job is None:
                # No job in main queue - check for delayed jobs
                moved = await self.scheduler.process_delayed_jobs()
                if moved > 0:
                    # Jobs were moved to main queue - try again
                    continue
                else:
                    # No more work - pipeline is complete
                    break

            # Verify job belongs to our pipeline
            if job.pipeline_id != pipeline_id:
                logger.warning(
                    f"Job pipeline mismatch: expected={pipeline_id} "
                    f"got={job.pipeline_id}, requeueing"
                )
                await self.scheduler.requeue_step(job, delay=0)
                continue

            # Execute the step
            await self._execute_step(job, heartbeat_callback)

        if iterations >= max_iterations:
            logger.error(
                f"Pipeline execution hit iteration limit: {max_iterations}"
            )

    async def _execute_step(
        self,
        job: StepJob,
        heartbeat_callback: Optional[Callable[[], Awaitable[None]]],
    ) -> None:
        """Execute a single pipeline step.

        Follows DreamerWorker.process_job() pattern:
        1. Acquire distributed lock
        2. Load state and scenario
        3. Check dependencies satisfied
        4. Execute step via executor
        5. Save result to CVM
        6. Update state
        7. Mark complete and enqueue downstream steps

        Args:
            job: StepJob to process
            heartbeat_callback: Optional callback to refresh heartbeat
        """
        # 1. Acquire distributed lock
        lock_acquired = await self.state_store.acquire_lock(
            job.pipeline_id, job.step_id, ttl=300
        )

        if not lock_acquired:
            logger.warning(f"Could not acquire lock for step {job.step_id}")
            return

        try:
            # Refresh heartbeat before starting step
            if heartbeat_callback:
                await heartbeat_callback()

            # 1b. Check if step should be processed
            current_status = await self.state_store.get_step_status(
                job.pipeline_id, job.step_id
            )
            if current_status == StepStatus.COMPLETE:
                return
            if current_status == StepStatus.FAILED:
                return

            # 2. Check state type and route accordingly
            state_type = await self.state_store.get_state_type(job.pipeline_id)

            if state_type is None:
                logger.error(f"Pipeline state not found: {job.pipeline_id}")
                return

            if state_type == 'dialogue':
                # Dialogue flow - use DialogueScenario
                await self._execute_dialogue_step(job, heartbeat_callback)
                return

            # Standard flow - load PipelineState
            state = await self.state_store.load_state(job.pipeline_id)
            scenario = load_scenario(state.scenario_name)
            scenario.compute_dependencies()
            step_def = scenario.steps.get(job.step_id)

            if step_def is None:
                logger.error(
                    f"Step {job.step_id} not found in scenario {state.scenario_name}"
                )
                return

            # 3. Check dependencies satisfied
            if not await self.scheduler.all_deps_complete(job.pipeline_id, step_def):
                await self.scheduler.requeue_step(job, delay=5)
                return

            # Mark step as RUNNING
            await self.state_store.set_step_status(
                job.pipeline_id, job.step_id, StepStatus.RUNNING
            )

            # Load persona
            persona = self.roster.personas.get(state.persona_id)

            if persona is None:
                await self.scheduler.mark_failed(
                    job.pipeline_id,
                    job.step_id,
                    f"Persona {state.persona_id} not found"
                )
                return

            # Refresh heartbeat before LLM call
            if heartbeat_callback:
                await heartbeat_callback()

            # 4. Execute step
            result, context_doc_ids, is_initial_context = await execute_step(
                state=state,
                scenario=scenario,
                step_def=step_def,
                cvm=self.cvm,
                persona=persona,
                config=self.config,
                model_set=self.model_set,
            )

            # Refresh heartbeat after LLM call
            if heartbeat_callback:
                await heartbeat_callback()

            # 5. Persist result to CVM immediately
            message = create_message(state, step_def, result)
            self.cvm.insert(message)

            # 6. Update state with doc_id reference and accumulated context
            state.completed_steps.append(job.step_id)
            state.step_doc_ids[job.step_id] = result.doc_id
            state.step_counter += 1
            state.updated_at = datetime.now(timezone.utc)

            # Update accumulated context
            if is_initial_context:
                state.context_doc_ids = context_doc_ids + [result.doc_id]
            else:
                state.context_doc_ids.append(result.doc_id)

            await self.state_store.save_state(state)

            # 7. Mark complete
            await self.scheduler.mark_complete(job.pipeline_id, job.step_id)

            # 8. Enqueue downstream steps
            for next_step_id in step_def.next:
                next_step = scenario.steps.get(next_step_id)
                if next_step and await self.scheduler.all_deps_complete(
                    job.pipeline_id, next_step
                ):
                    await self.scheduler.enqueue_step(job.pipeline_id, next_step_id)

            # 9. Check if pipeline complete
            if not step_def.next:
                await self.scheduler.check_pipeline_complete(job.pipeline_id, scenario)

            logger.info(
                f"Step complete: {job.step_id} pipeline={job.pipeline_id[:8]}..."
            )

        except RetryableError as e:
            # Retryable error - requeue with exponential backoff
            if job.attempt < job.max_attempts:
                delay = 30 * job.attempt
                incremented_job = job.increment_attempt()
                await self.scheduler.requeue_step(incremented_job, delay=delay)
                logger.warning(
                    f"Step {job.step_id} failed (retryable), "
                    f"attempt {job.attempt}/{job.max_attempts}: {e}"
                )
            else:
                await self.scheduler.mark_failed(
                    job.pipeline_id,
                    job.step_id,
                    f"Max retries exceeded: {str(e)}"
                )

        except Exception as e:
            # Non-retryable error - mark failed immediately
            await self.scheduler.mark_failed(
                job.pipeline_id,
                job.step_id,
                f"Execution error: {str(e)}"
            )
            logger.error(f"Step {job.step_id} failed: {e}")

        finally:
            # Always release lock
            await self.state_store.release_lock(job.pipeline_id, job.step_id)

    async def _execute_dialogue_step(
        self,
        job: StepJob,
        heartbeat_callback: Optional[Callable[[], Awaitable[None]]],
    ) -> None:
        """Execute a dialogue flow step.

        Dialogue flows use DialogueState and DialogueStrategy instead
        of PipelineState. This method mirrors DreamerWorker._process_dialogue_step().

        Args:
            job: StepJob to process
            heartbeat_callback: Optional callback to refresh heartbeat
        """
        # Load dialogue state
        dialogue_state = await self.state_store.load_dialogue_state(job.pipeline_id)

        if dialogue_state is None:
            logger.error(f"Dialogue state not found: {job.pipeline_id}")
            await self.scheduler.mark_failed(
                job.pipeline_id,
                job.step_id,
                "Dialogue state not found"
            )
            return

        # Load scenario for DAG operations
        scenario = load_scenario(dialogue_state.strategy_name)
        scenario.compute_dependencies()

        # Mark step as RUNNING
        await self.state_store.set_step_status(
            job.pipeline_id, job.step_id, StepStatus.RUNNING
        )

        # Load persona
        persona = self.roster.personas.get(dialogue_state.persona_id)

        if persona is None:
            await self.scheduler.mark_failed(
                job.pipeline_id,
                job.step_id,
                f"Persona {dialogue_state.persona_id} not found"
            )
            return

        # Refresh heartbeat before LLM call
        if heartbeat_callback:
            await heartbeat_callback()

        # Create DialogueStrategy and DialogueScenario
        strategy = DialogueStrategy.load(scenario.name)
        dialogue_scenario = DialogueScenario(
            strategy=strategy,
            persona=persona,
            config=self.config,
            cvm=self.cvm,
            model_set=self.model_set,
        )

        # Set the state (already initialized)
        dialogue_scenario.state = dialogue_state

        # Execute the step
        turn = await dialogue_scenario.execute_step(job.step_id)

        # Refresh heartbeat after LLM call
        if heartbeat_callback:
            await heartbeat_callback()

        # Save updated state
        await self.state_store.save_dialogue_state(dialogue_scenario.state)

        # Mark complete
        await self.scheduler.mark_complete(job.pipeline_id, job.step_id)

        # Enqueue downstream steps
        step = strategy.get_step(job.step_id)
        for next_step_id in step.next:
            await self.scheduler.enqueue_step(job.pipeline_id, next_step_id)

        # Check if pipeline complete (no more steps)
        if not step.next:
            await self.scheduler.check_pipeline_complete(job.pipeline_id, scenario)

        logger.info(
            f"Dialogue step complete: {job.step_id} pipeline={job.pipeline_id[:8]}..."
        )
