# aim/dreamer/worker.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Worker process for consuming and executing pipeline steps."""

import asyncio
import logging
import signal
from datetime import datetime
from typing import Optional

from .executor import execute_step, create_message, RetryableError

logger = logging.getLogger(__name__)
from .models import StepJob, StepStatus
from .scenario import load_scenario
from .scheduler import Scheduler
from .state import StateStore
from .dialogue.strategy import DialogueStrategy
from .dialogue.scenario import DialogueScenario
from ..config import ChatConfig
from ..conversation.model import ConversationModel
from ..agents.roster import Roster


class DreamerWorker:
    """Worker that consumes steps from the queue and executes them."""

    def __init__(
        self,
        config: ChatConfig,
        state_store: StateStore,
        scheduler: Scheduler,
    ):
        """Initialize worker with configuration and dependencies.

        Args:
            config: ChatConfig with provider keys and settings
            state_store: StateStore for pipeline state management
            scheduler: Scheduler for queue operations
        """
        self.config = config
        self.state_store = state_store
        self.scheduler = scheduler
        self.running = False

        # Initialize shared resources (loaded once, reused across jobs)
        self.cvm: Optional[ConversationModel] = None
        self.roster: Optional[Roster] = None

    async def start(self) -> None:
        """Start the worker loop.

        Initializes shared resources (CVM, Roster) and enters the main
        processing loop, consuming jobs from the queue until stopped.
        """
        # Initialize shared resources once
        self.cvm = ConversationModel.from_config(self.config)
        self.roster = Roster.from_config(self.config)

        # Set running flag
        self.running = True

        # Setup signal handlers for graceful shutdown
        self.setup_signal_handlers()

        # Main worker loop
        while self.running:
            try:
                # Check if paused
                if await self._is_paused():
                    await asyncio.sleep(1)
                    continue

                # Pop next job (blocking with timeout to check running flag)
                job = await self.scheduler.pop_step_job(timeout=1)

                if job is None:
                    # Timeout - check for delayed jobs and continue
                    await self.scheduler.process_delayed_jobs()
                    continue

                # Process the job
                await self.process_job(job)

            except Exception as e:
                # Log error but continue processing
                logger.error(f"Error in worker loop: {e}")
                continue

    async def stop(self) -> None:
        """Gracefully stop the worker.

        Sets the running flag to False, allowing the current job
        to complete before shutting down.
        """
        self.running = False

    async def _is_paused(self) -> bool:
        """Check if worker is paused via Redis flag.

        Returns:
            bool: True if paused, False if running
        """
        value = await self.scheduler.redis.get("aim:dreamer:paused")
        return value == b"1"

    async def process_job(self, job: StepJob) -> None:
        """Process a single step job.

        Full processing flow:
        1. Acquire distributed lock
        2. Load state and scenario
        3. Check dependencies satisfied
        4. Execute step via executor
        5. Save result to CVM
        6. Update state
        7. Mark complete and enqueue downstream steps

        Error handling:
        - RetryableError: requeue with exponential backoff
        - Other errors: mark failed
        - Always release lock in finally

        Args:
            job: StepJob to process
        """
        # 1. Acquire distributed lock
        lock_acquired = await self.state_store.acquire_lock(
            job.pipeline_id, job.step_id, ttl=300
        )

        if not lock_acquired:
            # Another worker got it - skip
            return

        try:
            # 1b. Check if step should be processed
            current_status = await self.state_store.get_step_status(
                job.pipeline_id, job.step_id
            )
            if current_status == StepStatus.COMPLETE:
                # Already done - skip
                return
            if current_status == StepStatus.FAILED:
                # Already failed - needs explicit resume, skip
                return

            # 1c. Check dependencies are satisfied
            # Load scenario to get step definition with depends_on
            state_type = await self.state_store.get_state_type(job.pipeline_id)
            if state_type == 'dialogue':
                temp_state = await self.state_store.load_dialogue_state(job.pipeline_id)
                temp_scenario = load_scenario(temp_state.strategy_name) if temp_state else None
            else:
                temp_state = await self.state_store.load_state(job.pipeline_id)
                temp_scenario = load_scenario(temp_state.scenario_name) if temp_state else None

            if temp_scenario:
                temp_scenario.compute_dependencies()
                step_def = temp_scenario.steps.get(job.step_id)
                if step_def and not await self.scheduler.all_deps_complete(job.pipeline_id, step_def):
                    # Dependencies not satisfied - skip this stale job
                    logger.warning(f"Skipping {job.step_id}: dependencies not complete")
                    return

            # 1d. Mark step as RUNNING
            await self.state_store.set_step_status(
                job.pipeline_id, job.step_id, StepStatus.RUNNING
            )

            # 2. Check state type and route accordingly
            state_type = await self.state_store.get_state_type(job.pipeline_id)

            if state_type is None:
                print(f"Pipeline state not found: {job.pipeline_id}")
                return

            if state_type == 'dialogue':
                # Dialogue flow - load DialogueState and route to dialogue handler
                dialogue_state = await self.state_store.load_dialogue_state(job.pipeline_id)
                scenario = load_scenario(dialogue_state.strategy_name)
                persona = self.roster.personas.get(dialogue_state.persona_id)
                if persona is None:
                    await self.scheduler.mark_failed(
                        job.pipeline_id, job.step_id,
                        f"Persona {dialogue_state.persona_id} not found"
                    )
                    return
                await self._process_dialogue_step(job, scenario, persona)
                return

            # Standard flow - load PipelineState
            state = await self.state_store.load_state(job.pipeline_id)
            scenario = load_scenario(state.scenario_name)
            scenario.compute_dependencies()
            step_def = scenario.steps.get(job.step_id)

            if step_def is None:
                # Step not found in scenario - log and skip
                print(f"Step {job.step_id} not found in scenario {state.scenario_name}")
                return

            # 3. Check dependencies satisfied
            if not await self.scheduler.all_deps_complete(job.pipeline_id, step_def):
                # Dependencies not ready - requeue with short delay
                await self.scheduler.requeue_step(job, delay=5)
                return

            # Load persona
            persona = self.roster.personas.get(state.persona_id)

            if persona is None:
                # Persona not found - mark failed
                await self.scheduler.mark_failed(
                    job.pipeline_id,
                    job.step_id,
                    f"Persona {state.persona_id} not found"
                )
                return

            # 4. Execute step (standard flow)
            result, context_doc_ids, is_initial_context = await execute_step(
                state=state,
                scenario=scenario,
                step_def=step_def,
                cvm=self.cvm,
                persona=persona,
                config=self.config,
            )

            # 5. Persist result to CVM immediately
            message = create_message(state, step_def, result)
            self.cvm.insert(message)

            # 6. Update state with doc_id reference and accumulated context
            state.completed_steps.append(job.step_id)
            state.step_doc_ids[job.step_id] = result.doc_id
            state.step_counter += 1
            state.updated_at = datetime.utcnow()

            # Update accumulated context:
            # - If initial context (from DSL): set context_doc_ids to DSL result + step output
            # - Otherwise: append step output to existing context
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

        except RetryableError as e:
            # Retryable error - requeue with exponential backoff
            if job.attempt < job.max_attempts:
                delay = 30 * job.attempt  # 30s, 60s, 90s
                incremented_job = job.increment_attempt()
                await self.scheduler.requeue_step(incremented_job, delay=delay)
            else:
                # Max attempts reached - mark failed
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

        finally:
            # Always release lock
            await self.state_store.release_lock(job.pipeline_id, job.step_id)

    async def _process_dialogue_step(
        self,
        job: StepJob,
        scenario: 'Scenario',
        persona: 'Persona',
    ) -> None:
        """
        Process a dialogue flow step using DialogueScenario.

        Dialogue flows use a different execution model:
        - DialogueState instead of PipelineState
        - DialogueStrategy provides step definitions
        - DialogueScenario handles role flipping and turn building

        Args:
            job: StepJob to process
            scenario: Loaded Scenario with flow="dialogue"
            persona: Persona for the dialogue
        """
        # Load dialogue state
        dialogue_state = await self.state_store.load_dialogue_state(job.pipeline_id)

        if dialogue_state is None:
            print(f"Dialogue state not found: {job.pipeline_id}")
            await self.scheduler.mark_failed(
                job.pipeline_id,
                job.step_id,
                "Dialogue state not found"
            )
            return

        # Create DialogueStrategy and DialogueScenario
        strategy = DialogueStrategy.load(scenario.name)
        dialogue_scenario = DialogueScenario(
            strategy=strategy,
            persona=persona,
            config=self.config,
            cvm=self.cvm,
        )

        # Set the state (already initialized)
        dialogue_scenario.state = dialogue_state

        # Execute the step
        turn = await dialogue_scenario.execute_step(job.step_id)

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
            # Use scenario for completion check (same DAG structure)
            await self.scheduler.check_pipeline_complete(job.pipeline_id, scenario)

    def setup_signal_handlers(self) -> None:
        """Setup handlers for graceful shutdown.

        Registers signal handlers for SIGINT and SIGTERM to trigger
        graceful shutdown via stop().
        """
        def signal_handler(signum, frame):
            """Handle shutdown signals."""
            print(f"Received signal {signum}, shutting down gracefully...")
            # Create task to stop the worker
            asyncio.create_task(self.stop())

        # Register handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def run_worker(config: ChatConfig) -> None:
    """Entry point for running a dreamer worker.

    Creates Redis client, initializes StateStore and Scheduler,
    and starts the worker loop.

    Args:
        config: ChatConfig with Redis connection settings
    """
    import redis.asyncio as redis

    # Create Redis client
    redis_client = redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        db=config.redis_db,
        password=config.redis_password,
        decode_responses=False,  # We handle encoding/decoding
    )

    # Initialize StateStore and Scheduler
    state_store = StateStore(redis_client)
    scheduler = Scheduler(redis_client, state_store)

    # Create and start worker
    worker = DreamerWorker(config, state_store, scheduler)

    try:
        await worker.start()
    finally:
        # Cleanup Redis connection
        await redis_client.close()
