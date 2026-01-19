# andimud_worker/worker/dreamer.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Dreamer mixin for MUD worker.

Adds dream handling to MUDAgentWorker, including:
- reason="dream" turn processing
- Step-by-step dream pipeline initialization and execution

Dreams are special introspective turns where the agent processes
scenarios like journaling, analysis, or daydreaming instead of
responding to MUD events. The DreamerRunner executes pipelines
inline within the worker to prevent concurrent LLM calls.
"""

from typing import TYPE_CHECKING, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import uuid
import logging

from ..dreamer.runner import DreamerRunner, DreamRequest, DreamResult

if TYPE_CHECKING:
    from aim_mud_types.coordination import DreamingState
    from aim.dreamer.core.framework import ScenarioFramework
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


@dataclass
class DreamDecision:
    """Worker's decision about what to dream."""
    scenario: str  # "analysis_dialogue" or "summarizer"
    conversation_id: str
    query: Optional[str] = None
    guidance: Optional[str] = None


class DreamerMixin:
    """Mixin adding dreamer capabilities to MUDAgentWorker.

    This mixin provides:
    - process_dream_turn(): Handle reason="dream" turns

    Expected attributes from MUDAgentWorker:
    - self.chat_config: ChatConfig
    - self.cvm: ConversationModel
    - self.roster: Roster
    - self.redis: Redis client
    - self.config: MUDConfig (has agent_id, persona_id)
    - self.conversation_manager: MUDConversationManager
    """

    _dreamer_runner: Optional[DreamerRunner] = None

    def _init_dreamer(self: "MUDAgentWorker") -> None:
        """Initialize the DreamerRunner.

        Called lazily on first dream request to avoid initialization
        overhead if dreams are never used.
        """
        self._dreamer_runner = DreamerRunner(
            config=self.chat_config,
            cvm=self.cvm,
            roster=self.roster,
            persona_id=self.config.persona_id,
        )
        logger.info(f"Initialized DreamerRunner for {self.config.persona_id}")

    async def process_dream_turn(
        self: "MUDAgentWorker",
        scenario: str,
        query: Optional[str] = None,
        guidance: Optional[str] = None,
        triggered_by: str = "manual",
        target_conversation_id: Optional[str] = None,
    ) -> DreamResult:
        """Process a dream turn.

        Called when turn_request.reason == "dream". Executes the specified
        scenario inline using the DreamerRunner.

        Args:
            scenario: Name of scenario YAML to run (e.g., "analysis_dialogue")
            query: Optional query text for scenarios that use it
            guidance: Optional guidance text to influence generation
            triggered_by: How dream was triggered ("manual" or "auto")
            target_conversation_id: Explicit conversation ID for analysis commands.
                If provided, the dream will analyze this conversation.
                If None, uses the current MUD conversation for analysis scenarios,
                or a standalone conversation for creative scenarios.

        Returns:
            DreamResult with success status and metadata
        """
        if not self._dreamer_runner:
            self._init_dreamer()

        request = DreamRequest(
            scenario=scenario,
            query=query,
            guidance=guidance,
            triggered_by=triggered_by,
            target_conversation_id=target_conversation_id,
        )

        # Get MUD conversation ID as fallback for analysis scenarios
        base_conversation_id = self.conversation_manager.conversation_id

        # Create heartbeat callback that refreshes turn_request heartbeat
        async def heartbeat(pipeline_id: str, step_id: str) -> None:
            """Refresh turn request heartbeat during long-running dream steps.

            Uses atomic update to prevent partial hash creation during shutdown.
            """
            result = await self.atomic_heartbeat_update()

            if result == 0:
                logger.debug("Turn request deleted during dream, stopping heartbeat")
            elif result == -1:
                logger.error("Corrupted turn_request during dream heartbeat")

        result = await self._dreamer_runner.run_dream(
            request, base_conversation_id, heartbeat
        )

        return result

    async def initialize_dream_pipeline(
        self: "MUDAgentWorker",
        scenario_name: str,
        conversation_id: str,
        query: Optional[str] = None,
        guidance: Optional[str] = None,
    ) -> "DreamingState":
        """
        Initialize a new dream pipeline for step-by-step execution.

        Creates DreamingState with execution order pre-computed.
        Returns state ready to execute first step.

        Args:
            scenario_name: Name of scenario YAML to run (e.g., "analysis_dialogue")
            conversation_id: Target conversation ID
            query: Optional query text for scenarios that use it
            guidance: Optional guidance text to influence generation

        Returns:
            DreamingState ready for step execution
        """
        from aim.dreamer.core.scenario import load_scenario
        from aim.dreamer.core.memory_dsl import execute_memory_actions
        from aim_mud_types.coordination import DreamingState, DreamStatus

        # Load scenario and persona
        scenario = load_scenario(scenario_name)
        persona = self.roster.get_persona(self.config.persona_id)

        # Create ModelSet to resolve base_model
        from aim.llm.model_set import ModelSet
        model_set = ModelSet.from_config(self.chat_config, persona)

        # Compute execution order
        if scenario.flow == "dialogue":
            from aim.dreamer.core.dialogue.strategy import DialogueStrategy

            strategy = DialogueStrategy.load(scenario_name)
            execution_order = strategy.get_execution_order()
        else:
            scenario.compute_dependencies()
            execution_order = scenario.topological_order()

        # Execute seed memory actions (one-time setup)
        # These are NOT steps, they initialize context only
        seed_context_ids = []
        if scenario.seed:
            from aim.dreamer.core.models import PipelineState

            logger.info(f"Executing {len(scenario.seed)} seed actions")

            # Create minimal state for seed actions (needs conversation_id for load_conversation)
            seed_state = PipelineState(
                pipeline_id=str(uuid.uuid4()),  # Temporary ID
                scenario_name=scenario_name,
                conversation_id=conversation_id,
                persona_id=persona.persona_id,
                user_id="user",
                model=model_set.default_model,
                branch=0,
                step_counter=0,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            seed_context_ids = execute_memory_actions(
                actions=scenario.seed,
                state=seed_state,
                cvm=self.cvm,
                query_text=query,
            )
            logger.info(f"Seed actions produced {len(seed_context_ids)} documents")

        # Create state
        state = DreamingState(
            pipeline_id=str(uuid.uuid4()),
            agent_id=self.config.agent_id,
            status=DreamStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            scenario_name=scenario_name,
            execution_order=execution_order,
            query=query,
            guidance=guidance,
            conversation_id=conversation_id,
            base_model=model_set.default_model,
            step_index=0,
            completed_steps=[],
            step_doc_ids={},
            context_doc_ids=seed_context_ids,  # Seed context available for first step
            current_step_attempts=0,
            max_step_retries=3,
            scenario_config=scenario.model_dump(),  # Freeze config at init time
            persona_config=persona.to_dict(),  # Freeze persona at init time
        )

        # Save to Redis
        await self.save_dreaming_state(state)

        logger.info(
            f"Initialized dream pipeline {state.pipeline_id} for {scenario_name} "
            f"with {len(execution_order)} steps"
        )

        return state

    async def execute_dream_step(
        self: "MUDAgentWorker",
        state: "DreamingState",
    ):
        """
        Execute the next step in the dream pipeline.

        Args:
            state: Current DreamingState

        Returns:
            CommandResult indicating success/failure
        """
        from aim.dreamer.core.models import Scenario, PipelineState
        from aim.agents.persona import Persona
        from aim.dreamer.core.executor import execute_step, create_message
        from aim.llm.model_set import ModelSet
        from aim_mud_types.coordination import DreamStatus
        from aim_mud_types import TurnRequestStatus
        from ..commands.result import CommandResult

        if state.step_index >= len(state.execution_order):
            logger.warning(f"Dream {state.pipeline_id} already complete")
            # Archive and delete
            await self.archive_dreaming_state(state)
            await self.delete_dreaming_state(self.config.agent_id)
            return CommandResult(
                complete=True,
                flush_drain=True,
                saved_event_id=None,
                status=TurnRequestStatus.DONE,
                message="Dream already complete",
            )

        step_id = state.execution_order[state.step_index]

        logger.info(
            f"Executing dream step {state.step_index + 1}/{len(state.execution_order)}: "
            f"{step_id} (attempt {state.current_step_attempts + 1})"
        )

        try:
            # Reconstruct scenario and persona from frozen config
            scenario = Scenario(**state.scenario_config)
            persona = Persona(**state.persona_config)

            # Create ModelSet for persona-aware model selection
            model_set = ModelSet.from_config(self.chat_config, persona)

            # Build PipelineState for this step
            pipeline_state = PipelineState(
                pipeline_id=state.pipeline_id,
                scenario_name=state.scenario_name,
                conversation_id=state.conversation_id,
                persona_id=persona.persona_id,
                user_id="user",  # Default user_id for MUD dreams
                model=state.base_model,
                thought_model=model_set.thought_model,
                codex_model=model_set.codex_model,
                guidance=state.guidance,
                query_text=state.query,
                persona_mood=None,
                branch=0,  # Dreams use branch 0
                step_counter=state.step_index + 1,
                created_at=state.created_at,
                updated_at=datetime.now(timezone.utc),
                completed_steps=state.completed_steps,
                step_doc_ids=state.step_doc_ids,
                context_doc_ids=state.context_doc_ids,
            )

            # Get step definition
            step_def = scenario.steps[step_id]

            # Create heartbeat callback
            async def heartbeat_callback(pipeline_id: str, step_id: str) -> None:
                """Refresh heartbeat during long-running step."""
                await self.update_dreaming_heartbeat(state.agent_id)

            # Execute step (reuses existing function - identical behavior)
            step_result, context_doc_ids, is_initial_context = await execute_step(
                state=pipeline_state,
                scenario=scenario,
                step_def=step_def,
                cvm=self.cvm,
                persona=persona,
                config=self.chat_config,
                model_set=model_set,
                heartbeat_callback=heartbeat_callback,
            )

            # Update accumulated context if this was initial context from DSL
            if is_initial_context:
                state.context_doc_ids = context_doc_ids

            # Create and save message to CVM
            message = create_message(pipeline_state, step_def, step_result)
            self.cvm.insert(message)

            # Update state with results
            state.completed_steps.append(step_id)
            state.step_doc_ids[step_id] = step_result.doc_id

            # Append step output to accumulated context
            if step_result.doc_id not in state.context_doc_ids:
                state.context_doc_ids.append(step_result.doc_id)

            state.step_index += 1
            state.current_step_attempts = 0  # Reset retry counter on success
            state.next_retry_at = None
            state.last_error = None
            state.updated_at = datetime.now(timezone.utc)

            # Check if pipeline complete
            if state.step_index >= len(state.execution_order):
                state.status = DreamStatus.COMPLETE
                state.completed_at = datetime.now(timezone.utc)
                await self.save_dreaming_state(state)

                logger.info(
                    f"Dream pipeline complete: {state.scenario_name} "
                    f"on {state.conversation_id}"
                )

                # Archive and delete
                await self.archive_dreaming_state(state)
                await self.delete_dreaming_state(self.config.agent_id)

                # Refresh CVM index
                self.cvm.refresh()

                return CommandResult(
                    complete=True,
                    flush_drain=True,
                    saved_event_id=None,
                    status=TurnRequestStatus.DONE,
                    message=f"Dream complete: {state.scenario_name}",
                )
            else:
                state.status = DreamStatus.RUNNING

            # Save updated state
            await self.save_dreaming_state(state)

            # Refresh CVM index (same as inline execution)
            self.cvm.refresh()

            return CommandResult(
                complete=True,
                flush_drain=True,
                saved_event_id=None,
                status=TurnRequestStatus.DONE,
                message=f"Dream step {state.step_index} completed",
            )

        except Exception as e:
            logger.error(f"Dream step {step_id} failed: {e}", exc_info=True)

            state.current_step_attempts += 1
            state.last_error = str(e)
            state.updated_at = datetime.now(timezone.utc)

            # Check retry limit
            if state.current_step_attempts >= state.max_step_retries:
                state.status = DreamStatus.FAILED
                state.completed_at = datetime.now(timezone.utc)
                await self.save_dreaming_state(state)

                logger.error(
                    f"Dream {state.pipeline_id} failed - max retries exhausted for step {step_id}"
                )

                # Archive and delete
                await self.archive_dreaming_state(state)
                await self.delete_dreaming_state(self.config.agent_id)

                return CommandResult(
                    complete=True,
                    flush_drain=True,
                    saved_event_id=None,
                    status=TurnRequestStatus.DONE,
                    message=f"Dream failed: {str(e)}",
                )

            # Schedule retry with exponential backoff
            backoff_seconds = 60 * (2 ** (state.current_step_attempts - 1))  # 60s, 120s, 240s
            state.next_retry_at = datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds)

            logger.warning(
                f"Dream step {step_id} failed, will retry in {backoff_seconds}s "
                f"(attempt {state.current_step_attempts}/{state.max_step_retries})"
            )

            await self.save_dreaming_state(state)
            return CommandResult(
                complete=True,
                flush_drain=True,
                saved_event_id=None,
                status=TurnRequestStatus.DONE,
                message=f"Dream step failed, retry scheduled",
            )

    async def _decide_dream_action(self: "MUDAgentWorker") -> Optional[DreamDecision]:
        """Decide if/what to dream based on agent state.

        Returns DreamDecision if agent wants to dream, None otherwise.

        Decision logic:
        1. Query conversation reports for unanalyzed conversations
        2. Select oldest unanalyzed conversation
        3. Check if context is too large (needs summarization)
        4. Return decision or None
        """

        # Load conversation report
        report = self.cvm.get_conversation_report()

        if report.empty:
            logger.debug("No conversations to analyze")
            return None

        # Find unanalyzed conversations
        unanalyzed = []
        for _, row in report.iterrows():
            conv_id = row['conversation_id']
            has_docs = (
                row.get("conversation", 0) > 0 or
                row.get("mud-world", 0) > 0 or
                row.get("mud-agent", 0) > 0
            )
            has_analysis = row.get("analysis", 0) > 0
            has_summary = row.get("summary", 0) > 0

            if (has_docs or has_summary) and not has_analysis:
                timestamp = row.get("timestamp_max", "")
                unanalyzed.append((conv_id, timestamp))

        if not unanalyzed:
            logger.debug("All conversations analyzed")
            return None

        # Select oldest unanalyzed conversation
        unanalyzed.sort(key=lambda x: x[1])
        conversation_id, _ = unanalyzed[0]

        logger.info(
            f"Found {len(unanalyzed)} unanalyzed conversation(s), "
            f"selecting oldest: {conversation_id}"
        )

        # Check if context is too large
        should_summarize = await self._should_summarize_conversation(
            conversation_id=conversation_id
        )

        scenario = "summarizer" if should_summarize else "analysis_dialogue"

        return DreamDecision(
            scenario=scenario,
            conversation_id=conversation_id
        )

    async def _should_summarize_conversation(
        self: "MUDAgentWorker",
        conversation_id: str
    ) -> bool:
        """Check if conversation needs summarization before analysis.

        Logic moved from mediator to worker.

        Returns:
            True if conversation should be summarized first, False if ready for analysis.
        """
        from aim.utils.tokens import count_tokens
        from aim.llm.models import LanguageModelV2
        from aim.llm.model_set import ModelSet
        from aim.dreamer.core.scenario import load_scenario

        # If conversation already has summary, skip summarization
        report = self.cvm.get_conversation_report()
        conv_row = report[report['conversation_id'] == conversation_id]
        if not conv_row.empty and conv_row.iloc[0].get("summary", 0) > 0:
            logger.debug(f"Conversation {conversation_id} already has summary")
            return False

        # Load scenario and get the step that loads the conversation
        scenario = load_scenario("analysis_dialogue")
        step_config = scenario.steps["ner_request"].config

        # Create ModelSet and resolve actual model for this step
        persona = self.roster.get_persona(self.config.persona_id)
        model_set = ModelSet.from_config(self.chat_config, persona)
        model_name = step_config.get_model(model_set)

        # Get model context window
        model_index = LanguageModelV2.index_models(self.chat_config)
        model = model_index.get(model_name)
        if not model:
            logger.warning(f"Model {model_name} not found")
            return False

        max_tokens = model.max_tokens

        # Query documents from conversation (exclude analysis/summary)
        docs_df = self.cvm.get_conversation_history(
            conversation_id=conversation_id,
            query_document_type=["mud-world", "mud-agent", "conversation"],
        )
        if docs_df.empty:
            return False

        # Count tokens (matches watcher pattern)
        token_count = sum(
            count_tokens(str(content))
            for content in docs_df['content'].fillna("").tolist()
        )

        # Use 75% of context window as threshold
        threshold = int(max_tokens * 0.75)

        logger.debug(
            f"Conversation {conversation_id}: {token_count} tokens "
            f"(threshold: {threshold}, max: {max_tokens})"
        )

        if token_count > threshold:
            logger.info(
                f"Conversation {conversation_id} exceeds threshold "
                f"({token_count} > {threshold}), will summarize first"
            )
            return True
        else:
            logger.info(
                f"Conversation {conversation_id} under threshold "
                f"({token_count} <= {threshold}), ready for analysis"
            )
            return False

    async def initialize_pending_dream(
        self: "MUDAgentWorker",
        state: "DreamingState",
    ):
        """Initialize PENDING dream from manual command.

        Transitions PENDING → RUNNING by loading scenario and computing execution order.
        PENDING stub already has scenario_name (+ conversation_id/query/guidance).
        """
        from aim.dreamer.core.scenario import load_scenario
        from aim_mud_types.coordination import DreamStatus

        # scenario_name is already set by manual command
        if not state.scenario_name:
            raise ValueError("PENDING dream missing scenario_name")

        # Load scenario
        scenario = load_scenario(state.scenario_name)
        persona = self.roster.get_persona(self.config.persona_id)

        # Create ModelSet to resolve base_model
        from aim.llm.model_set import ModelSet
        model_set = ModelSet.from_config(self.chat_config, persona)

        # Compute execution order
        if scenario.flow == "dialogue":
            from aim.dreamer.core.dialogue.strategy import DialogueStrategy
            strategy = DialogueStrategy.load(state.scenario_name)
            execution_order = strategy.get_execution_order()
        else:
            execution_order = scenario.topological_order()

        # Update state: PENDING → RUNNING
        state.status = DreamStatus.RUNNING
        state.execution_order = execution_order
        state.base_model = model_set.default_model
        state.scenario_config = scenario.model_dump()
        state.persona_config = persona.to_dict()
        state.step_index = 0
        state.completed_steps = []
        state.step_doc_ids = {}
        state.context_doc_ids = []
        state.updated_at = datetime.now(timezone.utc)

        await self.save_dreaming_state(state)

        logger.info(
            f"Initialized PENDING → RUNNING: {state.scenario_name} with "
            f"{len(execution_order)} steps (conversation={state.conversation_id}, query={state.query})"
        )

        return state

    async def initialize_auto_dream(
        self: "MUDAgentWorker",
        decision: DreamDecision,
    ):
        """Initialize dream from auto-analysis decision.

        Creates a new RUNNING DreamingState (skips PENDING) and saves to Redis.
        """
        from aim.dreamer.core.scenario import load_scenario
        from aim_mud_types.coordination import DreamingState, DreamStatus

        # Load scenario
        scenario = load_scenario(decision.scenario)
        persona = self.roster.get_persona(self.config.persona_id)

        # Compute execution order
        if scenario.flow == "dialogue":
            from aim.dreamer.core.dialogue.strategy import DialogueStrategy
            strategy = DialogueStrategy.load(decision.scenario)
            execution_order = strategy.get_execution_order()
        else:
            execution_order = scenario.topological_order()

        # Create ModelSet to resolve base_model
        from aim.llm.model_set import ModelSet
        model_set = ModelSet.from_config(self.chat_config, persona)

        # Create new RUNNING state (skip PENDING)
        state = DreamingState(
            pipeline_id=str(uuid.uuid4()),
            agent_id=self.config.agent_id,
            status=DreamStatus.RUNNING,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            scenario_name=decision.scenario,
            conversation_id=decision.conversation_id,
            query=decision.query,
            guidance=decision.guidance,
            execution_order=execution_order,
            base_model=model_set.default_model,
            scenario_config=scenario.model_dump(),
            persona_config=persona.to_dict(),
            step_index=0,
            completed_steps=[],
            step_doc_ids={},
            context_doc_ids=[],
        )

        await self.save_dreaming_state(state)

        logger.info(
            f"Auto-analysis initialized: {decision.scenario} with "
            f"{len(execution_order)} steps on conversation {decision.conversation_id}"
        )

        return state

    async def execute_scenario_step(
        self: "MUDAgentWorker",
        pipeline_id: str,
    ) -> bool:
        """Execute one step of a strategy-based scenario.

        This method executes scenarios using the new step type strategy pattern
        (ScenarioFramework + ScenarioState + StepFactory). It loads serialized
        framework/state from Redis, executes one step via the appropriate strategy,
        and saves updated state back to Redis.

        The "loop" is external - repeated idle turns resume where the previous
        step left off. This is why both framework AND state are persisted.

        Args:
            pipeline_id: Identifier for the active dream in Redis

        Returns:
            True if scenario is complete (reached 'end' or 'abort'), False otherwise
        """
        import json
        import traceback

        from aim.dreamer.core.framework import ScenarioFramework
        from aim.dreamer.core.state import ScenarioState
        from aim.dreamer.core.strategy import ScenarioExecutor, StepFactory
        from aim_mud_types.coordination import DreamStatus

        # 1. Load DreamingState from Redis
        dreaming_state = await self.load_dreaming_state(self.config.agent_id)
        if not dreaming_state:
            logger.error(f"No dreaming state found for agent {self.config.agent_id}")
            return True  # Consider complete if no state

        # Verify pipeline_id matches
        if dreaming_state.pipeline_id != pipeline_id:
            logger.error(
                f"Pipeline ID mismatch: expected {pipeline_id}, "
                f"got {dreaming_state.pipeline_id}"
            )
            return True

        # Check if already complete
        if dreaming_state.status in (DreamStatus.COMPLETE, DreamStatus.FAILED, DreamStatus.ABORTED):
            logger.info(f"Dream {pipeline_id} already in terminal state: {dreaming_state.status}")
            return True

        # 2. Deserialize framework and state from JSON
        if not dreaming_state.framework or not dreaming_state.state:
            logger.error(
                f"Dream {pipeline_id} missing framework or state JSON - "
                f"this is a strategy-based scenario but has no serialized state"
            )
            # Mark as failed
            dreaming_state.status = DreamStatus.FAILED
            dreaming_state.metadata = json.dumps({
                "error_message": "Missing framework or state JSON fields",
                "error_step_id": None,
            })
            await self.save_dreaming_state(dreaming_state)
            await self.archive_dreaming_state(dreaming_state)
            await self.delete_dreaming_state(self.config.agent_id)
            return True

        try:
            framework = ScenarioFramework.model_validate(
                json.loads(dreaming_state.framework)
            )
            scenario_state = ScenarioState.model_validate(
                json.loads(dreaming_state.state)
            )
        except Exception as e:
            logger.error(f"Failed to deserialize scenario state: {e}", exc_info=True)
            dreaming_state.status = DreamStatus.FAILED
            dreaming_state.metadata = json.dumps({
                "error_message": f"Deserialization error: {e}",
                "error_traceback": traceback.format_exc(),
            })
            await self.save_dreaming_state(dreaming_state)
            await self.archive_dreaming_state(dreaming_state)
            await self.delete_dreaming_state(self.config.agent_id)
            return True

        # Check if scenario already complete
        if scenario_state.is_complete():
            logger.info(f"Scenario already complete (current_step={scenario_state.current_step})")
            dreaming_state.status = (
                DreamStatus.COMPLETE if scenario_state.current_step == "end"
                else DreamStatus.ABORTED
            )
            dreaming_state.completed_at = datetime.now(timezone.utc)
            await self.save_dreaming_state(dreaming_state)
            await self.archive_dreaming_state(dreaming_state)
            await self.delete_dreaming_state(self.config.agent_id)
            return True

        # 3. Create ScenarioExecutor
        persona = self.roster.get_persona(self.config.persona_id)

        # Create heartbeat callback
        async def heartbeat_callback() -> None:
            """Refresh heartbeat during long-running LLM calls."""
            await self.update_dreaming_heartbeat(dreaming_state.agent_id)
            # Also refresh turn request heartbeat
            await self.atomic_heartbeat_update()

        executor = ScenarioExecutor.create(
            state=scenario_state,
            framework=framework,
            config=self.chat_config,
            cvm=self.cvm,
            persona=persona,
            heartbeat_callback=heartbeat_callback,
        )

        # 4. Get current step and create strategy
        current_step_id = scenario_state.current_step
        if current_step_id not in framework.steps:
            logger.error(f"Current step '{current_step_id}' not found in framework")
            dreaming_state.status = DreamStatus.FAILED
            dreaming_state.metadata = json.dumps({
                "error_message": f"Step '{current_step_id}' not found in framework",
                "error_step_id": current_step_id,
            })
            await self.save_dreaming_state(dreaming_state)
            await self.archive_dreaming_state(dreaming_state)
            await self.delete_dreaming_state(self.config.agent_id)
            return True

        step_def = framework.steps[current_step_id]
        strategy = StepFactory.create(executor, step_def)

        logger.info(
            f"Executing scenario step '{current_step_id}' "
            f"(type={step_def.type}) for dream {pipeline_id}"
        )

        # 5. Execute strategy
        try:
            result = await executor.execute(strategy)

            logger.info(
                f"Step '{current_step_id}' complete: success={result.success}, "
                f"next_step='{result.next_step}', doc_created={result.doc_created}"
            )

            # executor.state has been mutated - scenario_state is the same object
            # result.next_step was already written to scenario_state.current_step

        except Exception as e:
            logger.error(f"Strategy execution failed for step '{current_step_id}': {e}", exc_info=True)

            # Record error in metadata
            metadata = json.loads(dreaming_state.metadata) if dreaming_state.metadata else {}
            metadata["error_step_id"] = current_step_id
            metadata["error_message"] = str(e)
            metadata["error_traceback"] = traceback.format_exc()

            # Update retry tracking
            dreaming_state.current_step_attempts += 1
            dreaming_state.last_error = str(e)

            if dreaming_state.current_step_attempts >= dreaming_state.max_step_retries:
                dreaming_state.status = DreamStatus.FAILED
                dreaming_state.completed_at = datetime.now(timezone.utc)
                dreaming_state.metadata = json.dumps(metadata)
                await self.save_dreaming_state(dreaming_state)
                await self.archive_dreaming_state(dreaming_state)
                await self.delete_dreaming_state(self.config.agent_id)
                logger.error(f"Dream {pipeline_id} failed - max retries exhausted")
                return True

            # Schedule retry with exponential backoff
            backoff_seconds = 60 * (2 ** (dreaming_state.current_step_attempts - 1))
            dreaming_state.next_retry_at = datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds)
            dreaming_state.metadata = json.dumps(metadata)
            await self.save_dreaming_state(dreaming_state)

            logger.warning(
                f"Dream step '{current_step_id}' failed, retry in {backoff_seconds}s "
                f"(attempt {dreaming_state.current_step_attempts}/{dreaming_state.max_step_retries})"
            )
            return False  # Not complete, will retry

        # 6. Serialize and save updated state
        dreaming_state.state = json.dumps(scenario_state.model_dump())
        dreaming_state.updated_at = datetime.now(timezone.utc)
        dreaming_state.current_step_attempts = 0  # Reset on success
        dreaming_state.next_retry_at = None
        dreaming_state.last_error = None

        # 7. Update dream status based on result
        if scenario_state.is_complete():
            dreaming_state.status = (
                DreamStatus.COMPLETE if scenario_state.current_step == "end"
                else DreamStatus.ABORTED
            )
            dreaming_state.completed_at = datetime.now(timezone.utc)
            await self.save_dreaming_state(dreaming_state)
            await self.archive_dreaming_state(dreaming_state)
            await self.delete_dreaming_state(self.config.agent_id)

            # Refresh CVM index
            self.cvm.refresh()

            logger.info(
                f"Strategy-based scenario complete: {framework.name} "
                f"(status={scenario_state.current_step})"
            )
            return True
        else:
            dreaming_state.status = DreamStatus.RUNNING
            await self.save_dreaming_state(dreaming_state)

            logger.debug(
                f"Scenario step complete, next step: {scenario_state.current_step}"
            )
            return False

    async def initialize_scenario_dream(
        self: "MUDAgentWorker",
        framework: "ScenarioFramework",
        conversation_id: Optional[str] = None,
        guidance: Optional[str] = None,
        query_text: Optional[str] = None,
    ) -> "DreamingState":
        """Initialize a strategy-based scenario dream.

        Creates a DreamingState with serialized ScenarioFramework and ScenarioState.
        This is used for new-style scenarios that use the step type strategy pattern.

        Args:
            framework: The ScenarioFramework to execute
            conversation_id: Target conversation (optional)
            guidance: External guidance text (optional)
            query_text: Query/topic text (optional)

        Returns:
            DreamingState ready for step-by-step execution via execute_scenario_step()
        """
        import json
        from aim.dreamer.core.state import ScenarioState
        from aim_mud_types.coordination import DreamingState, DreamStatus

        # Create initial ScenarioState
        scenario_state = ScenarioState.initial(
            first_step=framework.first_step,
            conversation_id=conversation_id,
            guidance=guidance,
            query_text=query_text,
        )

        persona = self.roster.get_persona(self.config.persona_id)

        # Create ModelSet to get base_model
        from aim.llm.model_set import ModelSet
        model_set = ModelSet.from_config(self.chat_config, persona)

        # Create DreamingState with serialized framework and state
        dreaming_state = DreamingState(
            pipeline_id=str(uuid.uuid4()),
            agent_id=self.config.agent_id,
            status=DreamStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            scenario_name=framework.name,
            execution_order=[],  # Not used for strategy-based scenarios
            query=query_text,
            guidance=guidance,
            conversation_id=conversation_id or "",
            base_model=model_set.default_model,
            step_index=0,  # Not used for strategy-based scenarios
            completed_steps=[],  # Not used for strategy-based scenarios
            step_doc_ids={},  # Not used for strategy-based scenarios
            context_doc_ids=[],  # Not used for strategy-based scenarios
            scenario_config={},  # Not used for strategy-based scenarios
            persona_config={},  # Not used for strategy-based scenarios
            # Strategy-based scenario fields
            framework=json.dumps(framework.model_dump()),
            state=json.dumps(scenario_state.model_dump()),
        )

        await self.save_dreaming_state(dreaming_state)

        logger.info(
            f"Initialized strategy-based scenario '{framework.name}' "
            f"with first_step='{framework.first_step}' "
            f"(pipeline_id={dreaming_state.pipeline_id})"
        )

        return dreaming_state
