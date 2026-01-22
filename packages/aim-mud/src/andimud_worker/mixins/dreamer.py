# andimud_worker/worker/dreamer.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Dreamer mixin for MUD worker.

Adds dream handling to MUDAgentWorker, including:
- reason="dream" turn processing
- Step-by-step dream pipeline initialization and execution

Dreams are special introspective turns where the agent processes
scenarios like journaling, analysis, or daydreaming instead of
responding to MUD events. Dreams are executed inline within the
worker using the strategy-based ScenarioFramework system.
"""

from typing import TYPE_CHECKING, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import uuid
import logging

if TYPE_CHECKING:
    from aim_mud_types.coordination import DreamingState
    from aim.dreamer.core.framework import ScenarioFramework
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


# Scenarios that operate on the MUD conversation history
CONVERSATION_ANALYSIS_SCENARIOS = {"analysis_dialogue", "summarizer"}


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
    - initialize_scenario_dream(): Create new strategy-based dream
    - execute_scenario_step(): Execute one step of strategy-based scenario

    Expected attributes from MUDAgentWorker:
    - self.chat_config: ChatConfig
    - self.cvm: ConversationModel
    - self.roster: Roster
    - self.redis: Redis client
    - self.config: MUDConfig (has agent_id, persona_id)
    - self.conversation_manager: MUDConversationManager
    """

    async def process_dream_turn(
        self: "MUDAgentWorker",
        scenario: str,
        query: Optional[str] = None,
        guidance: Optional[str] = None,
        triggered_by: str = "manual",
        target_conversation_id: Optional[str] = None,
    ) -> DreamResult:
        """Process a dream turn using the strategy-based system.

        Called when turn_request.reason == "dream". Executes the specified
        scenario using ScenarioFramework + ScenarioState.

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
        from aim.dreamer.core.builder import load_scenario_framework
        from aim_mud_types.coordination import DreamStatus
        from ..conversation.storage import generate_conversation_id
        import time

        start_time = time.time()

        try:
            # 1. Load scenario framework (strategy-based system)
            framework = load_scenario_framework(scenario)

            # 2. Determine conversation_id
            if target_conversation_id:
                conv_id = target_conversation_id
            elif scenario in CONVERSATION_ANALYSIS_SCENARIOS:
                conv_id = self.conversation_manager.conversation_id
            else:
                prefix = scenario.split("_")[0]
                conv_id = generate_conversation_id(prefix)

            # 3. Initialize dream state
            dreaming_state = await self.initialize_scenario_dream(
                framework=framework,
                conversation_id=conv_id,
                guidance=guidance,
                query_text=query,
            )

            # 4. Mark as RUNNING
            dreaming_state.status = DreamStatus.RUNNING
            await self.save_dreaming_state(dreaming_state)

            logger.info(
                f"Starting dream: scenario={scenario} "
                f"conversation={conv_id} triggered_by={triggered_by}"
            )

            # 5. Execute all steps in a loop
            while True:
                is_complete = await self.execute_scenario_step(dreaming_state.pipeline_id)
                if is_complete:
                    break
                # Heartbeat between steps
                await self.atomic_heartbeat_update()

            duration = time.time() - start_time
            logger.info(
                f"Dream completed: pipeline={dreaming_state.pipeline_id} "
                f"duration={duration:.1f}s scenario={scenario}"
            )

            return DreamResult(
                success=True,
                pipeline_id=dreaming_state.pipeline_id,
                scenario=scenario,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Dream failed: scenario={scenario} "
                f"error={str(e)} duration={duration:.1f}s"
            )
            return DreamResult(
                success=False,
                scenario=scenario,
                error=str(e),
                duration_seconds=duration,
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
        MIN_DOCS_FOR_ANALYSIS = 6
        unanalyzed = []
        for _, row in report.iterrows():
            conv_id = row['conversation_id']
            total_docs = (
                row.get("conversation", 0) +
                row.get("mud-world", 0) +
                row.get("mud-action", 0) +
                row.get("mud-agent", 0)
            )
            has_enough_docs = total_docs >= MIN_DOCS_FOR_ANALYSIS
            has_analysis = row.get("analysis", 0) > 0
            has_summary = row.get("summary", 0) > 0

            if (has_enough_docs or has_summary) and not has_analysis:
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
        from aim.dreamer.core.builder import load_scenario_framework

        # If conversation already has summary, skip summarization
        report = self.cvm.get_conversation_report()
        conv_row = report[report['conversation_id'] == conversation_id]
        if not conv_row.empty and conv_row.iloc[0].get("summary", 0) > 0:
            logger.debug(f"Conversation {conversation_id} already has summary")
            return False

        # Load scenario framework and get the step that loads the conversation
        framework = load_scenario_framework("analysis_dialogue")
        step_def = framework.steps["ner_request"]
        step_config = step_def.config

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
            query_document_type=["mud-world", "mud-action", "mud-agent", "conversation"],
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
        """Initialize PENDING dream using the strategy-based system.

        Transitions PENDING -> RUNNING by loading framework and creating ScenarioState.
        PENDING stub already has scenario_name (+ conversation_id/query/guidance).
        """
        from aim.dreamer.core.builder import load_scenario_framework
        from aim.dreamer.core.state import ScenarioState
        from aim_mud_types.coordination import DreamStatus
        import json

        if not state.scenario_name:
            raise ValueError("PENDING dream missing scenario_name")

        # Load framework (strategy-based system)
        framework = load_scenario_framework(state.scenario_name)

        # Compute next branch for document creation
        branch = 0
        if state.conversation_id:
            branch = self.cvm.get_next_branch(state.conversation_id)

        # Create ScenarioState
        scenario_state = ScenarioState.initial(
            first_step=framework.first_step,
            conversation_id=state.conversation_id,
            guidance=state.guidance,
            query_text=state.query,
            branch=branch,
        )

        # Update dreaming state with serialized framework/state
        state.status = DreamStatus.RUNNING
        state.framework = json.dumps(framework.model_dump())
        state.state = json.dumps(scenario_state.model_dump())
        state.updated_at = datetime.now(timezone.utc)

        await self.save_dreaming_state(state)

        logger.info(
            f"Initialized PENDING -> RUNNING: {state.scenario_name} with "
            f"first_step='{framework.first_step}' (conversation={state.conversation_id})"
        )

        return state

    async def initialize_auto_dream(
        self: "MUDAgentWorker",
        decision: DreamDecision,
    ):
        """Initialize auto-analysis dream using the strategy-based system.

        Creates a RUNNING DreamingState and saves to Redis.
        """
        from aim.dreamer.core.builder import load_scenario_framework
        from aim_mud_types.coordination import DreamStatus

        # Load framework (strategy-based system)
        framework = load_scenario_framework(decision.scenario)

        # Use existing initialize_scenario_dream()
        state = await self.initialize_scenario_dream(
            framework=framework,
            conversation_id=decision.conversation_id,
            guidance=decision.guidance,
            query_text=decision.query,
        )

        # Mark as RUNNING immediately (skip PENDING for auto-analysis)
        state.status = DreamStatus.RUNNING
        await self.save_dreaming_state(state)

        logger.info(
            f"Auto-analysis initialized: {decision.scenario} with "
            f"first_step='{framework.first_step}' on conversation {decision.conversation_id}"
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
            dreaming_state.metadata = {
                "error_message": "Missing framework or state fields",
                "error_step_id": None,
            }
            await self.save_dreaming_state(dreaming_state)
            await self.archive_dreaming_state(dreaming_state)
            await self.delete_dreaming_state(self.config.agent_id)
            return True

        try:
            framework = ScenarioFramework.model_validate(dreaming_state.framework)
            scenario_state = ScenarioState.model_validate(dreaming_state.state)
        except Exception as e:
            logger.error(f"Failed to deserialize scenario state: {e}", exc_info=True)
            dreaming_state.status = DreamStatus.FAILED
            dreaming_state.metadata = {
                "error_message": f"Deserialization error: {e}",
                "error_traceback": traceback.format_exc(),
            }
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
            dreaming_state.metadata = {
                "error_message": f"Step '{current_step_id}' not found in framework",
                "error_step_id": current_step_id,
            }
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
            metadata = dreaming_state.metadata or {}
            metadata["error_step_id"] = current_step_id
            metadata["error_message"] = str(e)
            metadata["error_traceback"] = traceback.format_exc()

            # Update retry tracking
            dreaming_state.current_step_attempts += 1
            dreaming_state.last_error = str(e)

            if dreaming_state.current_step_attempts >= dreaming_state.max_step_retries:
                dreaming_state.status = DreamStatus.FAILED
                dreaming_state.completed_at = datetime.now(timezone.utc)
                dreaming_state.metadata = metadata
                await self.save_dreaming_state(dreaming_state)
                await self.archive_dreaming_state(dreaming_state)
                await self.delete_dreaming_state(self.config.agent_id)
                logger.error(f"Dream {pipeline_id} failed - max retries exhausted")
                return True

            # Schedule retry with exponential backoff
            backoff_seconds = 60 * (2 ** (dreaming_state.current_step_attempts - 1))
            dreaming_state.next_retry_at = datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds)
            dreaming_state.metadata = metadata
            await self.save_dreaming_state(dreaming_state)

            logger.warning(
                f"Dream step '{current_step_id}' failed, retry in {backoff_seconds}s "
                f"(attempt {dreaming_state.current_step_attempts}/{dreaming_state.max_step_retries})"
            )
            return False  # Not complete, will retry

        # 6. Save updated state
        dreaming_state.state = scenario_state.model_dump()
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

        # Compute next branch for document creation
        branch = 0
        if conversation_id:
            branch = self.cvm.get_next_branch(conversation_id)

        # Create initial ScenarioState
        scenario_state = ScenarioState.initial(
            first_step=framework.first_step,
            conversation_id=conversation_id,
            guidance=guidance,
            query_text=query_text,
            branch=branch,
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
            framework=framework.model_dump(),
            state=scenario_state.model_dump(),
        )

        await self.save_dreaming_state(dreaming_state)

        logger.info(
            f"Initialized strategy-based scenario '{framework.name}' "
            f"with first_step='{framework.first_step}' "
            f"(pipeline_id={dreaming_state.pipeline_id})"
        )

        return dreaming_state
