# andimud_worker/dreamer_runner.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Inline Dreamer pipeline execution for MUD workers.

This module provides DreamerRunner, which executes Dreamer pipelines
synchronously within the MUD worker process using the inline scheduler.
This prevents concurrent LLM calls between regular turns and dream processing.

The inline scheduler executes pipelines in-process without Redis queues,
state stores, or distributed infrastructure - perfect for ANDIMUD's
single-agent-per-worker architecture.
"""

from dataclasses import dataclass
from typing import Optional, Callable, Awaitable
import logging
import time

from aim.config import ChatConfig
from aim.conversation.model import ConversationModel
from aim.agents.roster import Roster
from aim.dreamer.inline import execute_pipeline_inline
from andimud_worker.conversation.storage import generate_conversation_id

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

    This class executes Dreamer pipelines synchronously using the inline
    scheduler. It prevents concurrent LLM calls between regular turns and
    dream processing by blocking until the pipeline completes.

    The inline scheduler executes pipelines in-process without Redis queues,
    state stores, or distributed infrastructure - perfect for ANDIMUD's
    single-agent-per-worker architecture.

    Example:
        runner = DreamerRunner(
            config=config,
            cvm=cvm,
            roster=roster,
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
        persona_id: str,
    ):
        """Initialize the DreamerRunner.

        Args:
            config: ChatConfig with provider keys and settings
            cvm: ConversationModel for memory operations (shared with MUD worker)
            roster: Roster with persona definitions
            persona_id: Persona ID to use for dreams
        """
        self.config = config
        self.cvm = cvm
        self.roster = roster
        self.persona_id = persona_id

        logger.info(f"DreamerRunner initialized for {persona_id}")

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
            # Standalone scenarios generate unique IDs per execution
            # Extract prefix from scenario name (e.g., "journaler_dialogue" -> "journaler")
            prefix = scenario.split("_")[0]
            return generate_conversation_id(prefix)

    async def run_dream(
        self,
        request: DreamRequest,
        base_conversation_id: str,
        heartbeat_callback: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> DreamResult:
        """Execute a complete dream pipeline inline.

        This method uses the inline scheduler to execute the pipeline
        synchronously, blocking until completion. All state is held in
        memory - no Redis queues or state stores needed.

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
            # Determine conversation ID
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

            # Wrap heartbeat_callback to match inline scheduler signature
            # Inline scheduler expects callback(pipeline_id, step_id)
            # Pass through pipeline/step context to heartbeat callback
            wrapped_heartbeat = None
            if heartbeat_callback:
                async def wrapped_heartbeat(pipeline_id: str, step_id: str) -> None:
                    await heartbeat_callback(pipeline_id, step_id)

            # Execute pipeline inline
            pipeline_id = await execute_pipeline_inline(
                scenario_name=request.scenario,
                config=self.config,
                cvm=self.cvm,
                roster=self.roster,
                persona_id=self.persona_id,
                conversation_id=target_conversation_id,
                query_text=request.query,
                guidance=request.guidance,
                heartbeat_callback=wrapped_heartbeat,
            )

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
