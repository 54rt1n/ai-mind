# andimud_worker/worker/dreamer.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Dreamer mixin for MUD worker.

Adds dream handling to MUDAgentWorker, including:
- reason="dream" turn processing

Dreams are special introspective turns where the agent processes
scenarios like journaling, analysis, or daydreaming instead of
responding to MUD events. The DreamerRunner executes pipelines
inline within the worker to prevent concurrent LLM calls.
"""

from typing import TYPE_CHECKING, Optional
import logging

from ..dreamer.runner import DreamerRunner, DreamRequest, DreamResult

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


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
