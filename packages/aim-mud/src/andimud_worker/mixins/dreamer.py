# andimud_worker/worker/dreamer.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Dreamer mixin for MUD worker.

Adds dream handling to MUDAgentWorker, including:
- reason="dream" turn processing
- @dreamer on/off toggle support
- Automatic dream trigger checking during idle turns

Dreams are special introspective turns where the agent processes
scenarios like journaling, analysis, or daydreaming instead of
responding to MUD events. The DreamerRunner executes pipelines
inline within the worker to prevent concurrent LLM calls.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional
import logging

from aim_mud_types import RedisKeys

from ..dreamer.runner import DreamerRunner, DreamRequest, DreamResult

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class DreamerMixin:
    """Mixin adding dreamer capabilities to MUDAgentWorker.

    This mixin provides:
    - process_dream_turn(): Handle reason="dream" turns
    - check_auto_dream_triggers(): Check if auto-dream should trigger during idle

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

        # Create heartbeat callback that refreshes turn_request TTL
        async def heartbeat(pipeline_id: str, step_id: str) -> None:
            """Refresh turn request TTL during long-running dream steps.

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

        # Update dreamer state
        await self._update_dreamer_state(result)

        return result

    async def _update_dreamer_state(
        self: "MUDAgentWorker",
        result: DreamResult,
    ) -> None:
        """Update agent's dreamer state after a dream.

        Updates the agent:{id}:dreamer hash with:
        - last_dream_at: timestamp of completion
        - last_dream_scenario: scenario that was run
        - pending_pipeline_id: cleared on completion

        Args:
            result: DreamResult from completed pipeline
        """
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)

        now = datetime.now(timezone.utc).isoformat()

        if result.success:
            await client.update_dreamer_state_fields(
                self.config.agent_id,
                last_dream_at=now,
                last_dream_scenario=result.scenario or "",
                pending_pipeline_id="",
            )
        else:
            # On failure, just clear pending_pipeline_id
            await client.update_dreamer_state_fields(
                self.config.agent_id,
                pending_pipeline_id="",
            )

    async def check_auto_dream_triggers(
        self: "MUDAgentWorker",
    ) -> Optional[DreamRequest]:
        """Check if automatic dream should be triggered.

        Called during idle turns when dreamer is enabled. Checks:
        1. Dreamer is enabled for this agent
        2. Enough time has passed since last dream (idle_threshold_seconds)
        3. Enough tokens have accumulated (token_threshold)

        Returns:
            DreamRequest if triggers met, None otherwise
        """
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)

        state = await client.get_dreamer_state(self.config.agent_id)

        if not state:
            return None

        # Check if dreamer is enabled
        if not state.enabled:
            return None

        # Check idle time threshold
        if state.last_dream_at:
            last = datetime.fromisoformat(state.last_dream_at)
            elapsed = (datetime.now(timezone.utc) - last).total_seconds()
            if elapsed < state.idle_threshold_seconds:
                return None

        # Check token accumulation
        if self.conversation_manager:
            tokens = await self.conversation_manager.get_total_tokens()
            if tokens < state.token_threshold:
                return None

        # Triggers met - select a scenario
        scenario = await self._select_auto_dream_scenario()

        return DreamRequest(
            scenario=scenario,
            triggered_by="auto",
        )

    async def _select_auto_dream_scenario(
        self: "MUDAgentWorker",
    ) -> str:
        """Select appropriate scenario for automatic dream.

        Logic:
        - If conversation has many tokens (>20000): summarizer
          (lightweight 5-step pipeline to efficiently consolidate large memory)
        - Otherwise: analysis_dialogue
          (thorough 20-step analysis when context is manageable)

        Note: journaler_dialogue is NEVER auto-triggered. That scenario is
        reserved for rare, life-changing moments that Andi manually chooses
        with the @journal command.

        Returns:
            Name of scenario to run
        """
        if self.conversation_manager:
            tokens = await self.conversation_manager.get_total_tokens()
            if tokens > 20000:
                return "summarizer"
        return "analysis_dialogue"
