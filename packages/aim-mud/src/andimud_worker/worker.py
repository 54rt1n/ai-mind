# aim/app/mud/worker/main.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Main worker class for the MUD agent worker.

Contains the MUDAgentWorker class with main loop and lifecycle management.
Extracted from worker.py lines 1-365, 374-620, 1870-1887
"""

import asyncio
import contextlib
from datetime import timedelta
import json
import logging
import signal
import warnings
from typing import Optional
import uuid

import redis.asyncio as redis

from aim.conversation.model import ConversationModel
from aim.agents.roster import Roster
from aim.chat.manager import ChatManager
from aim.agents.persona import Persona
from aim.config import ChatConfig
from aim.llm.llm import LLMProvider

from aim_mud_types import MUDAction, MUDTurnRequest, RedisKeys, TurnRequestStatus, TurnReason
from aim_mud_types.helper import _utc_now

from .config import MUDConfig
from .exceptions import AbortRequestedException

from .adapter import build_system_prompt
from aim_mud_types import MUDSession
from .conversation.storage import MUDMemoryRetriever
from .conversation import MUDConversationManager
from .conversation.memory import MUDDecisionStrategy, MUDResponseStrategy

# Import mixins
from .mixins.llm import LLMMixin
from .mixins.turns import TurnsMixin
from .mixins.dreamer import DreamerMixin
from .mixins.datastore.profile import ProfileMixin
from .mixins.datastore.events import EventsMixin
from .mixins.datastore.actions import ActionsMixin
from .mixins.datastore.state import StateMixin
from .mixins.datastore.turn_request import TurnRequestMixin
from .mixins.datastore.report import ReportMixin
from .mixins.planner import PlannerMixin
from .conversation.storage import generate_conversation_id
from .commands import (
    CommandRegistry,
    FlushCommand,
    ClearCommand,
    NewConversationCommand,
    AgentCommand,
    ChooseCommand,
    DreamCommand,
    IdleCommand,
    EventsCommand,
    RetryCommand,
)


logger = logging.getLogger(__name__)


class MUDAgentWorker(PlannerMixin, ProfileMixin, EventsMixin, LLMMixin, ActionsMixin, TurnsMixin, DreamerMixin, StateMixin, TurnRequestMixin, ReportMixin):
    """Worker that consumes events from Redis stream and processes them.

    Follows the DreamerWorker pattern from aim/dreamer/worker.py.

    Key design: Events are PUSHED by the mediator, not polled.
    - Mediator filters events by room (agents only see their location)
    - Mediator enriches events with room state (no separate REST call needed)
    - Agent blocks on stream read until events arrive

    Attributes:
        config: MUD worker configuration.
        redis: Async Redis client.
        running: Whether the worker loop is active.
        cvm: ConversationModel for memory operations.
        roster: Roster of available personas.
        persona: The persona for this agent.
        session: Current MUD session state.
    """

    def __init__(
        self,
        config: MUDConfig,
        redis_client: redis.Redis,
        chat_config: Optional[ChatConfig] = None,
    ):
        """Initialize worker with configuration and Redis client.

        Originally from worker.py lines 91-139

        Args:
            config: MUDConfig with agent identity and settings.
            redis_client: Async Redis client for stream operations.
            chat_config: Optional pre-loaded ChatConfig with API keys and paths.
                If None, will be loaded from environment in start().
        """
        self.config = config
        self.redis = redis_client
        self.running = False

        # Shared resources (loaded once, reused across turns)
        self.cvm: Optional[ConversationModel] = None
        self.roster: Optional[Roster] = None
        self.persona: Optional[Persona] = None
        self.session: Optional[MUDSession] = None

        # LLM and tool resources (loaded during start)
        # Store pre-loaded config if provided, otherwise will load in start()
        self.chat_config: Optional[ChatConfig] = chat_config
        self.model_set = None  # ModelSet instance (initialized in _init_llm_provider)
        self.model_name: Optional[str] = None
        self.model = None

        # Memory helpers
        self.memory_retriever: Optional[MUDMemoryRetriever] = None

        # Conversation manager (Redis-backed conversation list)
        self.conversation_manager: Optional[MUDConversationManager] = None

        # Phase 1 decision strategy
        self._decision_strategy: Optional[MUDDecisionStrategy] = None
        self._agent_action_spec: Optional[dict] = None
        self._chat_manager: Optional[ChatManager] = None

        # Phase 2 response strategy
        self._response_strategy: Optional[MUDResponseStrategy] = None

        # Turn request tracking
        self._last_turn_request_id: Optional[str] = None

        # Main loop task tracking for clean shutdown
        self._main_loop_task: Optional[asyncio.Task] = None

        # Event accumulation buffer
        self.pending_events: list = []

        # Command registry
        self.command_registry = CommandRegistry.register(
            FlushCommand(),
            ClearCommand(),
            NewConversationCommand(),
            AgentCommand(),
            ChooseCommand(),
            DreamCommand(),
            IdleCommand(),
            EventsCommand(),
            RetryCommand(),
        )

    async def start(self) -> None:
        """Start the worker loop.

        Originally from worker.py lines 140-364

        Initializes shared resources (CVM, Roster, Persona, Session) and
        enters the main processing loop, consuming events from the agent's
        stream until stopped.
        """
        logger.info(f"Starting MUD agent worker for {self.config.agent_id}")

        # Initialize shared resources once
        # Use pre-loaded ChatConfig if available, otherwise load from environment
        # This follows the pattern from aim/app/dreamer/__main__.py
        if self.chat_config is None:
            self.chat_config = ChatConfig.from_env()

        # Apply MUD-specific persona override (from_config derives memory path from persona_id)
        # LLM settings (model, temperature, max_tokens) come from ChatConfig/.env
        self.chat_config.persona_id = self.config.persona_id

        self.cvm = ConversationModel.from_config(self.chat_config)
        self.roster = Roster.from_config(self.chat_config)
        self._chat_manager = ChatManager(self.cvm, self.chat_config, self.roster)
        self.persona = self.roster.get_persona(self.config.persona_id)

        # Initialize LLM provider
        self._init_llm_provider()

        # System message is built by strategy mixins (PhaseOneMixin, PhaseTwoMixin)
        # Each strategy calls get_system_message(persona) in build_turns()
        self.chat_config.system_message = ""  # Placeholder, strategies override this

        self._init_agent_action_spec()

        # Initialize session
        self.session = MUDSession(
            agent_id=self.config.agent_id,
            persona_id=self.config.persona_id,
            max_recent_turns=self.config.max_recent_turns,
        )

        # Initialize memory helpers
        # Suppress deprecation warning - we intentionally use the deprecated class
        # during the transition period; MUDResponseStrategy handles memory retrieval
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.memory_retriever = MUDMemoryRetriever(
                cvm=self.cvm,
                top_n=self.config.top_n_memories,
            )

        # Initialize conversation manager (Redis-backed conversation list)
        self.conversation_manager = MUDConversationManager(
            redis=self.redis,
            agent_id=self.config.agent_id,
            persona_id=self.config.persona_id,
            max_tokens=self.config.conversation_max_tokens,
        )

        # Update conversation report cache
        await self._update_conversation_report()

        # Initialize decision strategy
        self._decision_strategy = MUDDecisionStrategy(self._chat_manager)
        self._decision_strategy.set_conversation_manager(self.conversation_manager)
        # Phase 1 strategy loads its own tools
        self._decision_strategy.init_tools(
            tool_file=self.config.decision_tool_file,
            tools_path=self.chat_config.tools_path,
        )

        # Initialize response strategy
        self._response_strategy = MUDResponseStrategy(self._chat_manager)
        self._response_strategy.set_conversation_manager(self.conversation_manager)

        # Set running flag
        self.running = True

        # Setup signal handlers for graceful shutdown
        self.setup_signal_handlers()

        # Load agent profile state (last_event_id, etc.)
        await self._load_agent_profile()

        # Load world state (inventory and room pointer) for wakeup context
        room_id, character_id = await self._load_agent_world_state()
        await self._load_room_profile(room_id, character_id)

        # Announce worker presence (now has world_state for mud_wakeup context)
        await self._announce_presence()

        logger.info(
            f"Worker initialized. Listening on stream: {self.config.agent_stream}"
        )
        logger.info(f"Using model: {self.chat_config.default_model}")

        # Run the main worker loop
        # Track main loop task for clean shutdown synchronization
        self._main_loop_task = asyncio.create_task(self._run_main_loop())
        await self._main_loop_task

        logger.info("Worker loop ended")

    async def _run_main_loop(self) -> None:
        """Run the main worker loop.

        Processes turn requests, drains events, executes turns, and handles
        state transitions. This method assumes all initialization is complete.

        Continues until self.running is set to False.
        """
        # Main worker loop - blocks until events arrive (push model)
        while self.running:
            try:
                # Check if paused
                if await self._is_paused():
                    logger.debug("Worker paused, sleeping...")
                    await asyncio.sleep(1)
                    continue

                # Check for abort request
                if await self._check_abort_requested():
                    logger.info("Turn abort detected, clearing abort flag")
                    continue

                # Check for turn request assignment
                turn_request = await self._get_turn_request()

                # If turn_request is missing, re-announce presence (Redis flush, expiration, etc.)
                if not turn_request:
                    logger.warning("turn_request missing, re-announcing presence")
                    await self._announce_presence()
                    await asyncio.sleep(self.config.turn_request_poll_interval)
                    continue

                # Check for immediate commands first (EXECUTE status)
                if turn_request.status == TurnRequestStatus.EXECUTE:
                    logger.info(
                        f"Worker {self.config.agent_id}: Processing immediate command "
                        f"(reason={turn_request.reason}, turn_id={turn_request.turn_id})"
                    )
                    # Transition to EXECUTING
                    turn_request.status = TurnRequestStatus.EXECUTING
                    turn_request.heartbeat_at = _utc_now()
                    await self.update_turn_request(turn_request, expected_turn_id=turn_request.turn_id)

                    # Execute command directly - NO event drain, NO turn guard
                    heartbeat_stop = asyncio.Event()
                    heartbeat_task = asyncio.create_task(
                        self._heartbeat_turn_request(heartbeat_stop)
                    )

                    try:
                        result = await self.command_registry.execute(self, **turn_request.model_dump())

                        # Handle result status
                        if result.complete:
                            current = await self._get_turn_request()
                            if current:
                                current.status = result.status
                                current.message = result.message or ""
                                current.completed_at = _utc_now()
                                current.heartbeat_at = _utc_now()
                                await self.update_turn_request(current, expected_turn_id=turn_request.turn_id)
                        else:
                            # Command returned incomplete - transition to DONE
                            current = await self._get_turn_request()
                            if current:
                                current.status = TurnRequestStatus.DONE
                                current.completed_at = _utc_now()
                                current.heartbeat_at = _utc_now()
                                await self.update_turn_request(current, expected_turn_id=turn_request.turn_id)

                    except AbortRequestedException:
                        logger.info(f"Immediate command {turn_request.turn_id} aborted by user request")
                        current = await self._get_turn_request()
                        if current:
                            current.status = TurnRequestStatus.ABORTED
                            current.message = "Aborted by user"
                            current.completed_at = _utc_now()
                            current.heartbeat_at = _utc_now()
                            await self.update_turn_request(current, expected_turn_id=turn_request.turn_id)
                    except Exception as e:
                        logger.error(f"Error during immediate command {turn_request.turn_id}: {e}", exc_info=True)
                        current = await self._get_turn_request()
                        if current:
                            # Use _handle_turn_failure to get proper RETRY/FAIL logic
                            await self._handle_turn_failure(
                                turn_id=turn_request.turn_id,
                                error_message=str(e),
                                error_type=f"Immediate command failed: {type(e).__name__}"
                            )
                    finally:
                        heartbeat_stop.set()
                        await heartbeat_task

                        # Transition EXECUTING commands to READY after completion
                        final_turn_request = await self._get_turn_request()
                        if final_turn_request:
                            final_status = final_turn_request.status
                            if final_status in (TurnRequestStatus.DONE, TurnRequestStatus.ABORTED):
                                final_turn_request.turn_id = str(uuid.uuid4())
                                final_turn_request.status = TurnRequestStatus.READY
                                final_turn_request.status_reason = f"Immediate command completed ({final_status.value})"
                                final_turn_request.heartbeat_at = _utc_now()
                                await self.update_turn_request(final_turn_request, expected_turn_id=turn_request.turn_id)

                    continue

                # If not assigned, refresh heartbeat and wait
                if turn_request.status != TurnRequestStatus.ASSIGNED:
                    # Keep turn_request alive when ready but idle - refresh TTL and heartbeat
                    if turn_request.status == TurnRequestStatus.READY:
                        # Atomic heartbeat update with validation
                        result = await self.atomic_heartbeat_update()

                        if result == -1:
                            # Corrupted hash detected - recreate with fresh READY state
                            logger.warning("Idle heartbeat detected corrupted hash, recreating state")
                            current = await self._get_turn_request()
                            if current:
                                current.turn_id = str(uuid.uuid4())
                                current.status = TurnRequestStatus.READY
                                current.status_reason = "Recovered from corrupted hash during idle"
                                current.heartbeat_at = _utc_now()
                                await self.update_turn_request(current, expected_turn_id=turn_request.turn_id)

                    await asyncio.sleep(self.config.turn_request_poll_interval)
                    continue

                turn_id = turn_request.turn_id
                reason = turn_request.reason

                turn_request.status = TurnRequestStatus.IN_PROGRESS
                turn_request.heartbeat_at = _utc_now()
                await self.update_turn_request(turn_request, expected_turn_id=turn_id)
                heartbeat_stop = asyncio.Event()
                heartbeat_task = asyncio.create_task(
                    self._heartbeat_turn_request(heartbeat_stop)
                )

                # Turn guard: Check if we have the oldest active turn
                if not await self._should_process_turn(turn_request):
                    logger.info(
                        f"Turn {turn_id} is not oldest active turn, delaying processing"
                    )
                    current = await self._get_turn_request()
                    if current:
                        current.status = TurnRequestStatus.ASSIGNED
                        current.message = "Delayed (waiting for older turn)"
                        current.heartbeat_at = _utc_now()
                        await self.update_turn_request(current, expected_turn_id=turn_id)
                    heartbeat_stop.set()
                    await heartbeat_task
                    await asyncio.sleep(0.5)
                    continue

                try:
                    # SAVE pre-drain event position for potential rollback
                    saved_event_id = self.session.last_event_id

                    # Ensure Phase 1 tools include plan tools whenever a plan is active
                    try:
                        active_plan = await self.check_active_plan()
                        if active_plan:
                            self.set_active_plan(active_plan)
                        elif self.get_active_plan():
                            self.clear_active_plan()
                    except Exception as e:
                        logger.error(f"Failed to refresh active plan state: {e}", exc_info=True)

                    # Get turn's sequence_id for filtering
                    if not turn_request.sequence_id:
                        logger.warning(f"Turn {turn_id} missing sequence_id, cannot filter drain")
                        max_seq = None
                    else:
                        max_seq = turn_request.sequence_id
                        logger.debug(f"Draining events with sequence_id < {max_seq}")

                    # Drain events into pending_events buffer, filtered by sequence_id
                    self.pending_events = await self._drain_with_settle(max_sequence_id=max_seq)

                    # Execute command via registry
                    logger.info(f"Executing command: {turn_request.model_dump()}")
                    result = await self.command_registry.execute(self, **turn_request.model_dump())

                    # Handle flush_drain flag
                    if result.flush_drain:
                        self.pending_events = []
                        saved_event_id = None  # Don't restore - events were consumed

                    # If command completed fully, set status and continue
                    if result.complete:
                        if result.status == TurnRequestStatus.FAIL:
                            # Handle failure with backoff
                            await self._handle_turn_failure(turn_id, result.message or "Command failed")
                        else:
                            # Non-failure status (DONE, ABORTED, etc.)
                            current = await self._get_turn_request()
                            if current:
                                current.status = result.status
                                current.message = result.message or ""
                                current.completed_at = _utc_now()
                                current.heartbeat_at = _utc_now()
                                await self.update_turn_request(current, expected_turn_id=turn_id)
                        self._last_turn_request_id = turn_id
                        continue

                    # Command returned complete=False, fall through to process_turn
                    events = self.pending_events
                    logger.info(
                        "Processing assigned turn %s (%s) with %d events",
                        turn_id,
                        reason,
                        len(events),
                    )
                    await self.process_turn(turn_request, events)

                    # CHECK: Did this turn include a speech event?
                    has_speech = False
                    last_turn = self.session.get_last_turn() if self.session else None
                    if last_turn:
                        for action in last_turn.actions_taken:
                            if action.tool == "speak":
                                has_speech = True
                                logger.info(f"Turn {turn_id} included speech event, consuming drained events")
                                break

                    # CONDITIONAL CONSUMPTION: Only speech events advance last_event_id
                    if not has_speech:
                        # Non-speech turn: restore event position (events remain for next turn)
                        logger.info(f"Turn {turn_id} was non-speech, restoring event position")
                        await self._restore_event_position(saved_event_id)
                        saved_event_id = None  # Prevent double-restore in exception handler
                    else:
                        # Speech turn: keep advanced last_event_id (events consumed)
                        logger.info(f"Turn {turn_id} was speech, events consumed")

                        # Persist the advanced event position ONLY for speech turns
                        # This ensures events are consumed only when the agent actually spoke
                        # Non-speech turns restore via _restore_event_position() which also persists
                        await self._update_agent_profile(last_event_id=self.session.last_event_id)

                        saved_event_id = None  # Events consumed, don't restore on exception

                    current = await self._get_turn_request()
                    if current:
                        current.status = TurnRequestStatus.DONE
                        current.completed_at = _utc_now()
                        current.heartbeat_at = _utc_now()
                        await self.update_turn_request(current, expected_turn_id=turn_id)
                    self._last_turn_request_id = turn_id
                except AbortRequestedException:
                    logger.info(f"Turn {turn_id} aborted by user request")
                    current = await self._get_turn_request()
                    if current:
                        current.status = TurnRequestStatus.ABORTED
                        current.message = "Aborted by user"
                        current.completed_at = _utc_now()
                        current.heartbeat_at = _utc_now()
                        await self.update_turn_request(current, expected_turn_id=turn_id)
                    await self._restore_event_position(saved_event_id)
                except Exception as e:
                    logger.error(f"Error during assigned turn {turn_id}: {e}", exc_info=True)

                    # Handle failure with exponential backoff
                    error_type = type(e).__name__
                    await self._handle_turn_failure(turn_id, str(e), error_type)

                    # Restore event position so retry gets the same events
                    # Only restore if saved_event_id is not None (not already restored)
                    if saved_event_id:
                        await self._restore_event_position(saved_event_id)
                finally:
                    heartbeat_stop.set()
                    heartbeat_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await heartbeat_task

                    # After turn completes (success or abort), return to ready
                    if turn_id:  # Only if we actually processed a turn
                        turn_request = await self._get_turn_request()
                        if turn_request:
                            status = turn_request.status
                            # Transition to ready if we completed or aborted (not if failed)
                            if status == TurnRequestStatus.DONE:
                                turn_request.turn_id = str(uuid.uuid4())
                                turn_request.status = TurnRequestStatus.READY
                                turn_request.status_reason = "Turn completed"
                                turn_request.heartbeat_at = _utc_now()
                                await self.update_turn_request(turn_request, expected_turn_id=turn_id)
                            elif status == TurnRequestStatus.ABORTED:
                                turn_request.turn_id = str(uuid.uuid4())
                                turn_request.status = TurnRequestStatus.READY
                                turn_request.status_reason = "Turn aborted"
                                turn_request.heartbeat_at = _utc_now()
                                await self.update_turn_request(turn_request, expected_turn_id=turn_id)

            except asyncio.CancelledError:
                logger.info("Worker cancelled, shutting down...")
                break
            except Exception as e:
                # Log error but continue processing
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                continue

    async def _handle_turn_failure(
        self,
        turn_id: str,
        error_message: str,
        error_type: Optional[str] = None
    ) -> None:
        """Handle turn failure with exponential backoff.

        Sets turn_request to RETRY status for temporary failures, FAIL for permanent failures:
        - Incremented attempt_count
        - Exponential backoff calculation for next_attempt_at
        - Appropriate status_reason
        - completed_at timestamp for terminal states

        Args:
            turn_id: The turn ID that failed
            error_message: Error message to store
            error_type: Optional error type name for status_reason
        """
        # Get current attempt count
        turn_request = await self._get_turn_request()
        attempt_count = turn_request.attempt_count + 1 if turn_request else 1

        # Determine if we're retrying or giving up
        if attempt_count < self.config.llm_failure_max_attempts:
            backoff_seconds = min(
                self.config.llm_failure_backoff_base_seconds * (2 ** (attempt_count - 1)),
                self.config.llm_failure_backoff_max_seconds
            )
            next_attempt_at = (_utc_now() + timedelta(seconds=backoff_seconds)).isoformat()
            status = TurnRequestStatus.RETRY
            logger.info(
                f"Turn {turn_id} failed (attempt {attempt_count}/{self.config.llm_failure_max_attempts}), "
                f"will retry in {backoff_seconds}s"
            )
        else:
            next_attempt_at = ""  # Max attempts reached
            status = TurnRequestStatus.FAIL
            logger.error(f"Turn {turn_id} failed permanently after {attempt_count} attempts, giving up")

        # Set failure/retry state with metadata
        status_reason = f"LLM call failed: {error_type}" if error_type else "Command failed"
        if turn_request:
            turn_request.status = status
            turn_request.message = error_message
            turn_request.attempt_count = attempt_count
            turn_request.next_attempt_at = next_attempt_at
            turn_request.status_reason = status_reason
            turn_request.completed_at = _utc_now()
            turn_request.heartbeat_at = _utc_now()
            await self.update_turn_request(turn_request, expected_turn_id=turn_id)

    async def stop(self) -> None:
        """Gracefully stop the worker.

        Originally from worker.py lines 365-373

        Sets the running flag to False, allowing the current turn
        to complete before shutting down. Cleans up turn_request state.
        """
        logger.info("Stopping worker...")
        self.running = False

        # Wait for main loop to exit cleanly before cleanup
        if hasattr(self, '_main_loop_task') and self._main_loop_task:
            try:
                await asyncio.wait_for(self._main_loop_task, timeout=5.0)
                logger.info("Main loop exited cleanly")
            except asyncio.TimeoutError:
                logger.warning("Main loop did not exit within 5s, forcing stop")
            except Exception as e:
                logger.warning(f"Error waiting for main loop: {e}")

        # NOW safe to delete turn_request (main loop has stopped)
        turn_request = await self._get_turn_request()
        if turn_request:
            logger.info("Deleting turn_request on shutdown")
            await self.redis.delete(self._turn_request_key())

    async def _announce_presence(self) -> None:
        """Announce worker online with ready status and wakeup message.

        Implements startup recovery logic:
        - Branch 1: No turn_request → create with status=READY
        - Branch 2: Problem states (in_progress/crashed/assigned/abort_requested) → convert to FAIL with recovery
        - Branch 3: Normal states (ready/done/fail) → update heartbeat only

        Uses persona.mud_wakeup if available, otherwise falls back to
        random wakeup message from persona.wakeup list. If mud_wakeup is
        provided, room state context is passed for formatting.
        """
        turn_request = await self._get_turn_request()

        # Branch 1: No turn_request → create with status=READY
        if not turn_request:
            logger.info("No turn_request found, creating fresh ready state")
            new_turn_request = MUDTurnRequest(
                turn_id=str(uuid.uuid4()),
                status=TurnRequestStatus.READY,
                reason=TurnReason.EVENTS,
                sequence_id=-1,
                attempt_count=0,
                status_reason="Worker online (fresh start)"
            )
            success = await self.create_turn_request(new_turn_request)
            if not success:
                logger.debug("turn_request already exists (created by concurrent process), skipping creation")
                return

        # Branch 2: Problem states → convert to FAIL with recovery logic
        elif turn_request.status in (TurnRequestStatus.IN_PROGRESS, TurnRequestStatus.CRASHED, TurnRequestStatus.ASSIGNED, TurnRequestStatus.ABORT_REQUESTED, TurnRequestStatus.EXECUTING):
            status = turn_request.status
            turn_id = turn_request.turn_id

            logger.warning(f"Startup recovery: found turn_request in problem state '{status}', converting to fail")

            # Get current attempt count or default to 0
            attempt_count = turn_request.attempt_count

            # Increment attempt count for this recovery
            attempt_count += 1

            # Check if we've exhausted max attempts
            if attempt_count >= self.config.llm_failure_max_attempts:
                # Max attempts reached - set fail state with no retry
                logger.error(
                    f"Turn {turn_id} abandoned after {attempt_count} attempts (startup recovery)"
                )
                turn_request.status = TurnRequestStatus.FAIL
                turn_request.message = f"Abandoned after {attempt_count} attempts (startup recovery from {status})"
                turn_request.attempt_count = attempt_count
                turn_request.next_attempt_at = ""  # Empty = no retry
                turn_request.status_reason = f"Max attempts reached during startup recovery from {status}"
                turn_request.completed_at = _utc_now()
                turn_request.heartbeat_at = _utc_now()
                await self.update_turn_request(turn_request, expected_turn_id=turn_id)
            else:
                # Will retry - use RETRY status
                backoff_seconds = min(
                    self.config.llm_failure_backoff_base_seconds * (2 ** (attempt_count - 1)),
                    self.config.llm_failure_backoff_max_seconds
                )
                next_attempt_at = (_utc_now() + timedelta(seconds=backoff_seconds)).isoformat()

                logger.info(
                    f"Turn {turn_id} marked for retry (attempt {attempt_count}/{self.config.llm_failure_max_attempts}), "
                    f"will retry in {backoff_seconds}s (startup recovery from {status})"
                )

                turn_request.status = TurnRequestStatus.RETRY
                turn_request.message = f"Worker restart during {status} state"
                turn_request.attempt_count = attempt_count
                turn_request.next_attempt_at = next_attempt_at
                turn_request.status_reason = f"Startup recovery from {status}"
                turn_request.completed_at = _utc_now()
                turn_request.heartbeat_at = _utc_now()
                await self.update_turn_request(turn_request, expected_turn_id=turn_id)

        # Branch 3: Normal states (ready/done/fail) → update heartbeat or fix corruption
        else:
            status = turn_request.status
            turn_id = turn_request.turn_id

            # Detect corrupted hash (missing status field)
            if not status:
                logger.error(
                    "Startup: turn_request corrupted (status is None), recreating with fresh state"
                )
                turn_request.turn_id = str(uuid.uuid4())
                turn_request.status = TurnRequestStatus.READY
                turn_request.status_reason = "Recovered from corrupted state during startup"
                turn_request.heartbeat_at = _utc_now()
                await self.update_turn_request(turn_request, expected_turn_id=turn_id)
            else:
                logger.info(f"Startup: turn_request in normal state '{status}', updating heartbeat")

                # Atomic heartbeat update with validation
                result = await self.atomic_heartbeat_update()

                if result == -1:
                    # Corrupted hash detected during atomic update - recreate with fresh state
                    logger.warning("Startup: atomic update detected corrupted hash, recreating state")
                    turn_request.turn_id = str(uuid.uuid4())
                    turn_request.status = TurnRequestStatus.READY
                    turn_request.status_reason = "Recovered from corrupted hash during startup"
                    turn_request.heartbeat_at = _utc_now()
                    await self.update_turn_request(turn_request, expected_turn_id=turn_id)

        # Get wakeup message with room context
        if self.persona.mud_wakeup and self.session and self.session.world_state:
            # Use MUD-specific wakeup with room state context
            world_state = self.session.world_state
            room = world_state.room_state

            # Build context for formatting
            context = {
                "room_name": room.name if room else "somewhere",
                "room_description": room.description if room else "",
                "persona_name": self.persona.name,
                "full_name": self.persona.full_name,
                "memory_count": 15,  # Placeholder for Active Memory count
            }

            # Add entities present (excluding self)
            others = [e for e in world_state.entities_present if not e.is_self and e.entity_type in ("player", "ai", "npc")]
            if others:
                context["others_present"] = ", ".join(e.name for e in others)
            else:
                context["others_present"] = ""

            # Add objects in room (excluding characters)
            objects = [e for e in world_state.entities_present if e.entity_type not in ("player", "ai", "npc")]
            if objects:
                context["objects_present"] = ", ".join(e.name for e in objects)
            else:
                context["objects_present"] = ""

            # Add inventory items
            if world_state.inventory:
                context["inventory"] = ", ".join(item.name for item in world_state.inventory)
            else:
                context["inventory"] = ""

            # Build contextual details for think block
            context_details_parts = []
            if context["others_present"]:
                context_details_parts.append(f"Others present: {context['others_present']}")
            if context["objects_present"]:
                context_details_parts.append(f"Objects here: {context['objects_present']}")
            if context["inventory"]:
                context_details_parts.append(f"I'm carrying: {context['inventory']}")

            context["context_details"] = "\n\n".join(context_details_parts) if context_details_parts else "The space is mine alone."

            # Format wakeup message with full context
            try:
                wakeup_msg = self.persona.mud_wakeup.format(**context)
            except KeyError as e:
                logger.warning(f"mud_wakeup template missing key {e}, using as-is")
                wakeup_msg = self.persona.mud_wakeup
        else:
            # Fall back to random wakeup message (no room context)
            wakeup_msg = self.persona.get_wakeup()

        # Add wakeup to conversation history only if this is a fresh session
        # This provides rich internal state for the agent's first actual response
        entry_count = await self.conversation_manager.get_entry_count()
        if entry_count == 0:
            await self.conversation_manager.push_assistant_turn(
                content=wakeup_msg,
                think=None,  # think block is already in wakeup_msg
                actions=[]
            )
            logger.info(f"Worker {self.config.agent_id} ready, seeded conversation with wakeup")
        else:
            logger.info(f"Worker {self.config.agent_id} ready (continuing existing conversation with {entry_count} entries)")

    async def _write_self_event(self, event: "MUDEvent") -> None:
        """Write self-action event to own stream immediately.

        Args:
            event: Self-action MUDEvent to write.
        """
        from aim_mud_types import MUDEvent

        event_data = event.to_redis_dict()
        stream_key = RedisKeys.agent_events(self.config.agent_id)
        await self.redis.xadd(
            stream_key,
            {"data": json.dumps(event_data)},
            maxlen=100,
            approximate=True,
        )
        logger.info(
            f"Wrote self-event: {event.event_type.value} (seq={event.sequence_id})"
        )

    def setup_signal_handlers(self) -> None:
        """Setup handlers for graceful shutdown.

        Originally from worker.py lines 1870-1887

        Registers signal handlers for SIGINT and SIGTERM to trigger
        graceful shutdown via stop().
        """

        def signal_handler(signum, frame):
            """Handle shutdown signals."""
            sig_name = signal.Signals(signum).name
            logger.info(f"Received {sig_name}, shutting down gracefully...")
            # Set flag to stop the worker - safe from any context
            self.running = False

        # Register handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
