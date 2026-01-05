# aim/app/mud/worker/main.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Main worker class for the MUD agent worker.

Contains the MUDAgentWorker class with main loop and lifecycle management.
Extracted from worker.py lines 1-365, 374-620, 1870-1887
"""

import asyncio
import contextlib
import logging
import signal
import warnings
from typing import Optional
import uuid

import redis.asyncio as redis

from aim_mud_types import MUDAction, RedisKeys

from ..adapter import build_system_prompt
from ..config import MUDConfig
from ..session import MUDSession
from aim.conversation.model import ConversationModel
from aim.agents.roster import Roster
from aim.chat.manager import ChatManager
from aim.agents.persona import Persona
from aim.config import ChatConfig
from aim.llm.llm import LLMProvider
from aim.llm.models import LanguageModelV2
from ..memory import MUDMemoryRetriever
from ..conversation import MUDConversationManager
from ..strategy import MUDDecisionStrategy, MUDResponseStrategy
from datetime import timedelta

# Import mixins
from .profile import ProfileMixin
from .events import EventsMixin
from .llm import LLMMixin
from .actions import ActionsMixin
from .turns import TurnsMixin
from .utils import _utc_now


logger = logging.getLogger(__name__)


class AbortRequestedException(Exception):
    """Raised when a turn is aborted by user request.

    Originally from worker.py lines 61-63
    """
    pass


class MUDAgentWorker(ProfileMixin, EventsMixin, LLMMixin, ActionsMixin, TurnsMixin):
    """Worker that consumes events from Redis stream and processes them.

    Originally from worker.py lines 71-1887

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
        self._llm_provider: Optional[LLMProvider] = None
        self.model_name: Optional[str] = None
        self.model = None

        # Memory helpers
        self.memory_retriever: Optional[MUDMemoryRetriever] = None

        # Conversation manager (Redis-backed conversation list)
        self.conversation_manager: Optional[MUDConversationManager] = None

        # Phase 1 decision tools
        self._decision_tool_user = None
        self._decision_strategy: Optional[MUDDecisionStrategy] = None
        self._agent_action_spec: Optional[dict] = None
        self._chat_manager: Optional[ChatManager] = None

        # Phase 2 response strategy
        self._response_strategy: Optional[MUDResponseStrategy] = None

        # Turn request tracking
        self._last_turn_request_id: Optional[str] = None

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

        # Initialize phase 1 decision tools
        self._init_decision_tools()

        # Build system message using XML decorator (includes wardrobe, features, attributes)
        # This follows the pattern from aim_server/modules/chat/route.py
        # Tools are included so Phase 1 (decision) can use them; Phase 2 ignores them
        from aim.utils.xml import XmlFormatter
        xml = XmlFormatter()
        xml = self.persona.xml_decorator(
            xml,
            disable_guidance=False,
            disable_pif=False,
            conversation_length=0,
        )
        # Add decision tools to system message
        if self._decision_tool_user:
            xml = self._decision_tool_user.xml_decorator(xml)
        self.chat_config.system_message = xml.render()
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

        # Initialize decision strategy
        self._decision_strategy = MUDDecisionStrategy(self.conversation_manager)
        self._decision_strategy.set_tool_user(self._decision_tool_user)

        # Initialize response strategy
        self._response_strategy = MUDResponseStrategy(self._chat_manager)
        self._response_strategy.set_conversation_manager(self.conversation_manager)

        # Set running flag
        self.running = True

        # Setup signal handlers for graceful shutdown
        self.setup_signal_handlers()

        # Load agent profile state (last_event_id, etc.)
        await self._load_agent_profile()

        # Announce worker presence
        await self._announce_presence()

        logger.info(
            f"Worker initialized. Listening on stream: {self.config.agent_stream}"
        )
        logger.info(f"Using model: {self.chat_config.default_model}")

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
                if not turn_request or turn_request.get("status") != "assigned":
                    await asyncio.sleep(self.config.turn_request_poll_interval)
                    continue

                turn_id = turn_request.get("turn_id") or uuid.uuid4().hex
                reason = turn_request.get("reason", "events")

                await self._set_turn_request_state(turn_id, "in_progress")
                heartbeat_stop = asyncio.Event()
                heartbeat_task = asyncio.create_task(
                    self._heartbeat_turn_request(heartbeat_stop)
                )
                saved_last_event_id = None  # For rollback on failure

                try:
                    if reason == "flush":
                        # @write console command - flush conversation to CVM
                        if self.conversation_manager:
                            flushed = await self.conversation_manager.flush_to_cvm(self.cvm)
                            logger.info(f"@write: Flushed {flushed} entries to CVM")
                            # Emote completion
                            action = MUDAction(tool="emote", args={"action": "feels more knowledgeable."})
                            await self._emit_actions([action])
                        else:
                            logger.warning("Flush requested but no conversation manager")
                        await self._set_turn_request_state(turn_id, "done")
                        self._last_turn_request_id = turn_id
                        continue
                    if reason == "clear":
                        # @clear console command - clear conversation history
                        if self.conversation_manager:
                            await self.conversation_manager.clear()
                            logger.info("@clear: Cleared conversation history")
                        else:
                            logger.warning("Clear requested but no conversation manager")
                        await self._set_turn_request_state(turn_id, "done")
                        self._last_turn_request_id = turn_id
                        continue
                    if reason == "agent":
                        # Save event position before draining (for rollback on failure)
                        saved_last_event_id = self.session.last_event_id
                        events = await self.drain_events(timeout=0)
                        guidance = turn_request.get("guidance", "")
                        logger.info(
                            "Processing @agent turn %s with %d events",
                            turn_id,
                            len(events),
                        )
                        await self.process_agent_turn(events, guidance)
                        await self._set_turn_request_state(turn_id, "done")
                        self._last_turn_request_id = turn_id
                        continue
                    if reason == "choose":
                        # Save event position before draining (for rollback on failure)
                        saved_last_event_id = self.session.last_event_id
                        events = await self.drain_events(timeout=0)
                        guidance = turn_request.get("guidance", "")
                        logger.info(
                            "Processing @choose turn %s with %d events",
                            turn_id,
                            len(events),
                        )
                        # Use process_agent_turn for @choose (same as @agent)
                        await self.process_agent_turn(events, guidance)
                        await self._set_turn_request_state(turn_id, "done")
                        self._last_turn_request_id = turn_id
                        continue
                    if reason == "idle":
                        events = []
                        saved_last_event_id = None
                    else:
                        # Save event position before draining (for rollback on failure)
                        saved_last_event_id = self.session.last_event_id
                        # Drain events with settling delay for cascades
                        events = await self._drain_with_settle()

                    logger.info(
                        "Processing assigned turn %s (%s) with %d events",
                        turn_id,
                        reason,
                        len(events),
                    )
                    await self.process_turn(events)
                    await self._set_turn_request_state(turn_id, "done")
                    self._last_turn_request_id = turn_id
                except AbortRequestedException:
                    logger.info(f"Turn {turn_id} aborted by user request")
                    # Restore event position so next drain re-reads these events
                    if saved_last_event_id is not None:
                        self.session.last_event_id = saved_last_event_id
                        logger.info("Restored last_event_id to %s after abort", saved_last_event_id)
                    await self._set_turn_request_state(turn_id, "aborted", message="Aborted by user")
                except Exception as e:
                    logger.error(f"Error during assigned turn {turn_id}: {e}", exc_info=True)

                    # Get current attempt count
                    turn_request = await self._get_turn_request()
                    attempt_count = int(turn_request.get("attempt_count", 0)) + 1

                    # Calculate next retry time with exponential backoff
                    if attempt_count < self.config.llm_failure_max_attempts:
                        backoff = min(
                            self.config.llm_failure_backoff_base_seconds * (2 ** (attempt_count - 1)),
                            self.config.llm_failure_backoff_max_seconds
                        )
                        next_attempt_at = (_utc_now() + timedelta(seconds=backoff)).isoformat()
                        logger.info(
                            f"Turn {turn_id} failed (attempt {attempt_count}/{self.config.llm_failure_max_attempts}), "
                            f"will retry in {backoff}s"
                        )
                    else:
                        next_attempt_at = ""  # Max attempts reached
                        logger.error(f"Turn {turn_id} failed after {attempt_count} attempts, giving up")

                    # Restore event position for retry
                    if saved_last_event_id is not None:
                        self.session.last_event_id = saved_last_event_id
                        logger.info(f"Restored last_event_id to {saved_last_event_id} for retry")

                    # Set failure state with retry metadata (use CAS to ensure we're updating the right turn)
                    error_type = type(e).__name__
                    await self._set_turn_request_state(
                        turn_id,
                        "fail",
                        message=str(e),
                        extra_fields={
                            "attempt_count": str(attempt_count),
                            "next_attempt_at": next_attempt_at,
                            "status_reason": f"LLM call failed: {error_type}"
                        },
                        expected_turn_id=turn_id  # CAS: only update if turn_id matches
                    )
                finally:
                    heartbeat_stop.set()
                    heartbeat_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await heartbeat_task

                    # After turn completes (success or abort), return to ready
                    if turn_id:  # Only if we actually processed a turn
                        turn_request = await self._get_turn_request()
                        if turn_request:
                            status = turn_request.get("status")
                            # Transition to ready if we completed or aborted (not if failed)
                            if status == "done":
                                await self._set_turn_request_state(
                                    str(uuid.uuid4()),
                                    "ready",
                                    extra_fields={"status_reason": "Turn completed"},
                                    expected_turn_id=turn_id
                                )
                            elif status == "aborted":
                                await self._set_turn_request_state(
                                    str(uuid.uuid4()),
                                    "ready",
                                    extra_fields={"status_reason": "Turn aborted"},
                                    expected_turn_id=turn_id
                                )

            except asyncio.CancelledError:
                logger.info("Worker cancelled, shutting down...")
                break
            except Exception as e:
                # Log error but continue processing
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                continue

        logger.info("Worker loop ended")

    async def stop(self) -> None:
        """Gracefully stop the worker.

        Originally from worker.py lines 365-373

        Sets the running flag to False, allowing the current turn
        to complete before shutting down.
        """
        logger.info("Stopping worker...")
        self.running = False

    async def _is_paused(self) -> bool:
        """Check if worker is paused via Redis flag.

        Originally from worker.py lines 374-382

        Returns:
            bool: True if paused, False if running.
        """
        value = await self.redis.get(self.config.pause_key)
        return value == b"1"

    async def _announce_presence(self) -> None:
        """Announce worker online with ready status.

        Clears any crashed status and sets worker to ready state.
        """
        turn_request = await self._get_turn_request()

        # Clear crashed status if present
        if turn_request and turn_request.get("status") == "crashed":
            logger.info("Clearing crashed status, worker is online")

        # Set to ready
        await self._set_turn_request_state(
            turn_id=str(uuid.uuid4()),
            status="ready",
            extra_fields={"status_reason": "Worker online"}
        )
        logger.info(f"Worker {self.config.agent_id} announced as ready")

    async def _check_abort_requested(self) -> bool:
        """Check if current turn has abort requested.

        Originally from worker.py lines 383-400

        Returns:
            bool: True if abort requested, False otherwise.
        """
        turn_request = await self._get_turn_request()
        status = turn_request.get("status")
        if status == "abort_requested":
            # Clear the abort flag
            await self._set_turn_request_state(
                turn_request.get("turn_id", "unknown"),
                "aborted",
                message="Aborted by user request"
            )
            return True
        return False

    def _turn_request_key(self) -> str:
        """Return the Redis key for this agent's turn request.

        Originally from worker.py lines 401-403
        """
        return RedisKeys.agent_turn_request(self.config.agent_id)

    def _agent_profile_key(self) -> str:
        """Return the Redis key for this agent's profile hash.

        Originally from worker.py lines 405-407
        """
        return RedisKeys.agent_profile(self.config.agent_id)

    async def _get_turn_request(self) -> dict[str, str]:
        """Fetch the current turn request hash.

        Originally from worker.py lines 567-580
        """
        data = await self.redis.hgetall(self._turn_request_key())
        if not data:
            return {}
        result: dict[str, str] = {}
        for k, v in data.items():
            if isinstance(k, bytes):
                k = k.decode("utf-8")
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            result[str(k)] = str(v)
        return result

    async def _set_turn_request_state(
        self,
        turn_id: str,
        status: str,
        message: Optional[str] = None,
        extra_fields: Optional[dict] = None,
        expected_turn_id: Optional[str] = None,
    ) -> bool:
        """Set turn request status with CAS pattern.

        Args:
            turn_id: Turn ID to set
            status: New status
            message: Optional status message
            extra_fields: Optional additional fields
            expected_turn_id: If provided, only update if current turn_id matches (CAS)

        Returns:
            True if update succeeded, False if CAS check failed
        """
        key = self._turn_request_key()

        # Use Lua script for atomic CAS
        lua_script = """
            local key = KEYS[1]
            local expected_turn_id = ARGV[1]
            local new_turn_id = ARGV[2]

            -- If CAS check requested, verify current turn_id matches
            if expected_turn_id ~= "" then
                local current = redis.call('HGET', key, 'turn_id')
                if current ~= expected_turn_id then
                    return 0  -- CAS failed
                end
            end

            -- Update fields (passed as key-value pairs starting at ARGV[3])
            for i = 3, #ARGV, 2 do
                redis.call('HSET', key, ARGV[i], ARGV[i+1])
            end

            return 1  -- Success
        """

        # Build field updates
        fields = [
            "turn_id", turn_id,
            "status", status,
            "heartbeat_at", _utc_now().isoformat()
        ]
        if message:
            fields.extend(["message", message])
        if extra_fields:
            for k, v in extra_fields.items():
                fields.extend([k, str(v)])

        # Execute CAS update
        result = await self.redis.eval(
            lua_script,
            1,  # number of keys
            key,
            expected_turn_id or "",  # Empty string = no CAS check
            turn_id,
            *fields
        )

        if result == 1:
            # Success - also update TTL
            await self.redis.expire(key, self.config.turn_request_ttl_seconds)
            return True
        else:
            logger.warning(f"CAS failed: expected turn_id {expected_turn_id}, update aborted")
            return False

    async def _heartbeat_turn_request(self, stop_event: asyncio.Event) -> None:
        """Refresh the turn request TTL while processing a turn.

        Originally from worker.py lines 602-619
        """
        try:
            while not stop_event.is_set():
                await asyncio.sleep(self.config.turn_request_heartbeat_seconds)
                if stop_event.is_set():
                    break
                await self.redis.expire(
                    self._turn_request_key(),
                    self.config.turn_request_ttl_seconds,
                )
                await self.redis.hset(
                    self._turn_request_key(),
                    mapping={"heartbeat_at": _utc_now().isoformat()},
                )
        except asyncio.CancelledError:
            return

    def _should_act_spontaneously(self) -> bool:
        """Determine if agent should act without new events.

        Originally from worker.py lines 620-645

        Checks time since last action against the spontaneous action interval.

        Returns:
            bool: True if spontaneous action should be triggered.
        """
        if self.session is None:
            return False

        if self.session.last_event_time is None:
            # No events yet - don't trigger spontaneous action
            return False

        now = _utc_now()
        elapsed_since_event = (now - self.session.last_event_time).total_seconds()
        if elapsed_since_event < self.config.spontaneous_action_interval:
            return False

        if self.session.last_action_time is None:
            return True

        elapsed_since_action = (now - self.session.last_action_time).total_seconds()
        return elapsed_since_action >= self.config.spontaneous_action_interval

    def _init_llm_provider(self) -> None:
        """Initialize the LLM provider from configuration.

        Originally from worker.py lines 779-805

        Uses the model specified in ChatConfig (from .env DEFAULT_MODEL) to
        create an appropriate LLMProvider instance via LanguageModelV2.
        """
        model_name = self.chat_config.default_model
        if not model_name:
            raise ValueError(
                "No model specified. Set DEFAULT_MODEL in .env or provide --model argument."
            )

        models = LanguageModelV2.index_models(self.chat_config)
        model = models.get(model_name)

        if not model:
            available = list(models.keys())[:5]
            raise ValueError(
                f"Model {model_name} not available. "
                f"Available models: {available}..."
            )

        self._llm_provider = model.llm_factory(self.chat_config)
        self.model_name = model_name
        self.model = model
        logger.info(f"Initialized LLM provider for model: {model_name}")

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
            # Create task to stop the worker
            asyncio.create_task(self.stop())

        # Register handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
