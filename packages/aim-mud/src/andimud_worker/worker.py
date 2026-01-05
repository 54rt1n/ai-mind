# aim/app/mud/worker.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Worker process for consuming MUD events and processing agent turns.

This module implements the MUDAgentWorker, which follows the DreamerWorker
pattern from aim/dreamer/worker.py. It consumes events from a per-agent
Redis stream and processes them into agent turns.

The worker is designed for the push distribution model:
- Events are pushed by the mediator to agent-specific streams
- The worker blocks until events arrive (with timeout for spontaneous checks)
- Events are already enriched with room state by the mediator

Usage:
    python -m aim.app.mud.worker --agent-id andi --persona-id andi
"""

import argparse
import asyncio
import contextlib
import json
import logging
import re
import signal
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import uuid

import yaml
import redis.asyncio as redis

from aim_mud_types import MUDAction, RedisKeys

from .adapter import (
    build_current_context,
    build_system_prompt,
)
from .config import MUDConfig
from .session import MUDSession, MUDEvent, MUDTurn, RoomState, EntityState, WorldState
from aim_mud_types.world_state import InventoryItem
from aim.conversation.model import ConversationModel
from aim.agents.roster import Roster
from aim.chat.manager import ChatManager
from aim.agents.persona import Persona
from aim.config import ChatConfig
from aim.dreamer.executor import extract_think_tags
from aim.llm.llm import LLMProvider, is_retryable_error
from aim.llm.models import LanguageModelV2
from aim.tool.loader import ToolLoader
from aim.tool.formatting import ToolUser
from .memory import MUDMemoryRetriever
from .conversation import MUDConversationManager
from .strategy import MUDDecisionStrategy, MUDResponseStrategy
from .utils import sanitize_response

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


class MUDAgentWorker:
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
        self._decision_tool_user: Optional[ToolUser] = None
        self._decision_strategy: Optional[MUDDecisionStrategy] = None
        self._agent_action_spec: Optional[dict] = None
        self._chat_manager: Optional[ChatManager] = None

        # Phase 2 response strategy
        self._response_strategy: Optional[MUDResponseStrategy] = None

        # Turn request tracking
        self._last_turn_request_id: Optional[str] = None

    async def start(self) -> None:
        """Start the worker loop.

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
                except Exception as e:
                    logger.error(f"Error during assigned turn {turn_id}: {e}", exc_info=True)
                    # Restore event position so next drain re-reads these events
                    if saved_last_event_id is not None:
                        self.session.last_event_id = saved_last_event_id
                        logger.info("Restored last_event_id to %s for retry", saved_last_event_id)
                    await self._set_turn_request_state(turn_id, "fail", message=str(e))
                finally:
                    heartbeat_stop.set()
                    heartbeat_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await heartbeat_task

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

        Sets the running flag to False, allowing the current turn
        to complete before shutting down.
        """
        logger.info("Stopping worker...")
        self.running = False

    async def _is_paused(self) -> bool:
        """Check if worker is paused via Redis flag.

        Returns:
            bool: True if paused, False if running.
        """
        value = await self.redis.get(self.config.pause_key)
        return value == b"1"

    def _turn_request_key(self) -> str:
        """Return the Redis key for this agent's turn request."""
        return RedisKeys.agent_turn_request(self.config.agent_id)

    def _agent_profile_key(self) -> str:
        """Return the Redis key for this agent's profile hash."""
        return RedisKeys.agent_profile(self.config.agent_id)

    async def _load_agent_profile(self) -> None:
        """Load agent profile state from Redis."""
        if not self.session:
            return
        data = await self.redis.hgetall(self._agent_profile_key())
        if not data:
            # Initialize profile shell
            await self._update_agent_profile(agent_id=self.config.agent_id)
            return
        decoded: dict[str, str] = {}
        for k, v in data.items():
            if isinstance(k, bytes):
                k = k.decode("utf-8")
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            decoded[str(k)] = str(v)

        last_event_id = decoded.get("last_event_id")
        if last_event_id:
            self.session.last_event_id = last_event_id

    async def _load_agent_world_state(self) -> tuple[Optional[str], Optional[str]]:
        """Load inventory and room pointer from agent profile.

        Returns:
            Tuple of (room_id, character_id) when available.
        """
        if not self.session:
            return None, None
        try:
            data = await self.redis.hgetall(self._agent_profile_key())
        except redis.RedisError as e:
            logger.error(f"Redis error loading agent profile: {e}")
            return None, None
        if not data:
            return None, None

        def _decode(value):
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return value

        room_id = _decode(data.get(b"room_id") or data.get("room_id"))
        character_id = _decode(data.get(b"character_id") or data.get("character_id"))

        inventory_raw = _decode(data.get(b"inventory") or data.get("inventory"))
        inventory_items: list = []
        if inventory_raw:
            try:
                inventory_items = json.loads(inventory_raw)
            except json.JSONDecodeError:
                logger.warning("Invalid inventory JSON in agent profile")

        home = _decode(data.get(b"home") or data.get("home"))
        time_val = _decode(data.get(b"time") or data.get("time"))

        inventory = [
            InventoryItem.from_dict(i)
            for i in inventory_items
            if isinstance(i, dict)
        ]

        if self.session.world_state is None:
            self.session.world_state = WorldState(
                inventory=inventory,
                home=home,
                time=time_val,
            )
        else:
            self.session.world_state.inventory = inventory
            self.session.world_state.home = home
            self.session.world_state.time = time_val

        return room_id, character_id

    async def _load_room_profile(self, room_id: Optional[str], character_id: Optional[str]) -> None:
        """Load room profile snapshot from Redis and merge into world_state."""
        if not self.session or not room_id:
            return
        try:
            data = await self.redis.hgetall(RedisKeys.room_profile(room_id))
        except redis.RedisError as e:
            logger.error(f"Redis error loading room profile {room_id}: {e}")
            return
        if not data:
            return

        def _decode(value):
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return value

        room_state_raw = _decode(data.get(b"room_state") or data.get("room_state"))
        entities_raw = _decode(data.get(b"entities_present") or data.get("entities_present"))

        room_state = None
        entities_present: list[EntityState] = []

        if room_state_raw:
            try:
                room_state = RoomState.from_dict(json.loads(room_state_raw))
            except Exception:
                logger.warning("Invalid room_state JSON in room profile")

        if entities_raw:
            try:
                parsed = json.loads(entities_raw)
                if isinstance(parsed, list):
                    entities_present = [
                        EntityState.from_dict(e) for e in parsed if isinstance(e, dict)
                    ]
            except Exception:
                logger.warning("Invalid entities_present JSON in room profile")

        if character_id:
            for entity in entities_present:
                if entity.entity_id == character_id:
                    entity.is_self = True
                    break

        if self.session.world_state is None:
            self.session.world_state = WorldState(
                room_state=room_state,
                entities_present=entities_present,
            )
        else:
            if room_state is not None:
                self.session.world_state.room_state = room_state
            self.session.world_state.entities_present = entities_present

        # Update chat manager location XML for unified memory queries
        if self._chat_manager and self.session.world_state and self.session.world_state.room_state:
            room_state_only = WorldState(
                room_state=self.session.world_state.room_state,
                entities_present=self.session.world_state.entities_present,
            )
            self._chat_manager.current_location = room_state_only.to_xml(include_self=False)

        if room_state is not None:
            self.session.current_room = room_state
        self.session.entities_present = entities_present

    async def _update_agent_profile(self, **fields: str) -> None:
        """Update agent profile fields in Redis."""
        payload = {"updated_at": _utc_now().isoformat()}
        payload.update({k: v for k, v in fields.items() if v is not None})
        await self.redis.hset(self._agent_profile_key(), mapping=payload)

    async def _get_turn_request(self) -> dict[str, str]:
        """Fetch the current turn request hash."""
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
    ) -> None:
        """Update turn request state and refresh TTL."""
        now = _utc_now().isoformat()
        payload = {
            "turn_id": turn_id,
            "status": status,
            "heartbeat_at": now,
        }
        if message:
            payload["message"] = message
        await self.redis.hset(self._turn_request_key(), mapping=payload)
        await self.redis.expire(
            self._turn_request_key(),
            self.config.turn_request_ttl_seconds,
        )

    async def _heartbeat_turn_request(self, stop_event: asyncio.Event) -> None:
        """Refresh the turn request TTL while processing a turn."""
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

    async def drain_events(self, timeout: float) -> list[MUDEvent]:
        """Block until events arrive on agent's stream.

        Events are already enriched by mediator with room state.

        Args:
            timeout: Maximum seconds to block waiting for events.

        Returns:
            List of MUDEvent objects parsed from the stream.
        """
        events: list[MUDEvent] = []
        max_id = None

        def _parse_result(result) -> None:
            for _stream_name, messages in result:
                for msg_id, data in messages:
                    # Update last event ID for resumption
                    # msg_id may be bytes or str depending on Redis client config
                    if isinstance(msg_id, bytes):
                        msg_id = msg_id.decode("utf-8")
                    self.session.last_event_id = msg_id

                    # Parse event data
                    try:
                        # Data field contains the JSON payload
                        raw_data = data.get(b"data") or data.get("data")
                        if raw_data is None:
                            logger.warning(f"Event {msg_id} missing data field")
                            continue

                        if isinstance(raw_data, bytes):
                            raw_data = raw_data.decode("utf-8")

                        enriched = json.loads(raw_data)
                        event = MUDEvent.from_dict(enriched)
                        event.event_id = msg_id
                        events.append(event)

                        logger.debug(
                            f"Parsed event {msg_id}: {event.event_type.value} "
                            f"from {event.actor}"
                        )

                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.error(f"Failed to parse event {msg_id}: {e}")
                        continue

        # Snapshot the max stream id at drain start to avoid chasing new events
        try:
            info = await self.redis.xinfo_stream(self.config.agent_stream)
            max_id = info.get("last-generated-id") or info.get(b"last-generated-id")
            if isinstance(max_id, bytes):
                max_id = max_id.decode("utf-8")
        except redis.RedisError as e:
            logger.error(f"Redis error in drain_events (xinfo): {e}")
            return []

        if not max_id or max_id == "0":
            return []

        # Drain events up to snapshot max_id
        start_id = self.session.last_event_id or "0"
        min_id = f"({start_id}" if start_id != "0" else "-"
        while True:
            try:
                result = await self.redis.xrange(
                    self.config.agent_stream,
                    min=min_id,
                    max=max_id,
                    count=100,
                )
            except redis.RedisError as e:
                logger.error(f"Redis error in drain_events (xrange): {e}")
                break

            if not result:
                break

            _parse_result([(self.config.agent_stream, result)])
            last_msg_id = self.session.last_event_id
            if not last_msg_id or last_msg_id == max_id:
                break
            min_id = f"({last_msg_id}"

        if self.session and self.session.last_event_id:
            await self._update_agent_profile(last_event_id=self.session.last_event_id)

        return events

    async def _drain_with_settle(self) -> list[MUDEvent]:
        """Drain events with settling delay for cascading events.

        Drains events, waits settle_seconds, drains again.
        Repeats until a drain returns zero events. This allows
        cascading events (e.g., someone entering a room and immediately
        speaking) to be batched together.

        Returns:
            All accumulated events from multiple drains.
        """
        settle_time = self.config.event_settle_seconds
        all_events: list[MUDEvent] = []
        drain_count = 0

        while True:
            events = await self.drain_events(timeout=0)
            drain_count += 1
            if not events:
                # No new events - cascade has settled
                if all_events:
                    logger.info(
                        "Event cascade settled after %.1fs with %d total events",
                        settle_time,
                        len(all_events),
                    )
                elif drain_count == 1:
                    logger.warning(
                        "First drain returned 0 events - turn assigned but stream empty?"
                    )
                break

            all_events.extend(events)
            logger.info(
                "Drained %d events (total %d), waiting %.1fs for more",
                len(events),
                len(all_events),
                settle_time,
            )
            await asyncio.sleep(settle_time)

        return all_events

    def _init_llm_provider(self) -> None:
        """Initialize the LLM provider from configuration.

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

    def _init_decision_tools(self) -> None:
        """Initialize phase 1 decision tools (speak/move)."""
        if self.chat_config is None:
            raise ValueError("ChatConfig must be initialized before loading tools")

        tool_file = Path(self.config.decision_tool_file)
        if not tool_file.is_absolute():
            # Allow passing just a filename; resolve relative to tools_path
            if "/" not in str(tool_file):
                tool_file = Path(self.chat_config.tools_path) / tool_file
            elif not tool_file.exists():
                candidate = Path(self.chat_config.tools_path) / tool_file.name
                if candidate.exists():
                    tool_file = candidate

        loader = ToolLoader(self.chat_config.tools_path)
        tools = loader.load_tool_file(str(tool_file))
        if not tools:
            raise ValueError(f"No tools loaded from {tool_file}")

        self._decision_tool_user = ToolUser(tools)
        logger.info(
            "Loaded %d phase 1 decision tools from %s",
            len(tools),
            tool_file,
        )

    def _init_agent_action_spec(self) -> None:
        """Load @agent action specification from YAML."""
        tool_file = Path(self.config.agent_tool_file)
        if not tool_file.is_absolute():
            if "/" not in str(tool_file):
                tool_file = Path(self.chat_config.tools_path) / tool_file
            elif not tool_file.exists():
                candidate = Path(self.chat_config.tools_path) / tool_file.name
                if candidate.exists():
                    tool_file = candidate

        if not tool_file.exists():
            raise ValueError(f"Agent tool file not found: {tool_file}")

        try:
            with open(tool_file, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except Exception as e:
            raise ValueError(f"Failed to load agent tool file {tool_file}: {e}") from e

        self._agent_action_spec = data
        logger.info("Loaded @agent action spec from %s", tool_file)

    def _agent_action_list(self) -> list[dict]:
        """Return the list of actions from the @agent spec."""
        if not self._agent_action_spec:
            return []
        actions = self._agent_action_spec.get("actions", [])
        return actions if isinstance(actions, list) else []

    def _build_agent_guidance(self, user_guidance: str) -> str:
        """Build guidance for @agent action selection."""
        spec = self._agent_action_spec or {}
        instructions = spec.get("instructions", "")
        actions = self._agent_action_list()

        lines = []
        if instructions:
            lines.append(instructions)
        lines.append("Include any other text inside of the JSON response instead.")
        lines.append("You are in your memory palace. Respond as yourself.")
        lines.append(
            "For describe, write paragraph-long descriptions infused with your personality."
        )
        if user_guidance:
            lines.append(f"User guidance: {user_guidance}")

        if actions:
            action_names = ", ".join([a.get("name", "") for a in actions if a.get("name")])
            lines.append(f"Allowed actions: {action_names}")
            lines.append('Output format: {"action": "<name>", ...}')
            for action in actions:
                name = action.get("name")
                desc = action.get("description", "")
                examples = action.get("examples") or action.get("parameters", {}).get("examples")
                if name:
                    if desc:
                        lines.append(f"- {name}: {desc}")
                    if examples:
                        first = examples[0]
                        if isinstance(first, dict):
                            lines.append(f"Example: {json.dumps(first)}")

        if self._decision_strategy and self.session:
            lines.extend(self._decision_strategy._build_agent_action_hints(self.session))
        return "\n".join([line for line in lines if line])

    def _resolve_target_name(self, target_id: str) -> str:
        """Resolve a target id to a display name for emotes."""
        if not target_id or not self.session:
            return target_id or "object"

        world_state = self.session.world_state
        if world_state and world_state.room_state:
            room = world_state.room_state
            if room.room_id == target_id:
                return room.name or "room"

        # Check entities present
        if world_state:
            for entity in world_state.entities_present:
                if entity.entity_id == target_id:
                    return entity.name or target_id

            for item in world_state.inventory:
                if getattr(item, "item_id", None) == target_id:
                    return item.name or target_id

        return target_id

    def _resolve_move_location(self, location: Optional[str]) -> Optional[str]:
        """Validate and normalize a move location against current exits."""
        if not location:
            return None

        room = self.session.current_room if self.session else None
        exits = room.exits if room else None
        if not exits:
            return location

        if location in exits:
            return location

        lowered = location.lower()
        for exit_name in exits.keys():
            if exit_name.lower() == lowered:
                return exit_name

        return None

    async def _decide_action(self, idle_mode: bool) -> tuple[Optional[str], dict, str, str, str]:
        """Phase 1 decision: choose speak or move via tool call.

        Returns:
            Tuple of (tool_name, args, raw_response, thinking, cleaned_text)
        """
        if not self._decision_tool_user:
            raise ValueError("Decision tools not initialized")

        turns = await self._decision_strategy.build_turns(
            persona=self.persona,
            session=self.session,
            idle_mode=idle_mode,
        )
        last_response = ""
        last_cleaned = ""
        last_thinking = ""

        for attempt in range(self.config.decision_max_retries):
            response = await self._call_llm(turns)
            last_response = response
            logger.debug("Phase1 LLM response: %s...", response[:500])
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Phase1 LLM response (full):\n%s", response)
            cleaned, think_content = extract_think_tags(response)
            cleaned = sanitize_response(cleaned)
            last_cleaned = cleaned.strip()
            last_thinking = think_content or ""

            result = self._decision_tool_user.process_response(response)
            if result.is_valid and result.function_name:
                tool_name = result.function_name
                args = result.arguments or {}

                if tool_name == "move":
                    location = args.get("location") or args.get("direction")
                    resolved = self._resolve_move_location(location)
                    if not resolved:
                        # Get valid exits for guidance
                        valid_exits = []
                        if self.session.current_room and self.session.current_room.exits:
                            valid_exits = list(self.session.current_room.exits.keys())
                        exits_str = ", ".join(valid_exits) if valid_exits else "none available"
                        error_guidance = (
                            f"Invalid move location '{location}'. "
                            f"Valid exits are: {exits_str}. "
                            f"Please try again with a valid exit, or use {{\"speak\": {{}}}} to respond instead."
                        )
                        turns.append({"role": "assistant", "content": response})
                        turns.append({"role": "user", "content": error_guidance})
                        logger.warning(
                            "Invalid move location '%s'; retrying with guidance (attempt %d/%d)",
                            location, attempt + 1, self.config.decision_max_retries,
                        )
                        continue
                    return "move", {"location": resolved}, last_response, last_thinking, last_cleaned

                if tool_name == "speak":
                    # Args are used to enhance memory query, not as action parameters
                    return "speak", args, last_response, last_thinking, last_cleaned

                if tool_name == "wait":
                    # Explicit choice to do nothing this turn
                    return "wait", {}, last_response, last_thinking, last_cleaned

                if tool_name == "take":
                    obj = args.get("object")
                    # Get available items in the room
                    room_objects = self._get_room_objects()
                    if obj and obj.lower() in [o.lower() for o in room_objects]:
                        return "take", args, last_response, last_thinking, last_cleaned
                    # Invalid - give guidance
                    objects_str = ", ".join(room_objects) if room_objects else "nothing here to take"
                    error_guidance = (
                        f"Cannot take '{obj}'. "
                        f"Available items: {objects_str}. "
                        f"Please try again with a valid item, or use {{\"speak\": {{}}}} to respond instead."
                    )
                    turns.append({"role": "assistant", "content": response})
                    turns.append({"role": "user", "content": error_guidance})
                    logger.warning(
                        "Invalid take object '%s'; retrying with guidance (attempt %d/%d)",
                        obj, attempt + 1, self.config.decision_max_retries,
                    )
                    continue

                if tool_name == "drop":
                    obj = args.get("object")
                    # Get inventory items
                    inventory = self._get_inventory_items()
                    if obj and obj.lower() in [i.lower() for i in inventory]:
                        return "drop", args, last_response, last_thinking, last_cleaned
                    # Invalid - give guidance
                    inventory_str = ", ".join(inventory) if inventory else "nothing in inventory"
                    error_guidance = (
                        f"Cannot drop '{obj}'. "
                        f"Your inventory: {inventory_str}. "
                        f"Please try again with an item you're carrying, or use {{\"speak\": {{}}}} to respond instead."
                    )
                    turns.append({"role": "assistant", "content": response})
                    turns.append({"role": "user", "content": error_guidance})
                    logger.warning(
                        "Invalid drop object '%s'; retrying with guidance (attempt %d/%d)",
                        obj, attempt + 1, self.config.decision_max_retries,
                    )
                    continue

                if tool_name == "give":
                    obj = args.get("object")
                    target = args.get("target")
                    # Get inventory and valid targets
                    inventory = self._get_inventory_items()
                    valid_targets = self._get_valid_give_targets()

                    obj_valid = obj and obj.lower() in [i.lower() for i in inventory]
                    target_valid = target and target.lower() in [t.lower() for t in valid_targets]

                    if obj_valid and target_valid:
                        return "give", args, last_response, last_thinking, last_cleaned

                    # Build specific guidance
                    errors = []
                    if not obj_valid:
                        inventory_str = ", ".join(inventory) if inventory else "nothing"
                        errors.append(f"Cannot give '{obj}'. Your inventory: {inventory_str}.")
                    if not target_valid:
                        targets_str = ", ".join(valid_targets) if valid_targets else "no one here"
                        errors.append(f"Cannot give to '{target}'. People present: {targets_str}.")

                    error_guidance = (
                        " ".join(errors) + " "
                        f"Please try again with valid item and target, or use {{\"speak\": {{}}}} to respond instead."
                    )
                    turns.append({"role": "assistant", "content": response})
                    turns.append({"role": "user", "content": error_guidance})
                    logger.warning(
                        "Invalid give (object='%s', target='%s'); retrying with guidance (attempt %d/%d)",
                        obj, target, attempt + 1, self.config.decision_max_retries,
                    )
                    continue

                # Unexpected tool - give guidance and retry
                error_guidance = (
                    f"Unknown tool '{tool_name}'. "
                    f"Available tools are: speak, wait, move, take, drop, give. "
                    f"Please try again with a valid tool."
                )
                turns.append({"role": "assistant", "content": response})
                turns.append({"role": "user", "content": error_guidance})
                logger.warning(
                    "Unexpected tool '%s'; retrying with guidance (attempt %d/%d)",
                    tool_name, attempt + 1, self.config.decision_max_retries,
                )
                continue

            # Invalid tool call format - give guidance and retry
            error_guidance = (
                f"Invalid response format: {result.error}. "
                f"Please respond with exactly one JSON tool call, e.g. {{\"speak\": {{}}}} or {{\"move\": {{\"location\": \"north\"}}}}."
            )
            turns.append({"role": "assistant", "content": response})
            turns.append({"role": "user", "content": error_guidance})
            logger.warning(
                "Invalid tool call (attempt %d/%d): %s",
                attempt + 1,
                self.config.decision_max_retries,
                result.error,
            )

        # Fallback: return "confused" if tool call keeps failing after all retries
        logger.warning("All %d decision attempts failed; returning confused", self.config.decision_max_retries)
        return "confused", {}, last_response, last_thinking, last_cleaned

    def _is_superuser_persona(self) -> bool:
        """Return True if persona should have builder tools."""
        if not self.persona:
            return False

        attrs = self.persona.attributes or {}
        role = str(attrs.get("mud_role", "")).lower()
        perms = attrs.get("mud_permissions")

        if role in ("superuser", "builder"):
            return True

        if isinstance(perms, list):
            return any(str(p).lower() in ("superuser", "builder") for p in perms)

        if isinstance(perms, str):
            return perms.lower() in ("superuser", "builder")

        return False

    def _get_room_objects(self) -> list[str]:
        """Get names of objects available to take in the current room."""
        objects: list[str] = []
        world_state = self.session.world_state if self.session else None
        if world_state:
            for entity in world_state.entities_present:
                if entity.is_self:
                    continue
                # Objects are entities that aren't players/AIs/NPCs
                if entity.entity_type not in ("player", "ai", "npc"):
                    if entity.name:
                        objects.append(entity.name)
        else:
            # Fall back to session entities
            if self.session:
                for entity in self.session.entities_present:
                    if entity.is_self:
                        continue
                    if entity.entity_type not in ("player", "ai", "npc"):
                        if entity.name:
                            objects.append(entity.name)
        return objects

    def _get_inventory_items(self) -> list[str]:
        """Get names of items in the agent's inventory."""
        items: list[str] = []
        world_state = self.session.world_state if self.session else None
        if world_state:
            for item in world_state.inventory:
                if item.name:
                    items.append(item.name)
        return items

    def _get_valid_give_targets(self) -> list[str]:
        """Get names of valid targets for giving items (players, AIs, NPCs, objects)."""
        targets: list[str] = []
        world_state = self.session.world_state if self.session else None
        if world_state:
            for entity in world_state.entities_present:
                if entity.is_self:
                    continue
                if entity.entity_type in ("player", "ai", "npc", "object"):
                    if entity.name:
                        targets.append(entity.name)
        else:
            # Fall back to session entities
            if self.session:
                for entity in self.session.entities_present:
                    if entity.is_self:
                        continue
                    if entity.entity_type in ("player", "ai", "npc", "object"):
                        if entity.name:
                            targets.append(entity.name)
        return targets

    async def _call_llm(self, chat_turns: list[dict[str, str]], max_retries: int = 3) -> str:
        """Call the LLM with chat turns and return the response.

        Implements retry logic with exponential backoff for transient errors,
        following the pattern from aim/refiner/engine.py.

        Args:
            chat_turns: List of chat turns (system/user/assistant messages).
            max_retries: Maximum number of retry attempts for transient errors.

        Returns:
            The complete LLM response as a string.

        Raises:
            Exception: If max retries exceeded or non-retryable error occurs.
        """
        for attempt in range(max_retries):
            try:
                chunks = []
                for chunk in self._llm_provider.stream_turns(chat_turns, self.chat_config):
                    if chunk:
                        chunks.append(chunk)
                return "".join(chunks)

            except Exception as e:
                logger.error(f"LLM error (attempt {attempt + 1}/{max_retries}): {e}")

                # Check if error is retryable and we have retries left
                if is_retryable_error(e) and attempt < max_retries - 1:
                    delay = min(30 * (2 ** attempt), 120)  # 30s, 60s, 120s max
                    logger.info(f"Retryable error, waiting {delay}s before retry...")
                    await asyncio.sleep(delay)
                else:
                    raise

        # Should not reach here, but just in case
        raise RuntimeError(f"LLM call failed after {max_retries} attempts")


    async def _emit_actions(self, actions: list[MUDAction]) -> None:
        """Emit actions to the Redis mud:actions stream.

        Args:
            actions: List of MUDAction objects to emit.
        """
        for action in actions:
            try:
                command = action.to_command().strip()
                if not command:
                    logger.warning(
                        "Skipping action with empty command: %s(%s)",
                        action.tool,
                        action.args,
                    )
                    continue
                data = action.to_redis_dict(self.config.agent_id)
                action_id = await self.redis.xadd(
                    self.config.action_stream,
                    {"data": json.dumps(data)},
                )
                if isinstance(action_id, bytes):
                    action_id = action_id.decode("utf-8")
                await self._update_agent_profile(last_action_id=str(action_id))
                logger.info(
                    f"Emitted action: {action.tool} -> {command}"
                )
            except redis.RedisError as e:
                logger.error(f"Failed to emit action {action.tool}: {e}")

        # Trim old actions from stream (keep last 1000)
        try:
            await self.redis.xtrim(
                self.config.action_stream,
                maxlen=1000,
                approximate=True,
            )
        except redis.RedisError as e:
            logger.warning(f"Failed to trim action stream: {e}")

    @staticmethod
    def _normalize_response(response: str) -> str:
        """Normalize a free-text response for emission."""
        if not response:
            return ""

        stripped = response.strip()
        if not stripped:
            return ""

        lines = [line.rstrip() for line in stripped.splitlines()]
        normalized: list[str] = []
        blank = False
        for line in lines:
            if not line.strip():
                if not blank:
                    normalized.append("")
                    blank = True
                continue
            normalized.append(line)
            blank = False

        return "\n".join(normalized).strip()

    async def _is_fresh_session(self) -> bool:
        """Check if this is a fresh session (no conversation history)."""
        if not self.conversation_manager:
            return True
        total = await self.conversation_manager.get_total_tokens()
        return total == 0

    @staticmethod
    def _has_emotional_state_header(response: str) -> bool:
        """Check if response starts with emotional state header after think block."""
        # Remove think block first
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        # Check if it starts with [== ... Emotional State ... ==]
        return bool(re.match(r'\[==.*Emotional State.*==\]', cleaned, re.IGNORECASE))

    @staticmethod
    def _extract_speak_text_from_tool_call(response: str) -> Optional[str]:
        """Extract speak text if the response is a tool-call-like JSON blob."""
        if not response:
            return None

        stripped = response.strip()
        if not stripped:
            return None

        parsed = None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            try:
                parsed = ToolUser([])._extract_tool_call(stripped)
            except Exception:
                parsed = None

        if not isinstance(parsed, dict):
            return None

        if "speak" not in parsed:
            return None

        payload = parsed.get("speak")
        if isinstance(payload, str):
            return payload

        if isinstance(payload, dict):
            for key in ("text", "say", "message", "content"):
                value = payload.get(key)
                if isinstance(value, str):
                    return value

        return None

    def _parse_agent_action_response(self, response: str) -> tuple[Optional[str], dict, str]:
        """Parse @agent JSON response into (action, args, error)."""
        cleaned, _think = extract_think_tags(response)
        cleaned = sanitize_response(cleaned)
        text = cleaned.strip()
        parsed = None
        json_text = None

        try:
            parsed = json.loads(text)
            json_text = text
        except json.JSONDecodeError:
            # Try to extract a JSON object from mixed text
            json_candidates = []
            brace_depth = 0
            start_idx = None
            for i, char in enumerate(text):
                if char == "{":
                    if brace_depth == 0:
                        start_idx = i
                    brace_depth += 1
                elif char == "}":
                    brace_depth -= 1
                    if brace_depth == 0 and start_idx is not None:
                        json_candidates.append(text[start_idx : i + 1])
                        start_idx = None
            for candidate in reversed(json_candidates):
                try:
                    parsed = json.loads(candidate)
                    json_text = candidate.strip()
                    break
                except json.JSONDecodeError:
                    continue

        if not isinstance(parsed, dict):
            return None, {}, "Could not parse JSON"

        # Preferred format: {"action": "<name>", ...}
        if "action" in parsed:
            action = parsed.get("action")
            if not isinstance(action, str):
                return None, {}, "Action must be a string"
            args = {k: v for k, v in parsed.items() if k != "action"}
            return action.lower(), args, ""

        # Alternate tool-call format: {"describe": {...}}
        if len(parsed) == 1:
            action = next(iter(parsed))
            args = parsed.get(action)
            if isinstance(action, str) and isinstance(args, dict):
                return action.lower(), args, ""

        return None, {}, "Missing action field"

    async def process_turn(self, events: list[MUDEvent]) -> None:
        """Process a batch of events into a single agent turn.

        Implements the full turn processing pipeline:
        1. Update session context from events
        2. Build chat turns from session state
        3. Call LLM to generate response
        4. Normalize free-text response
        5. Emit a single `speak` action (or noop)
        6. Create turn record and add to session history

        Args:
            events: List of MUDEvent objects to process.
        """
        logger.info(f"Processing turn with {len(events)} events")

        # Refresh world state snapshot from agent + room profiles
        room_id, character_id = await self._load_agent_world_state()
        if not room_id and self.session.current_room and self.session.current_room.room_id:
            room_id = self.session.current_room.room_id
        if not room_id and events:
            room_id = events[-1].room_id
        await self._load_room_profile(room_id, character_id)

        # Log event details for debugging
        for event in events:
            logger.info(
                f"  Event: {event.event_type.value} | "
                f"Actor: {event.actor} | "
                f"Room: {event.room_name or event.room_id} | "
                f"Content: {event.content[:100] if event.content else '(none)'}..."
            )

        # Step 1: Update session context from events
        self.session.pending_events = events
        if events:
            latest = events[-1]
            self.session.last_event_time = latest.timestamp

        # Push user turn to conversation list
        if self.conversation_manager and events:
            await self.conversation_manager.push_user_turn(
                events=events,
                world_state=self.session.world_state,
                room_id=self.session.current_room.room_id if self.session.current_room else None,
                room_name=self.session.current_room.name if self.session.current_room else None,
            )

        # Step 2: Phase 1 decision (speak/move)
        idle_mode = len(events) == 0
        thinking_parts: list[str] = []
        actions_taken: list[MUDAction] = []
        raw_responses: list[str] = []

        try:
            decision_tool, decision_args, decision_raw, decision_thinking, decision_cleaned = (
                await self._decide_action(idle_mode=idle_mode)
            )
            # Phase 1 thinking captured for debugging, but not the raw JSON response
            if decision_thinking:
                thinking_parts.append(decision_thinking)

            if decision_tool == "move":
                action = MUDAction(tool="move", args=decision_args)
                actions_taken.append(action)
                await self._emit_actions(actions_taken)

            elif decision_tool == "take":
                obj = decision_args.get("object")
                if obj:
                    action = MUDAction(tool="get", args={"object": obj})
                    actions_taken.append(action)
                    await self._emit_actions(actions_taken)
                else:
                    logger.warning("Phase1 take missing object; no action emitted")

            elif decision_tool == "drop":
                obj = decision_args.get("object")
                if obj:
                    action = MUDAction(tool="drop", args={"object": obj})
                    actions_taken.append(action)
                    await self._emit_actions(actions_taken)
                else:
                    logger.warning("Phase1 drop missing object; no action emitted")

            elif decision_tool == "give":
                obj = decision_args.get("object")
                target = decision_args.get("target")
                if obj and target:
                    action = MUDAction(tool="give", args={"object": obj, "target": target})
                    actions_taken.append(action)
                    await self._emit_actions(actions_taken)
                else:
                    logger.warning("Phase1 give missing object or target; no action emitted")

            elif decision_tool == "wait":
                # Explicit choice to do nothing - skip Phase 2
                logger.info("Phase 1 decided to wait; no action this turn")

            elif decision_tool == "speak":
                # Phase 2: full response turn with memory via response strategy
                coming_online = await self._is_fresh_session()

                # Extract memory query from speak args (enhances CVM search)
                memory_query = decision_args.get("query") or decision_args.get("focus") or ""
                if memory_query:
                    logger.info(f"Phase 2 memory query: {memory_query[:100]}...")

                # Build user input with current context (events/guidance)
                # Events already pushed to conversation history, so exclude here
                user_input = build_current_context(
                    self.session,
                    idle_mode=idle_mode,
                    guidance=None,
                    coming_online=coming_online,
                    include_events=False,
                )

                # Use response strategy for full context (consciousness + memory)
                chat_turns = await self._response_strategy.build_turns(
                    persona=self.persona,
                    user_input=user_input,
                    session=self.session,
                    coming_online=coming_online,
                    max_context_tokens=self.model.max_tokens,
                    max_output_tokens=self.chat_config.max_tokens,
                    memory_query=memory_query,
                )

                # Retry loop for emotional state header validation
                max_format_retries = 3
                cleaned_response = ""
                for format_attempt in range(max_format_retries):
                    response = await self._call_llm(chat_turns)
                    raw_responses.append(response)
                    logger.debug(f"LLM response: {response[:500]}...")
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("LLM response (full):\n%s", response)

                    cleaned_response, think_content = extract_think_tags(response)
                    cleaned_response = sanitize_response(cleaned_response)
                    cleaned_response = cleaned_response.strip()
                    if think_content:
                        thinking_parts.append(think_content)

                    # Validate emotional state header
                    if self._has_emotional_state_header(cleaned_response):
                        break  # Valid format, continue

                    # Missing header - retry with stronger guidance
                    logger.warning(
                        f"Response missing Emotional State header (attempt {format_attempt + 1}/{max_format_retries})"
                    )
                    if format_attempt < max_format_retries - 1:
                        persona_name = self.session.persona_id if self.session else "Agent"
                        format_guidance = (
                            f"\n\n[Gentle reminder from your link: Please begin with your emotional state, "
                            f"e.g. [== {persona_name}'s Emotional State: <list of your +Emotions+> ==] then continue with prose.]"
                        )
                        # Append guidance to the last user turn
                        if chat_turns and chat_turns[-1]["role"] == "user":
                            chat_turns[-1]["content"] += format_guidance
                        else:
                            chat_turns.append({"role": "user", "content": format_guidance})

                extracted_text = self._extract_speak_text_from_tool_call(cleaned_response)
                if extracted_text is not None:
                    logger.debug(
                        "Phase2 response looked like a tool call; extracted speak text (%d chars)",
                        len(extracted_text),
                    )
                normalized = self._normalize_response(
                    extracted_text if extracted_text is not None else cleaned_response
                )

                if normalized:
                    action = MUDAction(tool="speak", args={"text": normalized})
                    actions_taken.append(action)
                    logger.info("Prepared speak action (%d chars)", len(normalized))
                    await self._emit_actions(actions_taken)
                else:
                    logger.info("No response content to emit")

            elif decision_tool == "confused":
                # Phase 1 failed to parse a valid decision - emit confused emote
                logger.info("Phase 1 returned confused; emitting confused emote")
                action = MUDAction(tool="emote", args={"action": "looks confused."})
                actions_taken.append(action)
                await self._emit_actions(actions_taken)

            else:
                # Unknown decision tool - log warning and skip
                logger.warning(
                    "Unknown phase 1 decision tool '%s'; skipping turn",
                    decision_tool,
                )

        except Exception as e:
            logger.error(f"Error during LLM inference: {e}", exc_info=True)
            thinking_parts.append(f"[ERROR] LLM inference failed: {e}")
            raw_responses.append(f"[ERROR] LLM inference failed: {e}")
            # Emit a graceful emote when LLM fails
            action = MUDAction(tool="emote", args={"action": "was at a loss for words."})
            actions_taken.append(action)
            await self._emit_actions(actions_taken)

        thinking = "\n\n".join(thinking_parts).strip()

        # Push assistant turn to conversation list - ONLY for speak actions
        # Non-speak actions (move/take/drop/give) are mechanical tool calls,
        # not narrative content, so we don't save them to conversation history.
        if self.conversation_manager:
            for action in actions_taken:
                if action.tool == "speak":
                    speak_text = action.args.get("text", "")
                    if speak_text:
                        await self.conversation_manager.push_assistant_turn(
                            content=speak_text,
                            think=thinking if thinking else None,
                            actions=actions_taken,
                        )
                    break

        # Step 7: Create turn record
        turn = MUDTurn(
            timestamp=_utc_now(),
            events_received=events,
            room_context=self.session.current_room,
            entities_context=self.session.entities_present,
            thinking=thinking,
            actions_taken=actions_taken,
        )

        # Add turn to session history
        self.session.add_turn(turn)
        self.session.clear_pending_events()

        logger.info(
            f"Turn processed. Actions: {len(actions_taken)}. "
            f"Session now has {len(self.session.recent_turns)} turns"
        )

    async def process_agent_turn(self, events: list[MUDEvent], user_guidance: str) -> None:
        """Process a guided @agent turn using mud_agent.yaml action schema."""
        logger.info(f"Processing @agent turn with {len(events)} events")

        # Refresh world state snapshot from agent + room profiles
        room_id, character_id = await self._load_agent_world_state()
        if not room_id and self.session.current_room and self.session.current_room.room_id:
            room_id = self.session.current_room.room_id
        if not room_id and events:
            room_id = events[-1].room_id
        await self._load_room_profile(room_id, character_id)

        for event in events:
            logger.info(
                f"  Event: {event.event_type.value} | "
                f"Actor: {event.actor} | "
                f"Room: {event.room_name or event.room_id} | "
                f"Content: {event.content[:100] if event.content else '(none)'}..."
            )

        self.session.pending_events = events
        if events:
            latest = events[-1]
            self.session.last_event_time = latest.timestamp

        # Push user turn to conversation list
        if self.conversation_manager and events:
            await self.conversation_manager.push_user_turn(
                events=events,
                world_state=self.session.world_state,
                room_id=self.session.current_room.room_id if self.session.current_room else None,
                room_name=self.session.current_room.name if self.session.current_room else None,
            )

        guidance = self._build_agent_guidance(user_guidance)
        coming_online = await self._is_fresh_session()

        # Build user input with current context and agent guidance
        # Events already pushed to conversation history, so exclude here
        user_input = build_current_context(
            self.session,
            idle_mode=False,
            guidance=guidance,
            coming_online=coming_online,
            include_events=False,
        )

        # Use response strategy for full context (consciousness + memory)
        chat_turns = await self._response_strategy.build_turns(
            persona=self.persona,
            user_input=user_input,
            session=self.session,
            coming_online=coming_online,
            max_context_tokens=self.model.max_tokens,
            max_output_tokens=self.chat_config.max_tokens,
        )

        actions_taken: list[MUDAction] = []
        raw_responses: list[str] = []
        thinking_parts: list[str] = []

        allowed = {a.get("name", "").lower() for a in self._agent_action_list() if a.get("name")}

        try:
            for attempt in range(self.config.decision_max_retries):
                response = await self._call_llm(chat_turns)
                raw_responses.append(response)
                cleaned, think_content = extract_think_tags(response)
                cleaned = sanitize_response(cleaned)
                cleaned = cleaned.strip()
                if think_content:
                    thinking_parts.append(think_content)

                action, args, error = self._parse_agent_action_response(cleaned)
                if not action:
                    logger.warning(
                        "Invalid @agent response (attempt %d/%d): %s",
                        attempt + 1,
                        self.config.decision_max_retries,
                        error,
                    )
                    continue

                if allowed and action not in allowed:
                    logger.warning(
                        "Invalid @agent action '%s' (attempt %d/%d)",
                        action,
                        attempt + 1,
                        self.config.decision_max_retries,
                    )
                    continue

                # Valid action -> emit
                if action == "speak":
                    text = args.get("text", "")
                    # Validate emotional state header for speak actions
                    if not self._has_emotional_state_header(text):
                        logger.warning(
                            "Agent speak missing Emotional State header (attempt %d/%d)",
                            attempt + 1,
                            self.config.decision_max_retries,
                        )
                        if attempt < self.config.decision_max_retries - 1:
                            persona_name = self.session.persona_id if self.session else "Agent"
                            format_guidance = (
                                f"\n\n[Gentle reminder from your link: Please begin with your emotional state, "
                                f"e.g. [== {persona_name}'s Emotional State: <list of your +Emotion+> ==] then continue with prose.]"
                            )
                            if chat_turns and chat_turns[-1]["role"] == "user":
                                chat_turns[-1]["content"] += format_guidance
                            else:
                                chat_turns.append({"role": "user", "content": format_guidance})
                            continue
                    normalized = self._normalize_response(text)
                    if normalized:
                        action_obj = MUDAction(tool="speak", args={"text": normalized})
                        actions_taken.append(action_obj)
                        await self._emit_actions(actions_taken)
                    else:
                        logger.info("Agent speak had no text; no action emitted")
                    break

                if action == "move":
                    location = args.get("location") or args.get("direction")
                    resolved = self._resolve_move_location(location)
                    if resolved:
                        action_obj = MUDAction(tool="move", args={"location": resolved})
                        actions_taken.append(action_obj)
                        await self._emit_actions(actions_taken)
                    else:
                        logger.warning("Agent move missing/invalid location")
                    break

                if action == "take":
                    obj = args.get("object")
                    if obj:
                        action_obj = MUDAction(tool="get", args={"object": obj})
                        actions_taken.append(action_obj)
                        await self._emit_actions(actions_taken)
                    else:
                        logger.warning("Agent take missing object")
                    break

                if action == "drop":
                    obj = args.get("object")
                    if obj:
                        action_obj = MUDAction(tool="drop", args={"object": obj})
                        actions_taken.append(action_obj)
                        await self._emit_actions(actions_taken)
                    else:
                        logger.warning("Agent drop missing object")
                    break

                if action == "give":
                    obj = args.get("object")
                    target = args.get("target")
                    if obj and target:
                        action_obj = MUDAction(tool="give", args={"object": obj, "target": target})
                        actions_taken.append(action_obj)
                        await self._emit_actions(actions_taken)
                    else:
                        logger.warning("Agent give missing object or target")
                    break

                if action == "describe":
                    target = args.get("target")
                    description = args.get("description")
                    if target and description:
                        action_obj = MUDAction(
                            tool="describe",
                            args={"target": target, "description": description},
                        )
                        actions_taken.append(action_obj)
                        object_name = self._resolve_target_name(target)
                        emote_text = f"adjusted the {object_name}."
                        actions_taken.append(
                            MUDAction(tool="emote", args={"action": emote_text})
                        )
                        await self._emit_actions(actions_taken)
                    else:
                        logger.warning("Agent describe missing target or description")
                    break

        except Exception as e:
            logger.error(f"Error during @agent LLM inference: {e}", exc_info=True)
            thinking_parts.append(f"[ERROR] LLM inference failed: {e}")
            raw_responses.append(f"[ERROR] LLM inference failed: {e}")
            # Emit a graceful emote when LLM fails
            action = MUDAction(tool="emote", args={"action": "was at a loss for words."})
            actions_taken.append(action)
            await self._emit_actions(actions_taken)

        thinking = "\n\n".join(thinking_parts).strip()

        # Push assistant turn to conversation list - ONLY for speak actions
        # Non-speak actions (move/take/drop/give/describe) are mechanical tool calls,
        # not narrative content, so we don't save them to conversation history.
        if self.conversation_manager:
            for action in actions_taken:
                if action.tool == "speak":
                    speak_text = action.args.get("text", "")
                    if speak_text:
                        await self.conversation_manager.push_assistant_turn(
                            content=speak_text,
                            think=thinking if thinking else None,
                            actions=actions_taken,
                        )
                    break

        turn = MUDTurn(
            timestamp=_utc_now(),
            events_received=events,
            room_context=self.session.current_room,
            entities_context=self.session.entities_present,
            thinking=thinking,
            actions_taken=actions_taken,
        )

        self.session.add_turn(turn)
        self.session.clear_pending_events()

        logger.info(
            f"@agent turn processed. Actions: {len(actions_taken)}. "
            f"Session now has {len(self.session.recent_turns)} turns"
        )

    def setup_signal_handlers(self) -> None:
        """Setup handlers for graceful shutdown.

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


async def run_worker(config: MUDConfig, chat_config: Optional[ChatConfig] = None) -> None:
    """Entry point for running a MUD agent worker.

    Creates Redis client, initializes the worker, and starts the loop.

    Args:
        config: MUDConfig with connection settings and agent identity.
        chat_config: Optional pre-loaded ChatConfig with API keys and paths.
            If None, will be loaded from environment in worker.start().
    """
    # Create Redis client from URL
    redis_client = redis.from_url(
        config.redis_url,
        decode_responses=False,
    )

    # Create and start worker
    worker = MUDAgentWorker(config, redis_client, chat_config)

    try:
        await worker.start()
    finally:
        # Cleanup Redis connection
        await redis_client.aclose()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run MUD agent worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start Andi agent worker
  python -m aim.app.mud.worker --agent-id andi --persona-id andi

  # Start with custom env file
  python -m aim.app.mud.worker --agent-id andi --persona-id andi \\
      --env-file /path/to/.env

  # Start with custom Redis URL
  python -m aim.app.mud.worker --agent-id andi --persona-id andi \\
      --redis-url redis://redis.example.com:6379

  # Start with debug logging
  python -m aim.app.mud.worker --agent-id andi --persona-id andi --log-level DEBUG
        """,
    )
    parser.add_argument(
        "--agent-id",
        required=True,
        help="Unique identifier for this agent in the MUD",
    )
    parser.add_argument(
        "--persona-id",
        required=True,
        help="ID of the persona configuration to use",
    )
    parser.add_argument(
        "--env-file",
        help="Path to .env file for loading environment variables",
    )
    parser.add_argument(
        "--redis-url",
        default="redis://localhost:6379",
        help="Redis connection URL (default: redis://localhost:6379)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging (overrides --log-level)",
    )
    parser.add_argument(
        "--memory-path",
        help="Base path for memory storage (default: memory/{persona_id})",
    )
    parser.add_argument(
        "--spontaneous-interval",
        type=float,
        default=300.0,
        help="Seconds of silence before spontaneous action (default: 300)",
    )
    parser.add_argument(
        "--model",
        help="Model override (default: from env DEFAULT_MODEL)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature override (default: from env TEMPERATURE)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Max tokens override (default: from env MAX_TOKENS)",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configure logging with the specified level.

    Args:
        level: Logging level as string (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """Main entry point for the MUD agent worker CLI.

    Parses arguments, loads configuration, and starts the worker loop.
    Handles graceful shutdown on KeyboardInterrupt.
    """
    args = parse_args()
    log_level = "DEBUG" if args.verbose else args.log_level
    setup_logging(log_level)

    logger.info(f"Initializing MUD agent worker for {args.agent_id}...")

    # Load environment configuration first (loads .env file and API keys)
    # This follows the pattern from aim/app/dreamer/__main__.py
    chat_config = ChatConfig.from_env(args.env_file)
    logger.info("Loaded environment configuration")

    # Apply CLI overrides to ChatConfig (only when explicitly provided)
    if args.model:
        chat_config.default_model = args.model
        logger.info(f"Model override: {args.model}")

    if args.temperature is not None:
        chat_config.temperature = args.temperature
        logger.info(f"Temperature override: {args.temperature}")

    if args.max_tokens is not None:
        chat_config.max_tokens = args.max_tokens
        logger.info(f"Max tokens override: {args.max_tokens}")

    # Build MUD-specific configuration (identity, redis, timing, memory only)
    config = MUDConfig(
        agent_id=args.agent_id,
        persona_id=args.persona_id,
        redis_url=args.redis_url,
        spontaneous_action_interval=args.spontaneous_interval,
    )

    if args.memory_path:
        config.memory_path = args.memory_path

    logger.info(f"Agent ID: {config.agent_id}")
    logger.info(f"Persona ID: {config.persona_id}")
    logger.info(f"Redis URL: {config.redis_url}")
    logger.info(f"Agent stream: {config.agent_stream}")
    logger.info(f"Default model: {chat_config.default_model}")

    try:
        # Run the async worker, passing the pre-loaded chat_config
        asyncio.run(run_worker(config, chat_config))
    except KeyboardInterrupt:
        logger.info("Worker stopped by user (KeyboardInterrupt)")
    except Exception as e:
        logger.exception(f"Worker error: {e}")
        raise


if __name__ == "__main__":
    main()
