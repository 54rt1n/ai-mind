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
import json
import logging
import re
import signal
from dataclasses import replace
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as redis

from aim_mud_types import MUDAction

from .adapter import build_chat_turns
from .config import MUDConfig
from .session import MUDSession, MUDEvent, MUDTurn, RoomState, EntityState
from ...conversation.model import ConversationModel
from ...agents.roster import Roster
from ...agents.persona import Persona
from ...config import ChatConfig
from ...llm.llm import LLMProvider
from ...llm.models import LanguageModelV2
from ...tool.loader import ToolLoader
from ...tool.formatting import ToolUser, ToolCallResult
from ...utils.xml import XmlFormatter

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
    ):
        """Initialize worker with configuration and Redis client.

        Args:
            config: MUDConfig with agent identity and settings.
            redis_client: Async Redis client for stream operations.
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
        self.chat_config: Optional[ChatConfig] = None
        self.tool_user: Optional[ToolUser] = None
        self._llm_provider: Optional[LLMProvider] = None

    async def start(self) -> None:
        """Start the worker loop.

        Initializes shared resources (CVM, Roster, Persona, Session) and
        enters the main processing loop, consuming events from the agent's
        stream until stopped.
        """
        logger.info(f"Starting MUD agent worker for {self.config.agent_id}")

        # Initialize shared resources once
        # Create a ChatConfig for ConversationModel and Roster initialization
        self.chat_config = ChatConfig.from_env()
        self.chat_config.memory_path = self.config.memory_path
        self.chat_config.persona_id = self.config.persona_id
        self.chat_config.max_tokens = self.config.max_tokens
        self.chat_config.temperature = self.config.temperature

        self.cvm = ConversationModel.from_config(self.chat_config)
        self.roster = Roster.from_config(self.chat_config)
        self.persona = self.roster.get_persona(self.config.persona_id)

        # Initialize LLM provider
        self._init_llm_provider()

        # Initialize tool user with MUD tools
        self._init_tool_user()

        # Initialize session
        self.session = MUDSession(
            agent_id=self.config.agent_id,
            persona_id=self.config.persona_id,
        )

        # Set running flag
        self.running = True

        # Setup signal handlers for graceful shutdown
        self.setup_signal_handlers()

        logger.info(
            f"Worker initialized. Listening on stream: {self.config.agent_stream}"
        )
        logger.info(f"Using model: {self.config.model}")

        # Main worker loop - blocks until events arrive (push model)
        while self.running:
            try:
                # Check if paused
                if await self._is_paused():
                    logger.debug("Worker paused, sleeping...")
                    await asyncio.sleep(1)
                    continue

                # Block until events arrive (timeout for spontaneous check)
                events = await self.drain_events(
                    timeout=self.config.spontaneous_check_interval
                )

                if not events:
                    # No events - check for spontaneous action
                    if self._should_act_spontaneously():
                        logger.debug("No events, triggering spontaneous action")
                        await self.process_turn([])
                    continue

                # Process the turn with received events
                logger.info(f"Received {len(events)} events, processing turn")
                await self.process_turn(events)

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

    def _should_act_spontaneously(self) -> bool:
        """Determine if agent should act without new events.

        Checks time since last action against the spontaneous action interval.

        Returns:
            bool: True if spontaneous action should be triggered.
        """
        if self.session is None:
            return False

        if self.session.last_action_time is None:
            # Never acted - don't trigger spontaneous action
            return False

        elapsed = (_utc_now() - self.session.last_action_time).total_seconds()
        return elapsed >= self.config.spontaneous_action_interval

    async def drain_events(self, timeout: float) -> list[MUDEvent]:
        """Block until events arrive on agent's stream.

        Events are already enriched by mediator with room state.

        Args:
            timeout: Maximum seconds to block waiting for events.

        Returns:
            List of MUDEvent objects parsed from the stream.
        """
        try:
            result = await self.redis.xread(
                {self.config.agent_stream: self.session.last_event_id},
                block=int(timeout * 1000),
                count=100,
            )
        except redis.RedisError as e:
            logger.error(f"Redis error in drain_events: {e}")
            return []

        if not result:
            return []

        events = []
        for stream_name, messages in result:
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

        return events

    def _init_llm_provider(self) -> None:
        """Initialize the LLM provider from configuration.

        Uses the model specified in MUDConfig to create an appropriate
        LLMProvider instance via LanguageModelV2.
        """
        models = LanguageModelV2.index_models(self.chat_config)
        model = models.get(self.config.model)

        if not model:
            available = list(models.keys())[:5]
            raise ValueError(
                f"Model {self.config.model} not available. "
                f"Available models: {available}..."
            )

        self._llm_provider = model.llm_factory(self.chat_config)
        logger.info(f"Initialized LLM provider for model: {self.config.model}")

    def _init_tool_user(self) -> None:
        """Initialize the ToolUser with MUD tools.

        Loads MUD tool definitions from config/tools/mud.yaml and creates
        a ToolUser instance for formatting and parsing tool calls.
        """
        loader = ToolLoader(self.chat_config.tools_path)
        mud_tools = loader.load_tool_file(f"{self.chat_config.tools_path}/mud.yaml")

        if not mud_tools:
            raise ValueError("No MUD tools found in config/tools/mud.yaml")

        # Filter to player-level tools only (builder tools require permissions)
        # For now, include all tools - permission checking happens at execution
        self.tool_user = ToolUser(mud_tools)
        logger.info(f"Loaded {len(mud_tools)} MUD tools")

    def _build_system_prompt_with_tools(self) -> str:
        """Build system prompt with persona context and tool instructions.

        Combines the persona's system prompt with MUD-specific context
        and tool usage instructions.

        Returns:
            Complete system prompt string for LLM inference.
        """
        xml = XmlFormatter()

        # Add persona context
        self.persona.xml_decorator(
            xml,
            location=self.session.current_room.name if self.session.current_room else None,
        )

        # Add tool instructions
        self.tool_user.xml_decorator(xml)

        # Add MUD-specific instructions
        xml.add_element(
            "MUD", "Instructions",
            content=(
                "You are in a text-based world (MUD). "
                "Respond to events by using the available tools. "
                "You may use multiple tools in sequence. "
                "Output ONE tool call at a time as a JSON object. "
                "Think about what you want to do, then output the tool call."
            ),
            nowrap=True,
        )

        return xml.render()

    def _call_llm(self, chat_turns: list[dict[str, str]]) -> str:
        """Call the LLM with chat turns and return the response.

        Args:
            chat_turns: List of chat turns (system/user/assistant messages).

        Returns:
            The complete LLM response as a string.
        """
        chunks = []
        for chunk in self._llm_provider.stream_turns(chat_turns, self.chat_config):
            if chunk:
                chunks.append(chunk)
        return "".join(chunks)

    def _parse_tool_calls(self, response: str) -> list[ToolCallResult]:
        """Parse tool calls from LLM response.

        The LLM may output multiple tool calls. This method finds and
        parses all valid tool calls from the response.

        Args:
            response: The raw LLM response text.

        Returns:
            List of ToolCallResult objects (valid or invalid).
        """
        results = []

        # Try to extract the main tool call
        result = self.tool_user.process_response(response)
        if result.is_valid:
            results.append(result)
        elif result.error:
            logger.warning(f"Tool call parsing failed: {result.error}")

        return results

    def _extract_thinking(self, response: str) -> str:
        """Extract thinking/reasoning from LLM response.

        Looks for content before tool calls or in <think> tags.

        Args:
            response: The raw LLM response text.

        Returns:
            The thinking portion of the response.
        """
        # Check for <think>...</think> tags
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if think_match:
            return think_match.group(1).strip()

        # Otherwise, take content before the first JSON object
        json_start = response.find("{")
        if json_start > 0:
            return response[:json_start].strip()

        return ""

    async def _emit_actions(self, actions: list[MUDAction]) -> None:
        """Emit actions to the Redis mud:actions stream.

        Args:
            actions: List of MUDAction objects to emit.
        """
        for action in actions:
            try:
                data = action.to_redis_dict(self.config.agent_id)
                await self.redis.xadd(
                    self.config.action_stream,
                    {"data": json.dumps(data)},
                )
                logger.info(
                    f"Emitted action: {action.tool} -> {action.to_command()}"
                )
            except redis.RedisError as e:
                logger.error(f"Failed to emit action {action.tool}: {e}")

    async def process_turn(self, events: list[MUDEvent]) -> None:
        """Process a batch of events into a single agent turn.

        Implements the full turn processing pipeline:
        1. Update session context from events
        2. Build chat turns from session state
        3. Call LLM to generate response
        4. Parse tool calls from response
        5. Convert to MUDActions and emit to Redis
        6. Create turn record and add to session history

        Args:
            events: List of MUDEvent objects to process.
        """
        logger.info(f"Processing turn with {len(events)} events")

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
            if latest.room_state:
                self.session.current_room = RoomState.from_dict(latest.room_state)
            if latest.entities_present:
                self.session.entities_present = [
                    EntityState.from_dict(e) for e in latest.entities_present
                ]

        # Step 2: Build chat turns from session state
        chat_turns = build_chat_turns(self.session, self.persona)

        # Replace system prompt with tool-augmented version
        if chat_turns and chat_turns[0]["role"] == "system":
            chat_turns[0]["content"] = self._build_system_prompt_with_tools()

        # Step 3: Call LLM to generate response
        thinking = ""
        actions_taken: list[MUDAction] = []

        try:
            response = self._call_llm(chat_turns)
            logger.debug(f"LLM response: {response[:500]}...")

            # Step 4: Extract thinking and parse tool calls
            thinking = self._extract_thinking(response)
            tool_results = self._parse_tool_calls(response)

            # Step 5: Convert to MUDActions
            for result in tool_results:
                if result.is_valid:
                    action = MUDAction(
                        tool=result.function_name,
                        args=result.arguments,
                    )
                    actions_taken.append(action)
                    logger.info(
                        f"Parsed action: {action.tool}({action.args})"
                    )

            # Step 6: Emit actions to Redis
            if actions_taken:
                await self._emit_actions(actions_taken)
            else:
                logger.info("No valid actions to emit")

        except Exception as e:
            logger.error(f"Error during LLM inference: {e}", exc_info=True)
            thinking = f"[ERROR] LLM inference failed: {e}"

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


async def run_worker(config: MUDConfig) -> None:
    """Entry point for running a MUD agent worker.

    Creates Redis client, initializes the worker, and starts the loop.

    Args:
        config: MUDConfig with connection settings and agent identity.
    """
    # Create Redis client from URL
    redis_client = redis.from_url(
        config.redis_url,
        decode_responses=False,
    )

    # Create and start worker
    worker = MUDAgentWorker(config, redis_client)

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
        "--memory-path",
        help="Base path for memory storage (default: memory/{persona_id})",
    )
    parser.add_argument(
        "--spontaneous-interval",
        type=float,
        default=300.0,
        help="Seconds of silence before spontaneous action (default: 300)",
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
    setup_logging(args.log_level)

    logger.info(f"Initializing MUD agent worker for {args.agent_id}...")

    # Build configuration
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

    try:
        # Run the async worker
        asyncio.run(run_worker(config))
    except KeyboardInterrupt:
        logger.info("Worker stopped by user (KeyboardInterrupt)")
    except Exception as e:
        logger.exception(f"Worker error: {e}")
        raise


if __name__ == "__main__":
    main()
