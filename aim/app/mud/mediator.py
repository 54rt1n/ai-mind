# aim/app/mud/mediator.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Mediator service for coordinating MUD events and actions.

The Mediator is the central coordination point between Evennia and AIM agents.
It runs as an independent async service with two concurrent tasks:

1. Event Router: Reads from mud:events, filters by room, enriches, and
   distributes to per-agent streams.

2. Action Executor: Reads from mud:actions, applies round-robin ordering,
   and forwards to Evennia for execution.

Architecture:
    Evennia -> mud:events -> [Mediator] -> agent:andi:events -> AIM Worker
                                |      -> agent:roommate:events -> AIM Worker
                                v
    Evennia <- mud:actions <- [Mediator] <- AIM Workers

Usage:
    python -m aim.app.mud.mediator --agents andi roommate
"""

import argparse
import asyncio
import json
import logging
import signal
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import redis.asyncio as redis

from aim_mud_types import MUDEvent, MUDAction, EventType, ActorType, RedisKeys

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


@dataclass
class MediatorConfig:
    """Configuration for the Mediator service.

    Attributes:
        redis_url: Redis connection URL.
        event_stream: Stream to read raw events from Evennia.
        action_stream: Stream to read actions from agents.
        event_poll_timeout: Timeout in seconds for event stream polling.
        action_poll_timeout: Timeout in seconds for action stream polling.
        evennia_api_url: URL for Evennia REST API (for room state queries).
        pause_key: Redis key for mediator pause flag.
    """

    redis_url: str = "redis://localhost:6379"
    event_stream: str = RedisKeys.MUD_EVENTS
    action_stream: str = RedisKeys.MUD_ACTIONS
    event_poll_timeout: float = 5.0
    action_poll_timeout: float = 5.0
    evennia_api_url: str = "http://localhost:4001"
    pause_key: str = RedisKeys.MEDIATOR_PAUSE


class MediatorService:
    """Central coordination service between Evennia and AIM agents.

    The mediator runs two concurrent tasks:
    - Event Router: Consumes mud:events, filters by room, enriches with
      room state, and distributes to per-agent event streams.
    - Action Executor: Consumes mud:actions, applies round-robin ordering
      for fairness, and forwards to Evennia for execution.

    Attributes:
        redis: Async Redis client.
        config: MediatorConfig instance.
        running: Whether the service is running.
        agent_rooms: Mapping of agent_id to current room_id.
        registered_agents: Set of registered agent IDs.
        last_event_id: Last processed event stream ID.
        last_action_id: Last processed action stream ID.
        action_queues: Per-agent queues of pending actions.
        last_agent_index: Index for round-robin agent selection.
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        config: Optional[MediatorConfig] = None,
    ):
        """Initialize the mediator service.

        Args:
            redis_client: Async Redis client for stream operations.
            config: Optional MediatorConfig. Defaults to default config.
        """
        self.redis = redis_client
        self.config = config or MediatorConfig()
        self.running = False

        # Agent tracking
        self.agent_rooms: dict[str, str] = {}  # agent_id -> room_id
        self.registered_agents: set[str] = set()

        # Stream position tracking
        self.last_event_id: str = "0"
        self.last_action_id: str = "0"

        # Action execution state
        self.action_queues: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.last_agent_index: int = 0

        # Task references for shutdown
        self._event_task: Optional[asyncio.Task] = None
        self._action_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start both router and executor as concurrent tasks.

        Uses asyncio.gather() to run both tasks concurrently.
        Returns when either task completes or an error occurs.
        """
        logger.info("Starting Mediator service")
        logger.info(f"Event stream: {self.config.event_stream}")
        logger.info(f"Action stream: {self.config.action_stream}")
        logger.info(f"Registered agents: {self.registered_agents}")

        self.running = True
        self.setup_signal_handlers()

        # Create tasks for concurrent execution
        self._event_task = asyncio.create_task(
            self.run_event_router(),
            name="event_router",
        )
        self._action_task = asyncio.create_task(
            self.run_action_executor(),
            name="action_executor",
        )

        try:
            # Wait for both tasks (or until stopped)
            await asyncio.gather(
                self._event_task,
                self._action_task,
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            logger.info("Mediator tasks cancelled")

        logger.info("Mediator service stopped")

    async def stop(self) -> None:
        """Gracefully shutdown the mediator.

        Sets running flag to False and cancels both tasks.
        """
        logger.info("Stopping Mediator service...")
        self.running = False

        # Cancel tasks if they exist
        if self._event_task and not self._event_task.done():
            self._event_task.cancel()
        if self._action_task and not self._action_task.done():
            self._action_task.cancel()

    async def _is_paused(self) -> bool:
        """Check if mediator is paused via Redis flag.

        Returns:
            bool: True if paused, False if running.
        """
        value = await self.redis.get(self.config.pause_key)
        return value == b"1"

    async def run_event_router(self) -> None:
        """Read mud:events, filter by room, distribute to agents.

        Main event routing loop:
        1. XREAD from mud:events stream (blocking with timeout)
        2. For each event:
           - Parse into MUDEvent
           - Look up which agents are in that room
           - Enrich event with room state
           - XADD to each relevant agent's stream
        3. Track agent locations from movement events
        """
        logger.info("Event router started")

        while self.running:
            try:
                # Check if paused
                if await self._is_paused():
                    logger.debug("Event router paused, sleeping...")
                    await asyncio.sleep(1)
                    continue

                # Block-read from event stream
                result = await self.redis.xread(
                    {self.config.event_stream: self.last_event_id},
                    block=int(self.config.event_poll_timeout * 1000),
                    count=100,
                )

                if not result:
                    continue

                for stream_name, messages in result:
                    for msg_id, data in messages:
                        # Update last event ID
                        if isinstance(msg_id, bytes):
                            msg_id = msg_id.decode("utf-8")
                        self.last_event_id = msg_id

                        try:
                            await self._process_event(msg_id, data)
                        except Exception as e:
                            logger.error(
                                f"Error processing event {msg_id}: {e}",
                                exc_info=True,
                            )

            except asyncio.CancelledError:
                logger.info("Event router cancelled")
                break
            except Exception as e:
                logger.error(f"Error in event router: {e}", exc_info=True)
                continue

        logger.info("Event router stopped")

    async def _process_event(
        self, msg_id: str, data: dict[bytes, bytes] | dict[str, str]
    ) -> None:
        """Process a single event from the stream.

        Args:
            msg_id: Redis stream message ID.
            data: Raw event data from Redis.
        """
        # Parse event data
        raw_data = data.get(b"data") or data.get("data")
        if raw_data is None:
            logger.warning(f"Event {msg_id} missing data field")
            return

        if isinstance(raw_data, bytes):
            raw_data = raw_data.decode("utf-8")

        event_dict = json.loads(raw_data)
        event = MUDEvent.from_dict(event_dict)
        event.event_id = msg_id

        logger.debug(
            f"Processing event {msg_id}: {event.event_type.value} "
            f"from {event.actor} in {event.room_id}"
        )

        # Track agent location from movement events
        if event.event_type == EventType.MOVEMENT and event.actor_type == ActorType.AI:
            self.update_agent_room(event.actor, event.room_id)
            logger.debug(f"Updated agent {event.actor} location to {event.room_id}")

        # Enrich event with room state
        enriched = await self.enrich_event(event)

        # Determine which agents should receive this event
        agents_in_room = self.get_agents_in_room(event.room_id)
        # Also include agents with no room set (they see everything until placed)
        agents_without_room = [
            a for a in self.registered_agents if a not in self.agent_rooms
        ]
        agents_to_notify = list(set(agents_in_room + agents_without_room))

        if not agents_to_notify:
            logger.debug(f"No agents to notify for event in room {event.room_id}")
            return

        # Distribute to each relevant agent's stream
        for agent_id in agents_to_notify:
            stream_key = RedisKeys.agent_events(agent_id)
            await self.redis.xadd(
                stream_key,
                {"data": json.dumps(enriched)},
            )
            logger.debug(f"Distributed event {msg_id} to {agent_id}")

    async def enrich_event(self, event: MUDEvent) -> dict[str, Any]:
        """Add room state to event.

        Currently a placeholder that returns the event with empty room_state.
        Future implementation will query Evennia REST API for current room state.

        Args:
            event: The MUDEvent to enrich.

        Returns:
            Dictionary with event data and enrichment fields.
        """
        # Build base event dictionary
        enriched = event.to_redis_dict()
        enriched["id"] = event.event_id

        # Placeholder enrichment - actual room state query comes later
        enriched["room_state"] = {
            "room_id": event.room_id,
            "name": event.room_name,
            "description": "",
            "exits": {},
        }
        enriched["entities_present"] = []
        enriched["enriched"] = True

        return enriched

    async def run_action_executor(self) -> None:
        """Read mud:actions, round-robin execute.

        Main action execution loop:
        1. XREAD from mud:actions stream (short timeout for responsiveness)
        2. Collect actions into per-agent queues
        3. Round-robin through agents, executing one action per agent per cycle
        """
        logger.info("Action executor started")

        while self.running:
            try:
                # Check if paused
                if await self._is_paused():
                    logger.debug("Action executor paused, sleeping...")
                    await asyncio.sleep(1)
                    continue

                # Collect pending actions from stream
                result = await self.redis.xread(
                    {self.config.action_stream: self.last_action_id},
                    block=int(self.config.action_poll_timeout * 1000),
                    count=100,
                )

                if result:
                    for stream_name, messages in result:
                        for msg_id, data in messages:
                            # Update last action ID
                            if isinstance(msg_id, bytes):
                                msg_id = msg_id.decode("utf-8")
                            self.last_action_id = msg_id

                            try:
                                await self._queue_action(msg_id, data)
                            except Exception as e:
                                logger.error(
                                    f"Error queuing action {msg_id}: {e}",
                                    exc_info=True,
                                )

                # Round-robin execution
                await self._execute_round_robin()

            except asyncio.CancelledError:
                logger.info("Action executor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in action executor: {e}", exc_info=True)
                continue

        logger.info("Action executor stopped")

    async def _queue_action(
        self, msg_id: str, data: dict[bytes, bytes] | dict[str, str]
    ) -> None:
        """Queue an action for execution.

        Args:
            msg_id: Redis stream message ID.
            data: Raw action data from Redis.
        """
        raw_data = data.get(b"data") or data.get("data")
        if raw_data is None:
            logger.warning(f"Action {msg_id} missing data field")
            return

        if isinstance(raw_data, bytes):
            raw_data = raw_data.decode("utf-8")

        action = json.loads(raw_data)
        action["msg_id"] = msg_id

        agent_id = action.get("agent_id")
        if not agent_id:
            logger.warning(f"Action {msg_id} missing agent_id")
            return

        self.action_queues[agent_id].append(action)
        logger.debug(
            f"Queued action {msg_id} from {agent_id}: {action.get('command', '')}"
        )

    async def _execute_round_robin(self) -> None:
        """Execute one action per agent in round-robin order."""
        agents = list(self.action_queues.keys())
        if not agents:
            return

        # Select next agent in round-robin order
        agent_id = agents[self.last_agent_index % len(agents)]

        if self.action_queues[agent_id]:
            action = self.action_queues[agent_id].pop(0)
            await self._execute_action(action)

            # Clean up empty queues
            if not self.action_queues[agent_id]:
                del self.action_queues[agent_id]

        self.last_agent_index += 1

    async def _execute_action(self, action: dict[str, Any]) -> None:
        """Execute an action in Evennia.

        Currently a placeholder that logs the action.
        Future implementation will call Evennia REST API or execute command.

        Args:
            action: Action dictionary with agent_id, command, tool, args, etc.
        """
        agent_id = action.get("agent_id", "unknown")
        command = action.get("command", "")
        msg_id = action.get("msg_id", "unknown")

        logger.info(f"Executing action {msg_id} from {agent_id}: {command}")

        # TODO: Actual Evennia execution
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(
        #         f"{self.config.evennia_api_url}/api/execute",
        #         json={"agent_id": agent_id, "command": command}
        #     ) as resp:
        #         result = await resp.json()

    def register_agent(self, agent_id: str, initial_room: str = "") -> None:
        """Register an agent with the mediator.

        Args:
            agent_id: Unique identifier for the agent.
            initial_room: Optional initial room ID for the agent.
        """
        self.registered_agents.add(agent_id)
        if initial_room:
            self.agent_rooms[agent_id] = initial_room
        logger.info(
            f"Registered agent {agent_id}"
            + (f" in room {initial_room}" if initial_room else "")
        )

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the mediator.

        Args:
            agent_id: Agent ID to unregister.
        """
        self.registered_agents.discard(agent_id)
        self.agent_rooms.pop(agent_id, None)
        logger.info(f"Unregistered agent {agent_id}")

    def update_agent_room(self, agent_id: str, room_id: str) -> None:
        """Update agent's current room.

        Args:
            agent_id: Agent ID to update.
            room_id: New room ID for the agent.
        """
        self.agent_rooms[agent_id] = room_id
        logger.debug(f"Agent {agent_id} moved to room {room_id}")

    def get_agents_in_room(self, room_id: str) -> list[str]:
        """Get all agent IDs currently in a room.

        Args:
            room_id: Room ID to query.

        Returns:
            List of agent IDs in the specified room.
        """
        return [
            agent_id
            for agent_id, agent_room in self.agent_rooms.items()
            if agent_room == room_id
        ]

    def setup_signal_handlers(self) -> None:
        """Setup handlers for graceful shutdown.

        Registers signal handlers for SIGINT and SIGTERM.
        """

        def signal_handler(signum, frame):
            """Handle shutdown signals."""
            sig_name = signal.Signals(signum).name
            logger.info(f"Received {sig_name}, shutting down gracefully...")
            asyncio.create_task(self.stop())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def run_mediator(
    config: MediatorConfig,
    agents: list[str],
) -> None:
    """Entry point for running the mediator service.

    Creates Redis client, registers agents, and starts the service.

    Args:
        config: MediatorConfig with connection settings.
        agents: List of agent IDs to register.
    """
    # Create Redis client
    redis_client = redis.from_url(
        config.redis_url,
        decode_responses=False,
    )

    # Create mediator
    mediator = MediatorService(redis_client, config)

    # Register agents
    for agent_id in agents:
        mediator.register_agent(agent_id)

    try:
        await mediator.start()
    finally:
        await redis_client.aclose()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run MUD mediator service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start mediator with Andi agent
  python -m aim.app.mud.mediator --agents andi

  # Start with multiple agents
  python -m aim.app.mud.mediator --agents andi roommate

  # Start with custom Redis URL
  python -m aim.app.mud.mediator --redis-url redis://redis.example.com:6379 --agents andi
        """,
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=[],
        help="Agent IDs to register with the mediator",
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
        "--event-timeout",
        type=float,
        default=None,
        help=f"Event poll timeout in seconds (default: {MediatorConfig.event_poll_timeout})",
    )
    parser.add_argument(
        "--action-timeout",
        type=float,
        default=None,
        help=f"Action poll timeout in seconds (default: {MediatorConfig.action_poll_timeout})",
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
    """Main entry point for the mediator CLI.

    Parses arguments, creates configuration, and starts the service.
    """
    args = parse_args()
    setup_logging(args.log_level)

    logger.info("Initializing MUD mediator service...")

    config_kwargs = {"redis_url": args.redis_url}
    if args.event_timeout is not None:
        config_kwargs["event_poll_timeout"] = args.event_timeout
    if args.action_timeout is not None:
        config_kwargs["action_poll_timeout"] = args.action_timeout
    config = MediatorConfig(**config_kwargs)

    logger.info(f"Redis URL: {config.redis_url}")
    logger.info(f"Event stream: {config.event_stream}")
    logger.info(f"Action stream: {config.action_stream}")
    logger.info(f"Agents: {args.agents}")

    try:
        asyncio.run(run_mediator(config, args.agents))
    except KeyboardInterrupt:
        logger.info("Mediator stopped by user (KeyboardInterrupt)")
    except Exception as e:
        logger.exception(f"Mediator error: {e}")
        raise


if __name__ == "__main__":
    main()
