# aim/app/mud/mediator.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Mediator service for coordinating MUD events.

The Mediator is the central coordination point between Evennia and AIM agents.
It runs as an independent async service that routes events:

Event Router: Reads from mud:events, filters by room, enriches, and
distributes to per-agent streams.

Note: Action execution is handled by Evennia's ActionConsumer directly.
The mediator does not consume from mud:actions - agents emit actions
to that stream and Evennia's ActionConsumer executes them.

Architecture:
    Evennia -> mud:events -> [Mediator] -> agent:andi:events -> AIM Worker
                                       -> agent:roommate:events -> AIM Worker

    AIM Workers -> mud:actions -> Evennia ActionConsumer

Usage:
    python -m aim.app.mud.mediator --agents andi roommate
"""

import argparse
import asyncio
import json
import logging
import signal
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
import uuid

import redis.asyncio as redis

from aim_mud_types import MUDEvent, RedisKeys

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
        event_poll_timeout: Timeout in seconds for event stream polling.
        evennia_api_url: URL for Evennia REST API (for room state queries).
        pause_key: Redis key for mediator pause flag.
    """

    redis_url: str = "redis://localhost:6379"
    event_stream: str = RedisKeys.MUD_EVENTS
    event_poll_timeout: float = 5.0
    evennia_api_url: str = "http://localhost:4001"
    pause_key: str = RedisKeys.MEDIATOR_PAUSE
    # Stream bounds and turn coordination
    mud_events_maxlen: int = 1000
    agent_events_maxlen: int = 200
    turn_request_ttl_seconds: int = 120
    # Event processing hash
    events_processed_hash_max: int = 10000  # Max entries to keep in hash
    events_processed_cleanup_interval: int = 86400  # Seconds between cleanups (24h)


class MediatorService:
    """Central coordination service between Evennia and AIM agents.

    The mediator runs the Event Router task:
    - Event Router: Consumes mud:events, filters by room, enriches with
      room state, and distributes to per-agent event streams.

    Note: Action execution is handled by Evennia's ActionConsumer directly.
    The mediator does not consume from mud:actions.

    Attributes:
        redis: Async Redis client.
        config: MediatorConfig instance.
        running: Whether the service is running.
        registered_agents: Set of registered agent IDs.
        last_event_id: Last processed event stream ID.
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
        self.registered_agents: set[str] = set()

        # Round-robin turn assignment
        self._turn_index: int = 0

        # Stream position tracking
        self.last_event_id: str = "0"

        # Task references for shutdown
        self._event_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the event router task.

        Returns when the task completes or an error occurs.
        """
        logger.info("Starting Mediator service")
        logger.info(f"Event stream: {self.config.event_stream}")
        logger.info(f"Registered agents: {self.registered_agents}")

        self.running = True
        self.setup_signal_handlers()
        await self._load_last_event_id_from_hash()

        # Create event router task
        self._event_task = asyncio.create_task(
            self.run_event_router(),
            name="event_router",
        )

        try:
            # Wait for event router (or until stopped)
            await self._event_task
        except asyncio.CancelledError:
            logger.info("Mediator task cancelled")

        logger.info("Mediator service stopped")

    async def _load_last_event_id_from_hash(self) -> None:
        """Load last processed event ID from processing hash."""
        try:
            # Get all processed event IDs
            processed = await self.redis.hkeys(RedisKeys.EVENTS_PROCESSED)

            if not processed:
                logger.info("No processed events hash found, starting from 0")
                self.last_event_id = "0"
                return

            # Decode and find maximum event ID
            event_ids = []
            for key in processed:
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
                event_ids.append(key)

            # Event IDs are like "1704297296123-0" - max() works on string sort
            max_id = max(event_ids)
            self.last_event_id = max_id

            logger.info(
                "Loaded last_event_id from processed hash: %s (%d events in hash)",
                self.last_event_id,
                len(event_ids),
            )
        except Exception as e:
            logger.error(f"Failed to load last_event_id from hash: {e}")
            self.last_event_id = "0"

    async def stop(self) -> None:
        """Gracefully shutdown the mediator.

        Sets running flag to False and cancels the event router task.
        """
        logger.info("Stopping Mediator service...")
        self.running = False

        # Cancel event router task if it exists
        if self._event_task and not self._event_task.done():
            self._event_task.cancel()

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

                # Trim processed events from stream (based on hash)
                await self._trim_processed_events()

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
        # Check if already processed (idempotency check)
        try:
            already_processed = await self.redis.hexists(
                RedisKeys.EVENTS_PROCESSED,
                msg_id,
            )
            if already_processed:
                logger.debug(f"Event {msg_id} already processed, skipping")
                return
        except Exception as e:
            logger.error(f"Failed to check processed hash for {msg_id}: {e}")
            # Continue anyway - better to duplicate than to lose

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

        # Enrich event with room state
        enriched = await self.enrich_event(event)

        # Determine which agents should receive this event
        agents_to_notify = await self._agents_from_room_profile(event.room_id)
        if self.registered_agents:
            agents_to_notify = [
                a for a in agents_to_notify if a in self.registered_agents
            ]

        # Filter out self-events (actor shouldn't receive their own events)
        actor_agent_id = await self._agent_id_from_actor(event.room_id, event.actor_id)
        if actor_agent_id:
            agents_to_notify = [a for a in agents_to_notify if a != actor_agent_id]

        if not agents_to_notify:
            logger.debug(f"No agents to notify for event in room {event.room_id}")
            # Still mark as processed (we've looked at it)
            await self._mark_event_processed(msg_id, [])
            return

        # Round-robin: only assign turn to ONE available agent
        # Only the agent who gets the turn (or is first in line if all busy)
        # receives the event in their stream. This prevents multiple agents
        # from responding to the same event.
        assigned_agent: Optional[str] = None
        first_candidate: Optional[str] = None
        if agents_to_notify:
            n = len(agents_to_notify)
            for i in range(n):
                candidate = agents_to_notify[(self._turn_index + i) % n]
                if first_candidate is None:
                    first_candidate = candidate
                assigned = await self._maybe_assign_turn(candidate, reason="events")
                if assigned:
                    assigned_agent = candidate
                    self._turn_index = (self._turn_index + i + 1) % n
                    break

        # Distribute event to the agent who got the turn, or if all are busy,
        # to the first candidate (they'll process it when they finish).
        target_agent = assigned_agent or first_candidate
        if target_agent:
            stream_key = RedisKeys.agent_events(target_agent)
            await self.redis.xadd(
                stream_key,
                {"data": json.dumps(enriched)},
                maxlen=self.config.agent_events_maxlen,
                approximate=True,
            )
            logger.debug(f"Distributed event {msg_id} to {target_agent}")

        # Mark event as processed with the target agent
        await self._mark_event_processed(msg_id, [target_agent] if target_agent else [])

    async def _mark_event_processed(
        self, msg_id: str, agents: list[str]
    ) -> None:
        """Mark an event as processed in the hash.

        Args:
            msg_id: Redis stream message ID.
            agents: List of agent IDs that received the event.
        """
        try:
            timestamp = _utc_now().isoformat()
            agent_list = ",".join(agents) if agents else ""
            value = f"{timestamp}|{agent_list}"

            await self.redis.hset(
                RedisKeys.EVENTS_PROCESSED,
                msg_id,
                value,
            )

            logger.debug(
                f"Marked event {msg_id} as processed (agents: {agent_list or 'none'})"
            )
        except Exception as e:
            logger.error(f"Failed to mark event {msg_id} as processed: {e}")

    async def _trim_processed_events(self) -> None:
        """Trim processed events from the stream based on hash.

        Only trims events that are confirmed processed (in the hash).
        Uses the minimum ID in the hash as the trim point.
        """
        if self.last_event_id == "0":
            return

        try:
            # Get all processed event IDs
            processed_ids = await self.redis.hkeys(RedisKeys.EVENTS_PROCESSED)

            if not processed_ids:
                return

            # Decode and find minimum
            ids = []
            for key in processed_ids:
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
                ids.append(key)

            # Trim stream up to minimum processed event
            min_id = min(ids)
            await self.redis.xtrim(
                self.config.event_stream,
                minid=min_id,
                approximate=True,
            )
            logger.debug(f"Trimmed event stream up to {min_id}")
        except Exception as e:
            logger.error(f"Failed to trim event stream: {e}")

    async def _cleanup_processed_hash(self) -> None:
        """Remove old entries from processed hash, keeping most recent N.

        Called periodically to prevent unbounded hash growth.
        """
        try:
            # Get all processed event IDs
            processed_ids = await self.redis.hkeys(RedisKeys.EVENTS_PROCESSED)

            keep_count = self.config.events_processed_hash_max
            if len(processed_ids) <= keep_count:
                return  # No cleanup needed

            # Decode and sort
            ids = []
            for key in processed_ids:
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
                ids.append(key)

            ids.sort()  # Event IDs are sortable timestamps

            # Remove oldest entries
            to_remove = ids[:-keep_count]
            if to_remove:
                await self.redis.hdel(RedisKeys.EVENTS_PROCESSED, *to_remove)
                logger.info(
                    f"Cleaned up {len(to_remove)} old processed event entries"
                )
        except Exception as e:
            logger.error(f"Failed to cleanup processed hash: {e}")

    async def _get_turn_request(self, agent_id: str) -> dict[str, str]:
        """Fetch the current turn request hash for an agent."""
        key = RedisKeys.agent_turn_request(agent_id)
        data = await self.redis.hgetall(key)
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

    async def _maybe_assign_turn(self, agent_id: str, reason: str) -> bool:
        """Assign a turn request if none is active.

        Returns:
            True if turn was assigned, False if agent already has active turn.
        """
        current = await self._get_turn_request(agent_id)
        status = current.get("status")
        if status in ("assigned", "in_progress"):
            return False

        turn_id = uuid.uuid4().hex
        now = _utc_now().isoformat()
        key = RedisKeys.agent_turn_request(agent_id)
        payload = {
            "turn_id": turn_id,
            "status": "assigned",
            "reason": reason,
            "event_count": "1",
            "assigned_at": now,
            "heartbeat_at": now,
            "deadline_ms": str(self.config.turn_request_ttl_seconds * 1000),
        }
        await self.redis.hset(key, mapping=payload)
        await self.redis.expire(key, self.config.turn_request_ttl_seconds)
        return True

    async def _agents_from_room_profile(self, room_id: str) -> list[str]:
        """Lookup agent_ids present in a room profile."""
        if not room_id:
            return []
        try:
            raw = await self.redis.hget(RedisKeys.room_profile(room_id), "entities_present")
        except Exception as e:
            logger.error(f"Failed to read room profile for {room_id}: {e}")
            return []
        if not raw:
            return []
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            entities = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Invalid room profile entities for {room_id}")
            return []
        agent_ids: set[str] = set()
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            if entity.get("entity_type") != "ai":
                continue
            agent_id = entity.get("agent_id")
            if agent_id:
                agent_ids.add(str(agent_id))
        return list(agent_ids)

    async def _agent_id_from_actor(self, room_id: str, actor_id: str) -> Optional[str]:
        """Lookup agent_id for an actor by their entity_id (dbref).

        Args:
            room_id: The room where the event occurred.
            actor_id: The actor's entity_id (e.g., "#3").

        Returns:
            The agent_id if the actor is an AI agent, None otherwise.
        """
        if not room_id or not actor_id:
            return None
        try:
            raw = await self.redis.hget(RedisKeys.room_profile(room_id), "entities_present")
        except Exception as e:
            logger.error(f"Failed to read room profile for {room_id}: {e}")
            return None
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            entities = json.loads(raw)
        except json.JSONDecodeError:
            return None
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            if entity.get("entity_id") == actor_id:
                return entity.get("agent_id")
        return None

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

        # No enrichment when world_state is absent (worker pulls from agent profile)
        enriched["enriched"] = False
        return enriched

    def register_agent(self, agent_id: str, initial_room: str = "") -> None:
        """Register an agent with the mediator.

        Args:
            agent_id: Unique identifier for the agent.
            initial_room: Optional initial room ID for the agent.
        """
        self.registered_agents.add(agent_id)
        logger.info(
            f"Registered agent {agent_id}"
        )

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the mediator.

        Args:
            agent_id: Agent ID to unregister.
        """
        self.registered_agents.discard(agent_id)
        logger.info(f"Unregistered agent {agent_id}")

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
    config = MediatorConfig(**config_kwargs)

    logger.info(f"Redis URL: {config.redis_url}")
    logger.info(f"Event stream: {config.event_stream}")
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
