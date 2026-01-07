# andimud_mediator/service.py
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

import asyncio
import logging
import signal
from typing import Optional

import redis.asyncio as redis

from aim_mud_types import RedisKeys

from .config import MediatorConfig
from .mixins.agents import AgentsMixin
from .mixins.events import EventsMixin
from .mixins.dreamer import DreamerMixin

logger = logging.getLogger(__name__)


class MediatorService(AgentsMixin, EventsMixin, DreamerMixin):
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
