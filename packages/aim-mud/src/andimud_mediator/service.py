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
from datetime import datetime
from typing import Optional

import redis.asyncio as redis

from aim_mud_types.helper import _utc_now

from .config import MediatorConfig
from .mixins.agents import AgentsMixin
from .mixins.events import EventsMixin
from .mixins.dreamer import DreamerMixin
from .mixins.planner import PlannerMixin

logger = logging.getLogger(__name__)


class MediatorService(AgentsMixin, EventsMixin, DreamerMixin, PlannerMixin):
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

        # Auto-analysis state tracking
        self._last_auto_analysis_check: datetime = _utc_now()

    async def _next_sequence_id(self) -> int:
        """Get next global sequence ID for event/turn ordering.

        Uses Redis INCR for atomic, globally unique sequence IDs that persist
        across mediator restarts and work correctly with multiple mediator instances.

        Returns:
            Monotonically increasing integer for chronological ordering.
        """
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)
        return await client.next_sequence_id()

    async def start(self) -> None:
        """Start the event router task.

        Returns when the task completes or an error occurs.
        """
        logger.info("Starting Mediator service")
        logger.info(f"Event stream: {self.config.event_stream}")
        logger.info(f"Registered agents: {self.registered_agents}")

        self.running = True
        self.setup_signal_handlers()
        await self.load_last_event_id_from_hash()

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
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)
        return await client.is_paused(self.config.pause_key)

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
