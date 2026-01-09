# andimud_mediator/main.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Entry point for running the mediator service."""

import argparse
import asyncio
import logging
import redis.asyncio as redis
from andimud_mediator.service import MediatorService
from andimud_mediator.config import MediatorConfig

logger = logging.getLogger(__name__)


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
        "--disable-auto-analysis",
        action="store_true",
        help="Disable semi-autonomous analysis mode (enabled by default)"
    )
    parser.add_argument(
        "--auto-analysis-idle-seconds",
        type=int,
        default=300,
        help="Seconds of idle time before triggering auto-analysis (default: 300)"
    )
    parser.add_argument(
        "--auto-analysis-cooldown-seconds",
        type=int,
        default=60,
        help="Cooldown seconds between auto-analysis checks (default: 60)"
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
    config_kwargs["auto_analysis_enabled"] = not args.disable_auto_analysis
    config_kwargs["auto_analysis_idle_seconds"] = args.auto_analysis_idle_seconds
    config_kwargs["auto_analysis_cooldown_seconds"] = args.auto_analysis_cooldown_seconds
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