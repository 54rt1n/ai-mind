# aim/app/mud/worker/utils.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Utility functions for the MUD agent worker.

Extracted from worker.py lines 66-68, 1889-2070
"""

import argparse
import asyncio
import redis.asyncio as redis
import logging
from datetime import datetime, timezone
from typing import Optional

from aim.config import ChatConfig
from .config import MUDConfig
from .worker import MUDAgentWorker


logger = logging.getLogger(__name__)


async def run_worker(config: MUDConfig, chat_config: Optional[ChatConfig] = None) -> None:
    """Entry point for running a MUD agent worker.

    Creates Redis client, initializes the worker, and starts the loop.
    Originally from worker.py lines 1889-1913

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

    Originally from worker.py lines 1915-1997

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

  # Start with custom embedding device (second GPU)
  python -m aim.app.mud.worker --agent-id andi --persona-id andi \\
      --embedding-device cuda:1

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
    parser.add_argument(
        "--embedding-device",
        help="Embedding device override (e.g., 'cpu', 'cuda:0', 'cuda:1') (default: from env EMBEDDING_DEVICE or auto-detect)",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configure logging with the specified level.

    Originally from worker.py lines 1999-2010

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

    Originally from worker.py lines 2012-2067

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

    if args.embedding_device:
        chat_config.embedding_device = args.embedding_device
        logger.info(f"Embedding device override: {args.embedding_device}")

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
