# aim/app/dreamer/__main__.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""CLI entrypoint for running a Dreamer worker process.

This module provides a command-line interface for starting a Dreamer worker
that consumes and executes pipeline steps from Redis queues.

Usage:
    python -m aim.app.dreamer [options]

Options:
    --env-file PATH     Path to .env file (optional)
    --log-level LEVEL   Logging level (DEBUG, INFO, WARNING, ERROR)
    --redis-host HOST   Redis host (default: from env)
    --redis-port PORT   Redis port (default: from env)
"""

import argparse
import asyncio
import logging
import sys

from aim.config import ChatConfig
from aim_legacy.dreamer.server.worker import run_worker

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Run Dreamer pipeline worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start worker with default settings
  python -m aim.app.dreamer

  # Start worker with custom env file
  python -m aim.app.dreamer --env-file /path/to/.env

  # Start worker with debug logging
  python -m aim.app.dreamer --log-level DEBUG

  # Override Redis connection settings
  python -m aim.app.dreamer --redis-host redis.example.com --redis-port 6380
        """
    )
    parser.add_argument(
        "--env-file",
        help="Path to .env file for loading environment variables"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--redis-host",
        help="Redis host override (default: from env REDIS_HOST or localhost)"
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        help="Redis port override (default: from env REDIS_PORT or 6379)"
    )
    parser.add_argument(
        "--redis-db",
        type=int,
        help="Redis database number (default: from env REDIS_DB or 0)"
    )
    parser.add_argument(
        "--redis-password",
        help="Redis password (default: from env REDIS_PASSWORD)"
    )
    return parser.parse_args()


def setup_logging(level: str):
    """Configure logging with the specified level.

    Args:
        level: Logging level as string (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    """Main entry point for the Dreamer worker CLI.

    Parses arguments, loads configuration, and starts the worker loop.
    Handles graceful shutdown on KeyboardInterrupt and other exceptions.
    """
    args = parse_args()
    setup_logging(args.log_level)

    logger.info("Initializing Dreamer worker...")

    try:
        # Load configuration from environment
        config = ChatConfig.from_env(args.env_file)

        # Override Redis settings if provided via CLI
        if args.redis_host:
            config.redis_host = args.redis_host
            logger.info(f"Redis host override: {args.redis_host}")

        if args.redis_port:
            config.redis_port = args.redis_port
            logger.info(f"Redis port override: {args.redis_port}")

        if args.redis_db is not None:
            config.redis_db = args.redis_db
            logger.info(f"Redis DB override: {args.redis_db}")

        if args.redis_password:
            config.redis_password = args.redis_password
            logger.info("Redis password provided via CLI")

        # Log connection info
        logger.info(
            f"Connecting to Redis at {config.redis_host}:{config.redis_port} "
            f"(db={config.redis_db})"
        )

        logger.info("Starting Dreamer worker...")

        # Run the async worker
        asyncio.run(run_worker(config))

    except KeyboardInterrupt:
        logger.info("Worker stopped by user (KeyboardInterrupt)")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Worker error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
