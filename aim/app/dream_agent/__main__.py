# aim/app/dream_agent/__main__.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""CLI client for Dreamer pipeline management.

Usage:
    python -m aim.app.dream_agent start <scenario> <conversation_id> [options]
    python -m aim.app.dream_agent status <pipeline_id>
    python -m aim.app.dream_agent cancel <pipeline_id>
    python -m aim.app.dream_agent resume <pipeline_id>
    python -m aim.app.dream_agent list [options]
    python -m aim.app.dream_agent watch <pipeline_id>
    python -m aim.app.dream_agent scenarios
    python -m aim.app.dream_agent models

Environment Variables:
    DEFAULT_MODEL     Default model to use if --model not specified
    PERSONA_ID        Default persona ID (default: assistant)
    USER_ID           Default user ID (default: user)
    AIM_API_KEY       API key for HTTP mode
    ANTHROPIC_API_KEY, OPENAI_API_KEY, etc. - Provider API keys
    REDIS_HOST, REDIS_PORT, REDIS_DB - Redis connection settings

Examples:
    # Start an analyst pipeline with explicit model
    python -m aim.app.dream_agent start analyst conv-123 --model claude-3-5-sonnet

    # Start using DEFAULT_MODEL from .env
    python -m aim.app.dream_agent start analyst conv-123

    # Start with specific persona and user
    python -m aim.app.dream_agent start analyst conv-123 --persona Andi --user martin

    # Start a journaler with query text
    python -m aim.app.dream_agent start journaler conv-123 \\
        --query "What did I learn about Python today?"

    # Watch pipeline progress
    python -m aim.app.dream_agent watch abc-123-def

    # List running pipelines
    python -m aim.app.dream_agent list --status running

    # Use HTTP mode (connect to server instead of Redis directly)
    python -m aim.app.dream_agent --http --base-url http://server:8000 \\
        start analyst conv-123
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from ...config import ChatConfig
from ...dreamer.scenario import load_scenario
from ...llm.models import LanguageModelV2
from .client import DreamerClient, PipelineResult


logger = logging.getLogger(__name__)


# Available scenarios (discovered from config/scenario/)
SCENARIOS = ["analyst", "journaler", "philosopher", "daydream", "summarizer"]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m aim.app.dream_agent",
        description="Dreamer pipeline client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument(
        "--env-file",
        help="Path to .env file for loading environment variables",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    # Connection mode options
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--direct",
        action="store_true",
        default=True,
        help="Connect directly to Redis (default)",
    )
    mode_group.add_argument(
        "--http",
        action="store_true",
        help="Connect via HTTP API",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for HTTP mode (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        help="API key for HTTP mode (or set AIM_API_KEY env var)",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # start command
    start_parser = subparsers.add_parser(
        "start",
        help="Start a new pipeline",
        description="Start a new Dreamer pipeline for a conversation",
    )
    start_parser.add_argument(
        "scenario",
        choices=SCENARIOS,
        help="Scenario to run",
    )
    start_parser.add_argument(
        "conversation_id",
        help="Conversation ID to process",
    )
    start_parser.add_argument(
        "--model", "-m",
        help="Model name (e.g., claude-3-5-sonnet, gpt-4o). "
             "Falls back to DEFAULT_MODEL from .env if not specified.",
    )
    start_parser.add_argument(
        "--persona", "-p",
        help="Persona ID to use (default: PERSONA_ID from .env)",
    )
    start_parser.add_argument(
        "--user", "-u",
        help="User ID to use (default: USER_ID from .env)",
    )
    start_parser.add_argument(
        "--query", "-q",
        help="Query text for journaler/philosopher scenarios",
    )
    start_parser.add_argument(
        "--wait", "-w",
        action="store_true",
        help="Wait for pipeline to complete",
    )
    start_parser.add_argument(
        "--timeout",
        type=float,
        help="Timeout in seconds when using --wait",
    )

    # status command
    status_parser = subparsers.add_parser(
        "status",
        help="Get pipeline status",
        description="Get the current status of a pipeline",
    )
    status_parser.add_argument(
        "pipeline_id",
        help="Pipeline ID to check",
    )

    # cancel command
    cancel_parser = subparsers.add_parser(
        "cancel",
        help="Cancel a pipeline",
        description="Cancel a running pipeline",
    )
    cancel_parser.add_argument(
        "pipeline_id",
        help="Pipeline ID to cancel",
    )

    # resume command
    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume a failed pipeline",
        description="Resume a failed or cancelled pipeline",
    )
    resume_parser.add_argument(
        "pipeline_id",
        help="Pipeline ID to resume",
    )

    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List pipelines",
        description="List pipelines, optionally filtered by status",
    )
    list_parser.add_argument(
        "--status", "-s",
        choices=["pending", "running", "complete", "failed"],
        help="Filter by status",
    )
    list_parser.add_argument(
        "--limit", "-n",
        type=int,
        default=20,
        help="Maximum number of pipelines to show (default: 20)",
    )

    # watch command
    watch_parser = subparsers.add_parser(
        "watch",
        help="Watch pipeline progress",
        description="Watch a pipeline until completion",
    )
    watch_parser.add_argument(
        "pipeline_id",
        help="Pipeline ID to watch",
    )
    watch_parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Poll interval in seconds (default: 2.0)",
    )
    watch_parser.add_argument(
        "--timeout",
        type=float,
        help="Timeout in seconds",
    )

    # scenarios command
    subparsers.add_parser(
        "scenarios",
        help="List available scenarios",
        description="Show available scenarios and their descriptions",
    )

    # models command
    subparsers.add_parser(
        "models",
        help="List available models",
        description="Show available language models",
    )

    return parser.parse_args()


def setup_logging(level: str):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def print_status(status, prefix: str = ""):
    """Print pipeline status in a formatted way."""
    print(f"{prefix}Pipeline: {status.pipeline_id}")
    print(f"{prefix}Scenario: {status.scenario_name}")
    print(f"{prefix}Status:   {status.status.upper()}")
    print(f"{prefix}Progress: {status.progress_percent:.1f}%")

    if status.current_step:
        print(f"{prefix}Current:  {status.current_step}")

    if status.completed_steps:
        print(f"{prefix}Completed: {', '.join(status.completed_steps)}")

    if status.failed_steps:
        print(f"{prefix}Failed:   {', '.join(status.failed_steps)}")


def print_progress(status):
    """Print progress update (single line for watch mode)."""
    steps_done = len(status.completed_steps)
    steps_total = steps_done + len(status.failed_steps)

    # Try to estimate total from scenario
    current = status.current_step or "waiting"

    bar_width = 20
    filled = int(status.progress_percent / 100 * bar_width)
    bar = "=" * filled + "-" * (bar_width - filled)

    print(
        f"\r[{bar}] {status.progress_percent:5.1f}% | "
        f"{status.status:8s} | {current}",
        end="",
        flush=True,
    )


async def cmd_start(args, client: DreamerClient, config: ChatConfig) -> int:
    """Handle start command."""
    # Resolve model: CLI arg > config.default_model > error
    model = args.model or config.default_model

    if not model:
        print(
            "ERROR: No model specified. Use --model or set DEFAULT_MODEL in .env",
            file=sys.stderr,
        )
        return 1

    # Apply persona/user overrides to config
    if args.persona:
        config.persona_id = args.persona
    if args.user:
        config.user_id = args.user

    # Validate model exists
    try:
        models = LanguageModelV2.index_models(config)
        if model not in models:
            print(f"ERROR: Model '{model}' not found.", file=sys.stderr)
            print(f"Available models: {', '.join(sorted(models.keys())[:10])}...", file=sys.stderr)
            return 1
    except Exception as e:
        logger.warning(f"Could not validate model: {e}")

    print(f"Starting {args.scenario} pipeline for conversation {args.conversation_id}...")
    print(f"Model:   {model}")
    print(f"Persona: {config.persona_id}")
    print(f"User:    {config.user_id}")

    if args.query:
        print(f"Query: {args.query}")

    result = await client.start(
        scenario_name=args.scenario,
        conversation_id=args.conversation_id,
        model_name=model,
        query_text=args.query,
    )

    if not result.success:
        print(f"ERROR: {result.error}", file=sys.stderr)
        return 1

    print(f"Pipeline started: {result.pipeline_id}")

    if args.wait:
        print("Waiting for completion...")
        try:
            async for status in client.watch(
                result.pipeline_id,
                poll_interval=2.0,
                timeout=args.timeout,
            ):
                print_progress(status)

            # Final status
            print()  # Newline after progress bar
            final = await client.get_status(result.pipeline_id)
            if final.success:
                print_status(final.status)
                return 0 if final.status.status == "complete" else 1
            else:
                print(f"ERROR: {final.error}", file=sys.stderr)
                return 1

        except TimeoutError:
            print(f"\nTimeout: Pipeline did not complete in {args.timeout}s")
            return 1

    return 0


async def cmd_status(args, client: DreamerClient) -> int:
    """Handle status command."""
    result = await client.get_status(args.pipeline_id)

    if not result.success:
        print(f"ERROR: {result.error}", file=sys.stderr)
        return 1

    print_status(result.status)
    return 0


async def cmd_cancel(args, client: DreamerClient) -> int:
    """Handle cancel command."""
    result = await client.cancel(args.pipeline_id)

    if not result.success:
        print(f"ERROR: {result.error}", file=sys.stderr)
        return 1

    print(result.message)
    return 0


async def cmd_resume(args, client: DreamerClient) -> int:
    """Handle resume command."""
    result = await client.resume(args.pipeline_id)

    if not result.success:
        print(f"ERROR: {result.error}", file=sys.stderr)
        return 1

    print(result.message)
    return 0


async def cmd_list(args, client: DreamerClient) -> int:
    """Handle list command."""
    pipelines = await client.list(status_filter=args.status, limit=args.limit)

    if not pipelines:
        print("No pipelines found.")
        return 0

    print(f"{'PIPELINE ID':<40} {'SCENARIO':<12} {'STATUS':<10} {'PROGRESS':>8}")
    print("-" * 75)

    for p in pipelines:
        print(
            f"{p.pipeline_id:<40} {p.scenario_name:<12} "
            f"{p.status:<10} {p.progress_percent:>7.1f}%"
        )

    return 0


async def cmd_watch(args, client: DreamerClient) -> int:
    """Handle watch command."""
    print(f"Watching pipeline {args.pipeline_id}...")

    try:
        async for status in client.watch(
            args.pipeline_id,
            poll_interval=args.interval,
            timeout=args.timeout,
        ):
            print_progress(status)

        # Final newline and status
        print()
        final = await client.get_status(args.pipeline_id)
        if final.success:
            print_status(final.status)
            return 0 if final.status.status == "complete" else 1
        else:
            print(f"ERROR: {final.error}", file=sys.stderr)
            return 1

    except TimeoutError:
        print(f"\nTimeout: Pipeline did not complete in {args.timeout}s")
        return 1


def cmd_scenarios(config: ChatConfig) -> int:
    """Handle scenarios command."""
    print("Available Scenarios:")
    print("-" * 60)

    scenarios_dir = Path("config/scenario")

    for name in SCENARIOS:
        try:
            scenario = load_scenario(name, scenarios_dir)
            steps = len(scenario.steps)
            desc = scenario.description or "No description"
            print(f"\n  {name}")
            print(f"    {desc}")
            print(f"    Steps: {steps}")
        except Exception as e:
            print(f"\n  {name} (error loading: {e})")

    return 0


def cmd_models(config: ChatConfig) -> int:
    """Handle models command."""
    print("Available Models:")
    print("-" * 60)

    try:
        models = LanguageModelV2.index_models(config)

        for name, model in sorted(models.items()):
            provider = model.provider
            print(f"  {name:<30} ({provider})")

    except Exception as e:
        print(f"Error loading models: {e}", file=sys.stderr)
        return 1

    return 0


async def async_main():
    """Async main entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    if not args.command:
        print("No command specified. Use --help for usage.", file=sys.stderr)
        return 1

    # Load config
    try:
        config = ChatConfig.from_env(args.env_file)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    # Handle non-client commands
    if args.command == "scenarios":
        return cmd_scenarios(config)

    if args.command == "models":
        return cmd_models(config)

    # Create client
    if args.http:
        api_key = args.api_key or os.environ.get("AIM_API_KEY")
        if not api_key:
            print("ERROR: --api-key or AIM_API_KEY required for HTTP mode", file=sys.stderr)
            return 1
        client = DreamerClient.http(args.base_url, api_key)
    else:
        client = DreamerClient.direct(config)

    # Execute command
    async with client:
        if args.command == "start":
            return await cmd_start(args, client, config)
        elif args.command == "status":
            return await cmd_status(args, client)
        elif args.command == "cancel":
            return await cmd_cancel(args, client)
        elif args.command == "resume":
            return await cmd_resume(args, client)
        elif args.command == "list":
            return await cmd_list(args, client)
        elif args.command == "watch":
            return await cmd_watch(args, client)
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return 1


def main():
    """Main entry point."""
    try:
        exit_code = asyncio.run(async_main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
