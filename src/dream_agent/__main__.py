# aim/app/dream_agent/__main__.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""CLI client for Dreamer pipeline management.

Usage:
    python -m aim.app.dream_agent start <scenario> <conversation_id> [options]
    python -m aim.app.dream_agent restart <conversation_id> <branch> <step_id> [options]
    python -m aim.app.dream_agent inspect <conversation_id> <branch>
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
    # Start an analysis_dialogue pipeline with explicit model
    python -m aim.app.dream_agent start analysis_dialogue conv-123 --model claude-3-5-sonnet

    # Start using DEFAULT_MODEL from .env
    python -m aim.app.dream_agent start analysis_dialogue conv-123

    # Start with specific persona and user
    python -m aim.app.dream_agent start analysis_dialogue conv-123 --persona Andi --user martin

    # Start a journaler with query text
    python -m aim.app.dream_agent start journaler conv-123 \\
        --query "What did I learn about Python today?"

    # Watch pipeline progress
    python -m aim.app.dream_agent watch abc-123-def

    # List running pipelines
    python -m aim.app.dream_agent list --status running

    # Inspect a conversation branch to see restart options
    python -m aim.app.dream_agent inspect conv-123 0

    # Restart a pipeline from step #5 (scenario auto-inferred)
    python -m aim.app.dream_agent restart conv-123 0 5 --model claude-3-5-sonnet

    # Restart using step name instead of number
    python -m aim.app.dream_agent restart conv-123 0 reflection --model claude-3-5-sonnet

    # Restart with explicit scenario
    python -m aim.app.dream_agent restart conv-123 0 final_journal \\
        --scenario analysis_dialogue --model claude-3-5-sonnet

    # Use HTTP mode (connect to server instead of Redis directly)
    python -m aim.app.dream_agent --http --base-url http://server:8000 \\
        start analysis_dialogue conv-123
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from aim.config import ChatConfig
from aim.dreamer.core.scenario import load_scenario
from aim.llm.models import LanguageModelV2

from .client import DreamerClient, PipelineResult


logger = logging.getLogger(__name__)


# Available scenarios (discovered from config/scenario/)
SCENARIOS = [
    "analysis_dialogue",
    "journaler_dialogue",
    "philosopher_dialogue",
    "daydream_dialogue",
    "critique_dialogue",
    "researcher_dialogue",
    "summarizer",
]


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
        "--guidance", "-g",
        help="Guidance text for the pipeline",
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

    # restart command
    restart_parser = subparsers.add_parser(
        "restart",
        help="Restart a pipeline from a specific step",
        description="Restart a scenario pipeline from a specific step using existing conversation data",
    )
    restart_parser.add_argument(
        "conversation_id",
        help="Conversation ID to restart from",
    )
    restart_parser.add_argument(
        "branch",
        type=int,
        help="Branch number to restart from",
    )
    restart_parser.add_argument(
        "step",
        help="Step number or step ID to restart from (this step will be re-executed)",
    )
    restart_parser.add_argument(
        "--model", "-m",
        help="Model name (e.g., claude-3-5-sonnet). "
             "Falls back to DEFAULT_MODEL from .env if not specified.",
    )
    restart_parser.add_argument(
        "--scenario", "-s",
        choices=SCENARIOS,
        help="Scenario name (auto-inferred from conversation if not specified)",
    )
    restart_parser.add_argument(
        "--persona", "-p",
        help="Persona ID to use",
    )
    restart_parser.add_argument(
        "--user", "-u",
        help="User ID to use",
    )
    restart_parser.add_argument(
        "--query", "-q",
        help="Query text for journaler/philosopher scenarios",
    )
    restart_parser.add_argument(
        "--guidance", "-g",
        help="Guidance text for the pipeline",
    )
    restart_parser.add_argument(
        "--mood",
        help="Persona mood",
    )
    restart_parser.add_argument(
        "--all-history", "-a",
        action="store_true",
        dest="all_history",
        help="Load entire conversation history (all branches) into context",
    )
    restart_parser.add_argument(
        "--same-branch",
        action="store_true",
        dest="same_branch",
        help="Continue on the same branch instead of creating a new one",
    )
    restart_parser.add_argument(
        "--wait", "-w",
        action="store_true",
        help="Wait for pipeline to complete",
    )
    restart_parser.add_argument(
        "--timeout",
        type=float,
        help="Timeout in seconds when using --wait",
    )

    # inspect command
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a conversation branch for restart options",
        description="Show available restart points for a conversation branch",
    )
    inspect_parser.add_argument(
        "conversation_id",
        help="Conversation ID to inspect",
    )
    inspect_parser.add_argument(
        "branch",
        type=int,
        help="Branch number to inspect",
    )
    inspect_parser.add_argument(
        "--scenario", "-s",
        choices=SCENARIOS,
        help="Scenario name (required if auto-detection fails)",
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
    if args.guidance:
        print(f"Guidance: {args.guidance}")

    result = await client.start(
        scenario_name=args.scenario,
        conversation_id=args.conversation_id,
        model_name=model,
        query_text=args.query,
        guidance=args.guidance,
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


async def cmd_restart(args, client: DreamerClient, config: ChatConfig) -> int:
    """Handle restart command."""
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

    # Resolve step: could be a number or a step ID
    step_id = args.step
    step_num = None

    # Check if it's a number
    try:
        step_num = int(args.step)
    except ValueError:
        pass  # It's a step ID, not a number

    # If it's a number, we need to resolve it to a step ID
    if step_num is not None:
        # Get restart info to map step number to step ID
        info = await client.inspect(args.conversation_id, args.branch)
        if "error" in info:
            print(f"ERROR: {info['error']}", file=sys.stderr)
            return 1

        restart_points = info.get('available_restart_points', [])
        if not restart_points:
            print("ERROR: No restart points available (scenario not recognized)", file=sys.stderr)
            return 1

        # Find step by number
        found = False
        for point in restart_points:
            if isinstance(point, dict) and point.get('step_num') == step_num:
                step_id = point['step_id']
                found = True
                break

        if not found:
            print(f"ERROR: Step #{step_num} not found. Valid range: 1-{len(restart_points)}", file=sys.stderr)
            return 1

    print(f"Restarting from step '{step_id}' in conversation {args.conversation_id} (branch {args.branch})...")
    print(f"Model:   {model}")
    if args.scenario:
        print(f"Scenario: {args.scenario}")
    else:
        print("Scenario: (auto-inferring from conversation)")
    if args.all_history:
        print("Context: Loading all conversation history (all branches)")
    if args.same_branch:
        print(f"Branch:  Continuing on branch {args.branch}")
    else:
        print(f"Branch:  Creating new branch")

    result = await client.restart(
        conversation_id=args.conversation_id,
        branch=args.branch,
        step_id=step_id,
        model_name=model,
        scenario_name=args.scenario,
        query_text=args.query,
        persona_id=args.persona,
        user_id=args.user,
        guidance=args.guidance,
        mood=args.mood,
        include_all_history=args.all_history,
        same_branch=args.same_branch,
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


async def cmd_inspect(args, client: DreamerClient) -> int:
    """Handle inspect command."""
    print(f"Inspecting conversation {args.conversation_id} (branch {args.branch})...")
    print()

    info = await client.inspect(args.conversation_id, args.branch, args.scenario)

    if "error" in info:
        print(f"ERROR: {info['error']}", file=sys.stderr)
        return 1

    print(f"Conversation: {info.get('conversation_id', 'N/A')}")
    print(f"Branch:       {info.get('branch', 'N/A')}")
    print(f"Scenario:     {info.get('scenario_name', '(could not infer)')}")
    print()

    doc_types = info.get('doc_types', set())
    if doc_types:
        print(f"Document Types Found: {', '.join(sorted(doc_types))}")
    else:
        print("Document Types Found: (none)")
    print()

    step_outputs = info.get('step_outputs', {})
    if step_outputs:
        print("Step Outputs:")
        for step_id, doc_id in step_outputs.items():
            print(f"  {step_id}: {doc_id}")
    else:
        print("Step Outputs: (none found)")
    print()

    restart_points = info.get('available_restart_points', [])
    if restart_points:
        print("Available Restart Points:")
        print(f"  {'#':<4} {'Step ID':<20} {'Status'}")
        print(f"  {'-'*4} {'-'*20} {'-'*10}")
        for point in restart_points:
            # Handle both old format (list of strings) and new format (list of dicts)
            if isinstance(point, dict):
                step_num = point['step_num']
                step_id = point['step_id']
            else:
                step_num = "?"
                step_id = point
            has_output = step_id in step_outputs
            status = "completed" if has_output else "pending"
            print(f"  {step_num:<4} {step_id:<20} {status}")
        print()
        print("  Restart from any step # to re-run from that point forward.")
    else:
        print("Available Restart Points: (none - scenario not recognized)")

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
        elif args.command == "restart":
            return await cmd_restart(args, client, config)
        elif args.command == "inspect":
            return await cmd_inspect(args, client)
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
