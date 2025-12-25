#!/usr/bin/env python3
# aim/app/dream_watcher/__main__.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Dream Watcher CLI - Monitor conversations and trigger pipelines.

Usage:
    python -m aim.app.dream_watcher [OPTIONS]

Examples:
    # Run watcher with default rules
    python -m aim.app.dream_watcher

    # Dry run to see what would be triggered
    python -m aim.app.dream_watcher --dry-run

    # Custom poll interval
    python -m aim.app.dream_watcher --interval 30

    # Run once and exit
    python -m aim.app.dream_watcher --once
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime

from aim.config import ChatConfig
from aim.conversation.model import ConversationModel
from aim.watcher import Watcher
from aim.watcher.rules import (
    AnalysisWithSummaryRule,
    PostSummaryAnalysisRule,
    StaleConversationRule,
    UnanalyzedConversationRule,
)

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def create_default_rules(args: argparse.Namespace, config: ChatConfig) -> list:
    """Create the default set of rules."""
    rules = []

    # Rule 1: Analysis with summary chaining
    # Routes conversations over token threshold to summarizer first
    rules.append(AnalysisWithSummaryRule(
        config=config,
        persona_id=args.persona,
        model=args.model,
        token_threshold_ratio=args.token_threshold,
        min_messages=1,
    ))

    # Rule 2: Post-summary analysis
    # Catches summarized-but-not-analyzed conversations (self-healing chain)
    rules.append(PostSummaryAnalysisRule(
        persona_id=args.persona,
        model=args.model,
    ))

    return rules


def print_stats(watcher: Watcher) -> None:
    """Print watcher statistics."""
    stats = watcher.stats
    uptime = datetime.now() - stats.started_at

    print(f"\n--- Watcher Stats ---")
    print(f"Uptime:      {uptime}")
    print(f"Cycles:      {stats.cycles}")
    print(f"Matches:     {stats.matches_found}")
    print(f"Triggered:   {stats.pipelines_triggered}")
    print(f"Errors:      {stats.errors}")
    if stats.last_cycle:
        print(f"Last cycle:  {stats.last_cycle.strftime('%H:%M:%S')}")
    print()


async def run_watcher(args: argparse.Namespace) -> int:
    """Run the watcher."""
    # Load config
    try:
        config = ChatConfig.from_env(args.env_file)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    # Create CVM
    try:
        cvm = ConversationModel.from_config(config)
    except Exception as e:
        print(f"Error creating ConversationModel: {e}", file=sys.stderr)
        return 1

    # Create rules
    rules = create_default_rules(args, config)

    # Create watcher
    watcher = Watcher(
        config=config,
        cvm=cvm,
        rules=rules,
        poll_interval=args.interval,
        dry_run=args.dry_run,
        stability_seconds=args.stability,
    )

    # Set up callbacks
    def on_match(match):
        print(f"  Match: {match.conversation_id} -> {match.scenario}")

    def on_pipeline_started(pipeline_id, match):
        print(f"  Started: {pipeline_id}")

    def on_error(error, match):
        print(f"  Error: {error}", file=sys.stderr)

    watcher.on_match = on_match
    watcher.on_pipeline_started = on_pipeline_started
    watcher.on_error = on_error

    # Print startup info
    print(f"Dream Watcher starting...")
    print(f"  Rules: {len(rules)}")
    print(f"  Interval: {args.interval}s")
    print(f"  Stability: {args.stability}s")
    print(f"  Token threshold: {args.token_threshold:.0%} of model context")
    print(f"  Dry run: {args.dry_run}")
    print()

    try:
        if args.once:
            # Run single cycle
            print("Running single cycle...")
            triggered = await watcher.run_cycle()
            print(f"Triggered {triggered} pipelines")
        else:
            # Run continuous loop
            print("Watching... (Ctrl+C to stop)")
            await watcher.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
        watcher.stop()
    finally:
        print_stats(watcher)
        await watcher.close()

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dream Watcher - Monitor conversations and trigger pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env file",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Polling interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--stability",
        type=int,
        default=120,
        help="Seconds message count must be stable before triggering (default: 120)",
    )
    parser.add_argument(
        "--token-threshold",
        type=float,
        default=0.8,
        help="Token ratio (0.0-1.0) of model context above which summarization is required (default: 0.8 = 80%%)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually trigger pipelines, just show what would be triggered",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one cycle and exit",
    )
    parser.add_argument(
        "--persona",
        type=str,
        default=None,
        help="Persona ID to use for triggered pipelines",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use for triggered pipelines",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    return asyncio.run(run_watcher(args))


if __name__ == "__main__":
    sys.exit(main())
