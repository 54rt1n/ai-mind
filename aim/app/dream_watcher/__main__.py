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

    # Enable refiner for idle-time exploration
    python -m aim.app.dream_watcher --refiner

    # Refiner with custom intervals
    python -m aim.app.dream_watcher --refiner --refiner-interval 300 --refiner-idle-threshold 600
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from typing import Optional

from aim.config import ChatConfig
from aim.conversation.model import ConversationModel
from aim.utils.redis_cache import RedisCache
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


def print_stats(watcher: Watcher, refiner_stats: Optional[dict] = None) -> None:
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

    if refiner_stats:
        print(f"\n--- Refiner Stats ---")
        print(f"Checks:      {refiner_stats.get('checks', 0)}")
        print(f"Explorations:{refiner_stats.get('explorations', 0)}")
        print(f"Errors:      {refiner_stats.get('errors', 0)}")
    print()


async def run_refiner_loop(
    engine: "ExplorationEngine",
    config: ChatConfig,
    interval: int,
    stats: dict,
    stop_event: asyncio.Event,
) -> None:
    """Run the refiner exploration loop.

    Args:
        engine: The ExplorationEngine instance
        config: ChatConfig for Redis connection
        interval: Seconds between exploration checks
        stats: Mutable dict to track refiner statistics
        stop_event: Event to signal when to stop the loop
    """
    from aim.refiner.engine import ExplorationEngine  # Type hint import

    logger.info(f"Refiner loop started, checking every {interval}s")

    while not stop_event.is_set():
        try:
            # Check if refiner is enabled via Redis
            cache = RedisCache(config)
            if not cache.is_refiner_enabled():
                logger.debug("Refiner disabled via Redis, skipping cycle")
                # Wait for the interval, but respect stop_event
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=interval)
                    break  # stop_event was set
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue loop
                continue

            stats["checks"] = stats.get("checks", 0) + 1
            logger.debug(f"Refiner check #{stats['checks']}")

            pipeline_id = await engine.run_exploration()

            if pipeline_id:
                stats["explorations"] = stats.get("explorations", 0) + 1
                logger.info(f"Refiner started exploration: {pipeline_id}")

        except Exception as e:
            stats["errors"] = stats.get("errors", 0) + 1
            logger.error(f"Refiner error: {e}", exc_info=True)

        # Wait for the interval, but respect stop_event
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            break  # stop_event was set
        except asyncio.TimeoutError:
            pass  # Normal timeout, continue loop


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

    # Refiner setup
    refiner_task: Optional[asyncio.Task] = None
    refiner_stats: Optional[dict] = None
    refiner_stop_event: Optional[asyncio.Event] = None
    dreamer_client = None

    if args.refiner:
        print(f"  Refiner: enabled")
        print(f"  Refiner interval: {args.refiner_interval}s")
        print(f"  Refiner idle threshold: {args.refiner_idle_threshold}s")
    else:
        print(f"  Refiner: disabled")
    print()

    try:
        # Set up refiner if enabled
        if args.refiner:
            from aim.refiner.engine import ExplorationEngine
            from aim.app.dream_agent.client import DreamerClient

            dreamer_client = DreamerClient.direct(config)
            await dreamer_client.connect()

            engine = ExplorationEngine(
                config=config,
                cvm=cvm,
                dreamer_client=dreamer_client,
                idle_threshold_seconds=args.refiner_idle_threshold,
                model_name=args.model,
            )

            refiner_stats = {"checks": 0, "explorations": 0, "errors": 0}
            refiner_stop_event = asyncio.Event()

            refiner_task = asyncio.create_task(
                run_refiner_loop(
                    engine=engine,
                    config=config,
                    interval=args.refiner_interval,
                    stats=refiner_stats,
                    stop_event=refiner_stop_event,
                )
            )
            logger.info("Refiner task started")

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

        # Stop refiner task if running
        if refiner_stop_event:
            refiner_stop_event.set()
        if refiner_task:
            try:
                await asyncio.wait_for(refiner_task, timeout=5.0)
            except asyncio.TimeoutError:
                refiner_task.cancel()
                try:
                    await refiner_task
                except asyncio.CancelledError:
                    pass
    finally:
        print_stats(watcher, refiner_stats)
        await watcher.close()

        # Close dreamer client if opened
        if dreamer_client:
            await dreamer_client.close()

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
        "--refiner",
        action="store_true",
        help="Enable ExplorationEngine to run during idle periods",
    )
    parser.add_argument(
        "--refiner-interval",
        type=int,
        default=300,
        help="Seconds between refiner exploration checks (default: 300)",
    )
    parser.add_argument(
        "--refiner-idle-threshold",
        type=int,
        default=300,
        help="Seconds of API inactivity before considering idle (default: 300)",
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
