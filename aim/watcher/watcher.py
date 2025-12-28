# aim/watcher/watcher.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Main watcher loop for monitoring conversations and triggering pipelines.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable

import redis.asyncio as redis

from aim.config import ChatConfig
from aim.conversation.model import ConversationModel
from aim.watcher.rules import Rule, RuleMatch
from aim.watcher.stability import StabilityTracker
from aim.app.dream_agent.client import DreamerClient
from aim.utils.redis_cache import RedisCache

logger = logging.getLogger(__name__)


@dataclass
class WatcherStats:
    """Statistics for the watcher."""

    started_at: datetime = field(default_factory=datetime.now)
    cycles: int = 0
    matches_found: int = 0
    pipelines_triggered: int = 0
    errors: int = 0
    last_cycle: Optional[datetime] = None


class Watcher:
    """
    Main watcher class that evaluates rules and triggers pipelines.

    Runs a polling loop that:
    1. Evaluates all registered rules against the CVM
    2. For each match, triggers the appropriate pipeline
    3. Tracks what's been processed to avoid duplicates
    """

    def __init__(
        self,
        config: ChatConfig,
        cvm: ConversationModel,
        rules: list[Rule],
        poll_interval: int = 60,
        dry_run: bool = False,
        stability_seconds: int = 120,
    ):
        """
        Args:
            config: ChatConfig for pipeline execution
            cvm: ConversationModel to monitor
            rules: List of rules to evaluate
            poll_interval: Seconds between polling cycles
            dry_run: If True, don't actually trigger pipelines
            stability_seconds: How long message count must be stable before triggering
        """
        self.config = config
        self.cvm = cvm
        self.rules = rules
        self.poll_interval = poll_interval
        self.dry_run = dry_run
        self.stability_seconds = stability_seconds

        self.stats = WatcherStats()
        self._running = False
        self._processed: set[str] = set()  # Track processed conversation_ids
        self._client: Optional[DreamerClient] = None
        self._stability_tracker: Optional[StabilityTracker] = None
        self._redis_cache: Optional[RedisCache] = None

        # Callbacks
        self.on_match: Optional[Callable[[RuleMatch], None]] = None
        self.on_pipeline_started: Optional[Callable[[str, RuleMatch], None]] = None
        self.on_error: Optional[Callable[[Exception, RuleMatch], None]] = None

    async def _get_client(self) -> DreamerClient:
        """Get or create the DreamerClient."""
        if self._client is None:
            self._client = DreamerClient.direct(self.config)
            await self._client.connect()
        return self._client

    async def _get_stability_tracker(self) -> StabilityTracker:
        """Get or create the StabilityTracker."""
        if self._stability_tracker is None:
            redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=getattr(self.config, 'redis_password', None),
            )
            self._stability_tracker = StabilityTracker(
                redis_client,
                stability_seconds=self.stability_seconds,
            )
        return self._stability_tracker

    def _get_redis_cache(self) -> RedisCache:
        """Get or create RedisCache instance."""
        if self._redis_cache is None:
            self._redis_cache = RedisCache(self.config)
        return self._redis_cache

    def is_api_idle(self, idle_threshold: int = 120) -> bool:
        """Check if the API has been idle for the threshold duration.

        Args:
            idle_threshold: Seconds of inactivity to consider idle (default: 120)

        Returns:
            True if API is idle (no recent activity), False if active
        """
        import time
        cache = self._get_redis_cache()
        last_activity = cache.get_api_last_activity()

        if last_activity is None:
            return True

        elapsed = time.time() - last_activity
        return elapsed >= idle_threshold

    def _make_processed_key(self, match: RuleMatch) -> str:
        """Create a unique key for tracking processed matches."""
        return f"{match.conversation_id}:{match.scenario}"

    async def evaluate_rules(self) -> list[RuleMatch]:
        """
        Evaluate all rules and return matches.

        Filters out already-processed conversations and deduplicates
        matches from multiple rules within the same cycle.
        """
        # Use dict to deduplicate matches by conversation_id:scenario
        matches_by_key: dict[str, RuleMatch] = {}

        for rule in self.rules:
            try:
                logger.debug(f"Evaluating rule: {rule.name}")
                matches = rule.evaluate(self.cvm)

                # Filter out already processed and deduplicate within cycle
                for match in matches:
                    key = self._make_processed_key(match)
                    if key not in self._processed and key not in matches_by_key:
                        matches_by_key[key] = match

                if matches:
                    new_count = sum(1 for m in matches if self._make_processed_key(m) in matches_by_key)
                    if new_count > 0:
                        logger.info(f"Rule '{rule.name}' found {new_count} new matches")

            except Exception as e:
                logger.error(f"Error evaluating rule '{rule.name}': {e}")
                self.stats.errors += 1

        return list(matches_by_key.values())

    async def trigger_pipeline(self, match: RuleMatch) -> Optional[str]:
        """
        Trigger a pipeline for the given match.

        Args:
            match: RuleMatch describing what to trigger

        Returns:
            Pipeline ID if started, None if dry_run, api_busy, or error
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would trigger {match.scenario} for {match.conversation_id}")
            return None

        # Check if API is idle before triggering
        if not self.is_api_idle():
            logger.debug(f"API busy, skipping trigger for {match.conversation_id}")
            return None

        try:
            client = await self._get_client()

            # TODO: Add proper duplicate detection by checking running pipelines
            # PipelineStatus would need conversation_id exposed for this
            # For now, rely on in-memory _processed set to prevent duplicates
            # within a single watcher session

            # Use model from match or fall back to config default
            model_name = match.model or self.config.default_model or "inclusionAI/Ling-1T"

            result = await client.start(
                scenario_name=match.scenario,
                conversation_id=match.conversation_id,
                model_name=model_name,
                query_text=match.guidance,
                persona_id=match.persona_id,
                user_id=match.persona_id,  # user_id should match persona_id
            )

            if not result.success:
                raise RuntimeError(result.error)

            logger.info(f"Triggered pipeline {result.pipeline_id} for {match.conversation_id}")
            self.stats.pipelines_triggered += 1

            if self.on_pipeline_started:
                self.on_pipeline_started(result.pipeline_id, match)

            return result.pipeline_id

        except Exception as e:
            logger.error(f"Error triggering pipeline for {match.conversation_id}: {e}")
            self.stats.errors += 1

            if self.on_error:
                self.on_error(e, match)

            return None

    async def run_cycle(self) -> int:
        """
        Run a single evaluation cycle.

        Returns:
            Number of pipelines triggered
        """
        self.stats.cycles += 1
        self.stats.last_cycle = datetime.now()
        triggered = 0

        # Reload the tantivy index to pick up new documents
        self.cvm.index.index.reload()

        # Evaluate all rules
        matches = await self.evaluate_rules()
        self.stats.matches_found += len(matches)

        # Process matches
        tracker = await self._get_stability_tracker()

        for match in matches:
            if self.on_match:
                self.on_match(match)

            # Check stability before triggering (only for matches with message_count)
            if match.message_count > 0:
                is_stable, snapshot = await tracker.update_and_check(
                    match.conversation_id,
                    match.message_count,
                    match.token_count,
                )
                if not is_stable:
                    logger.debug(
                        f"Skipping {match.conversation_id}: not stable yet "
                        f"(msgs={match.message_count}, tokens={match.token_count}, needs {self.stability_seconds}s)"
                    )
                    continue

            pipeline_id = await self.trigger_pipeline(match)

            if pipeline_id or self.dry_run:
                # Mark as processed and cleanup stability tracking
                self._processed.add(self._make_processed_key(match))
                if match.message_count > 0:
                    await tracker.mark_processed(match.conversation_id)
                triggered += 1

        return triggered

    async def run(self) -> None:
        """
        Run the watcher loop indefinitely.

        Polls at the configured interval, evaluating rules
        and triggering pipelines for matches.
        """
        self._running = True
        logger.info(f"Watcher started with {len(self.rules)} rules, polling every {self.poll_interval}s")

        while self._running:
            try:
                triggered = await self.run_cycle()

                if triggered > 0:
                    logger.info(f"Cycle complete: triggered {triggered} pipelines")
                else:
                    logger.debug("Cycle complete: no new matches")

            except Exception as e:
                logger.error(f"Error in watcher cycle: {e}")
                self.stats.errors += 1

            # Wait for next cycle
            await asyncio.sleep(self.poll_interval)

        logger.info("Watcher stopped")

    def stop(self) -> None:
        """Signal the watcher to stop."""
        self._running = False

    async def close(self) -> None:
        """Clean up resources."""
        self.stop()
        if self._client:
            await self._client.close()
            self._client = None
