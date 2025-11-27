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

from aim.config import ChatConfig
from aim.conversation.model import ConversationModel
from aim.watcher.rules import Rule, RuleMatch
from aim.app.dream_agent.client import DreamerClient

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
    ):
        """
        Args:
            config: ChatConfig for pipeline execution
            cvm: ConversationModel to monitor
            rules: List of rules to evaluate
            poll_interval: Seconds between polling cycles
            dry_run: If True, don't actually trigger pipelines
        """
        self.config = config
        self.cvm = cvm
        self.rules = rules
        self.poll_interval = poll_interval
        self.dry_run = dry_run

        self.stats = WatcherStats()
        self._running = False
        self._processed: set[str] = set()  # Track processed conversation_ids
        self._client: Optional[DreamerClient] = None

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

    def _make_processed_key(self, match: RuleMatch) -> str:
        """Create a unique key for tracking processed matches."""
        return f"{match.conversation_id}:{match.scenario}"

    async def evaluate_rules(self) -> list[RuleMatch]:
        """
        Evaluate all rules and return matches.

        Filters out already-processed conversations.
        """
        all_matches = []

        for rule in self.rules:
            try:
                logger.debug(f"Evaluating rule: {rule.name}")
                matches = rule.evaluate(self.cvm)

                # Filter out already processed
                new_matches = [
                    m for m in matches
                    if self._make_processed_key(m) not in self._processed
                ]

                if new_matches:
                    logger.info(f"Rule '{rule.name}' found {len(new_matches)} new matches")
                    all_matches.extend(new_matches)

            except Exception as e:
                logger.error(f"Error evaluating rule '{rule.name}': {e}")
                self.stats.errors += 1

        return all_matches

    async def trigger_pipeline(self, match: RuleMatch) -> Optional[str]:
        """
        Trigger a pipeline for the given match.

        Args:
            match: RuleMatch describing what to trigger

        Returns:
            Pipeline ID if started, None if dry_run or error
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would trigger {match.scenario} for {match.conversation_id}")
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
        for match in matches:
            if self.on_match:
                self.on_match(match)

            pipeline_id = await self.trigger_pipeline(match)

            if pipeline_id or self.dry_run:
                # Mark as processed
                self._processed.add(self._make_processed_key(match))
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
