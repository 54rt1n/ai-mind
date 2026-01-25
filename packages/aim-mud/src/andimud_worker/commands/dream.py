# andimud_worker/commands/dream.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Dream command - run dreamer pipeline or planner pipeline."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDTurnRequest, TurnRequestStatus
from .base import Command
from .result import CommandResult

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class DreamCommand(Command):
    """@dream console command - run a dreamer pipeline or planner pipeline.

    Handles two pipeline types:
    1. Planner: metadata["pipeline"] == "planner" - creates a plan
    2. Dream: metadata["scenario"] - runs a dreamer scenario
    """

    @property
    def name(self) -> str:
        return "dream"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Execute dream or planner pipeline.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: Contains turn_id, metadata

        Returns:
            CommandResult with complete=True, status=TurnRequestStatus.DONE or FAIL
        """
        turn_id = kwargs.get("turn_id", "unknown")

        # Use Pydantic to parse metadata JSON
        turn_request = MUDTurnRequest.model_validate(kwargs)
        metadata = turn_request.metadata or {}

        # Check for planner pipeline
        if metadata.get("pipeline") == "planner":
            return await self._execute_planner(worker, metadata, turn_id)

        # Otherwise, execute dream pipeline
        return await self._execute_dream(worker, metadata, turn_id)

    async def _execute_planner(
        self,
        worker: "MUDAgentWorker",
        metadata: dict,
        turn_id: str,
    ) -> CommandResult:
        """Execute planner pipeline.

        Args:
            worker: MUDAgentWorker instance
            metadata: Turn request metadata with objective
            turn_id: Turn identifier for logging

        Returns:
            CommandResult with pipeline status
        """
        objective = metadata.get("objective", "")
        if not objective:
            logger.error("Planner pipeline missing objective")
            return CommandResult(
                complete=True,
                status=TurnRequestStatus.FAIL,
                message="Planner pipeline missing objective",
            )

        logger.info(f"[{turn_id}] Starting planner pipeline: {objective[:50]}...")

        # Import and create PlannerEngine
        from aim.planner import PlannerEngine
        from dream_agent.client import DreamerClient

        # Create DreamerClient for the planner to use
        async with DreamerClient.direct(worker.chat_config) as dreamer_client:
            engine = PlannerEngine(
                config=worker.chat_config,
                dreamer_client=dreamer_client,
                redis_client=worker.redis,
            )

            plan = await engine.create_plan(worker.config.agent_id, objective)

        if plan:
            logger.info(f"[{turn_id}] Plan created: {plan.plan_id}")
            # Refresh CVM to include new deliberation documents
            worker.cvm.refresh()
            return CommandResult(
                complete=True,
                status=TurnRequestStatus.DONE,
                message=f"Plan created: {plan.summary}",
            )
        else:
            logger.error(f"[{turn_id}] Plan creation failed")
            return CommandResult(
                complete=True,
                status=TurnRequestStatus.FAIL,
                message="Plan creation failed",
            )

    async def _execute_dream(
        self,
        worker: "MUDAgentWorker",
        metadata: dict,
        turn_id: str,
    ) -> CommandResult:
        """Execute dream pipeline.

        Args:
            worker: MUDAgentWorker instance
            metadata: Turn request metadata with scenario, query, guidance
            turn_id: Turn identifier for logging

        Returns:
            CommandResult with pipeline status
        """
        scenario = metadata.get("scenario", "")
        query = metadata.get("query")
        guidance = metadata.get("guidance")
        # Explicit conversation_id for analysis commands
        target_conversation_id = metadata.get("conversation_id")

        if not scenario:
            logger.error("Dream turn missing scenario in metadata")
            return CommandResult(
                complete=True,
                status=TurnRequestStatus.FAIL,
                message="Dream turn missing scenario in metadata",
            )

        logger.info(f"Processing dream turn: {scenario}")
        result = await worker.process_dream_turn(
            scenario=scenario,
            query=query,
            guidance=guidance,
            triggered_by="manual",
            target_conversation_id=target_conversation_id,
        )

        if result.success:
            logger.info(
                f"Dream completed: {result.pipeline_id} "
                f"in {result.duration_seconds:.1f}s"
            )
            # Refresh index and update conversation report
            worker.cvm.refresh()
            await worker._update_conversation_report()
            return CommandResult(
                complete=True,
                status=TurnRequestStatus.DONE,
                message=f"Dream completed: {scenario}",
            )
        else:
            logger.error(f"Dream failed: {result.error}")
            return CommandResult(
                complete=True,
                status=TurnRequestStatus.FAIL,
                message=result.error or "Dream failed",
            )
