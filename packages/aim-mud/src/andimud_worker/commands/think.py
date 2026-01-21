# andimud_worker/commands/think.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Think command - @think turn with injected thought content."""

import json
import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDTurnRequest, RedisKeys, TurnRequestStatus
from .base import Command
from .result import CommandResult

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class ThinkCommand(Command):
    """@think console command - process turn with injected thought content.

    Reads thought from Redis key `agent:{id}:thought` and injects it
    into the response strategy for Phase 2 processing. Falls through
    to process_turn similar to ChooseCommand.

    The thought content is read from Redis as JSON:
    {
        "content": str,      # The thought text
        "source": str,       # "manual" | "dreamer" | "system"
        "timestamp": int,    # Unix timestamp when set
    }
    """

    @property
    def name(self) -> str:
        return "think"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Process @think turn with thought injection.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: Contains turn_id, metadata (with optional guidance), etc.

        Returns:
            CommandResult with complete=False (falls through to process_turn)
        """
        turn_id = kwargs.get("turn_id", "unknown")

        # Use Pydantic to parse metadata JSON
        turn_request = MUDTurnRequest.model_validate(kwargs)
        metadata = turn_request.metadata or {}
        guidance = metadata.get("guidance", "") if metadata else ""

        # Read thought from Redis
        thought_key = RedisKeys.agent_thought(worker.config.agent_id)
        thought_raw = await worker.redis.get(thought_key)

        thought_content = None
        thought_source = "unknown"
        if thought_raw:
            try:
                raw_str = thought_raw.decode("utf-8") if isinstance(thought_raw, bytes) else thought_raw
                thought_data = json.loads(raw_str)
                thought_content = thought_data.get("content", "")
                thought_source = thought_data.get("source", "unknown")
                logger.info(
                    "[%s] ThinkCommand: loaded thought from %s (%d chars)",
                    turn_id,
                    thought_source,
                    len(thought_content),
                )
            except json.JSONDecodeError as e:
                logger.warning("[%s] Failed to parse thought JSON: %s", turn_id, e)
        else:
            logger.info("[%s] ThinkCommand: no thought content set", turn_id)

        # Inject thought into response strategy for Phase 2
        if worker._response_strategy and thought_content:
            worker._response_strategy.thought_content = thought_content
            logger.debug("[%s] Injected thought_content into response strategy", turn_id)

        # Build message
        message = "Think turn ready"
        if thought_content:
            preview = thought_content[:50] + "..." if len(thought_content) > 50 else thought_content
            message = f"Think turn with thought: {preview}"

        logger.info(
            "[%s] Processing @think turn with %d events, thought=%s",
            turn_id,
            len(worker.pending_events),
            "yes" if thought_content else "no",
        )

        # Fall through to process_turn (like ChooseCommand)
        return CommandResult(
            complete=False,
            flush_drain=False,
            saved_event_id=None,
            status=TurnRequestStatus.DONE,
            message=message,
            plan_guidance=guidance if guidance else None,
        )
