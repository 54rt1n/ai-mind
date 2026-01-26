# andimud_worker/commands/focus.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Focus commands for code agent workers.

These commands handle FOCUS and CLEAR_FOCUS turn reasons, setting or clearing
the code focus on the worker's decision and response strategies.
"""

import json
import logging
from typing import TYPE_CHECKING

from aim_mud_types.client import RedisMUDClient

from .base import Command
from .result import CommandResult


if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class FocusCommand(Command):
    """Handle FOCUS turn - set code focus from metadata.

    Sets focus on decision and response strategies using metadata from
    the turn request. This affects what code context appears in the
    agent's consciousness during subsequent turns.

    Expected metadata fields:
        files: List of file specs, each a dict with:
            - path: File path to focus on
            - start: Optional start line (1-indexed, inclusive)
            - end: Optional end line (1-indexed, inclusive)
        height: Levels UP the call graph (who calls these symbols)
        depth: Levels DOWN the call graph (what these symbols call)
    """

    @property
    def name(self) -> str:
        return "focus"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Set focus on code strategies from turn request metadata.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: Contains turn_id, metadata, turn_request fields

        Returns:
            CommandResult with complete=True
        """
        turn_id = kwargs.get("turn_id", "unknown")
        metadata = kwargs.get("metadata")

        if not metadata:
            logger.warning("[%s] Focus command received without metadata", turn_id)
            return CommandResult(complete=True)

        # Extract focus parameters
        files = metadata.get("files", [])
        height = metadata.get("height", 1)
        depth = metadata.get("depth", 1)

        if not files:
            logger.warning("[%s] Focus command received with no files", turn_id)
            return CommandResult(complete=True)

        # Import FocusRequest here to avoid circular imports
        from aim_code.strategy.base import FocusRequest

        focus_request = FocusRequest(
            files=files,  # Already list[dict] from MUD command
            height=height,
            depth=depth,
        )

        # Set focus on decision strategy
        decision_strategy = worker._decision_strategy
        if decision_strategy is not None and hasattr(decision_strategy, "set_focus"):
            decision_strategy.set_focus(focus_request)
            file_paths = [f.get("path", f) if isinstance(f, dict) else f for f in files]
            logger.info("[%s] Focus set on decision strategy: %s", turn_id, file_paths)

        # Set focus on response strategy
        response_strategy = worker._response_strategy
        if response_strategy is not None and hasattr(response_strategy, "set_focus"):
            response_strategy.set_focus(focus_request)
            file_paths = [f.get("path", f) if isinstance(f, dict) else f for f in files]
            logger.info("[%s] Focus set on response strategy: %s", turn_id, file_paths)

        # Persist focus to agent profile
        focus_data = {
            "files": files,  # list[dict] with path, start, end
            "height": height,
            "depth": depth,
        }
        client = RedisMUDClient(worker.redis)
        await client.update_agent_profile_fields(
            worker.config.agent_id,
            focus=json.dumps(focus_data),
        )
        logger.info("[%s] Focus persisted to profile", turn_id)

        return CommandResult(complete=True)


class ClearFocusCommand(Command):
    """Handle CLEAR_FOCUS turn - clear code focus.

    Clears focus on decision and response strategies, returning to
    default semantic search behavior for code context.
    """

    @property
    def name(self) -> str:
        return "clear_focus"

    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Clear focus on code strategies.

        Args:
            worker: MUDAgentWorker instance
            **kwargs: Contains turn_id, metadata, turn_request fields

        Returns:
            CommandResult with complete=True
        """
        turn_id = kwargs.get("turn_id", "unknown")

        # Clear focus on decision strategy
        decision_strategy = worker._decision_strategy
        if decision_strategy is not None and hasattr(decision_strategy, "clear_focus"):
            decision_strategy.clear_focus()
            logger.info("[%s] Focus cleared on decision strategy", turn_id)

        # Clear focus on response strategy
        response_strategy = worker._response_strategy
        if response_strategy is not None and hasattr(response_strategy, "clear_focus"):
            response_strategy.clear_focus()
            logger.info("[%s] Focus cleared on response strategy", turn_id)

        # Clear focus from agent profile
        client = RedisMUDClient(worker.redis)
        await client.update_agent_profile_fields(
            worker.config.agent_id,
            focus="",
        )
        logger.info("[%s] Focus cleared from profile", turn_id)

        return CommandResult(complete=True)
