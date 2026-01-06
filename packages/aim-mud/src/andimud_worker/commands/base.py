# andimud_worker/commands/base.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Base command interface for ANDIMUD worker commands.

Commands are executed in response to turn_request reasons (flush, dream, agent, etc.).
Each command receives the worker instance and unpacks relevant fields from turn_request.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .result import CommandResult

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


class Command(ABC):
    """Base interface for worker commands.

    Each command implements a specific turn_request reason handler.
    Commands receive the worker instance and unpack turn_request fields
    as needed from **kwargs.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The command name (matches turn_request.reason).

        Returns:
            Command name string (e.g., "flush", "dream", "agent")
        """
        pass

    @abstractmethod
    async def execute(self, worker: "MUDAgentWorker", **kwargs) -> CommandResult:
        """Execute the command.

        Args:
            worker: The MUDAgentWorker instance
            **kwargs: Fields from turn_request (turn_id, guidance, scenario, etc.)
                Commands unpack only the fields they need.

        Returns:
            CommandResult indicating completion status and event drain behavior

        Raises:
            Exception: Commands may raise exceptions which will be handled by
                the worker's error handling logic (retry, fail state, etc.)
        """
        pass
