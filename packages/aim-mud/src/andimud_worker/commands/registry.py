# andimud_worker/commands/registry.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Command registry for ANDIMUD worker commands."""

import logging
from typing import TYPE_CHECKING, Dict, Optional

from .base import Command
from .result import CommandResult

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class CommandRegistry:
    """Registry of commands keyed by name.

    Commands are looked up by turn_request.reason and executed.
    """

    def __init__(self):
        """Initialize empty command registry."""
        self._commands: Dict[str, Command] = {}

    @classmethod
    def register(cls, *commands: Command) -> "CommandRegistry":
        """Create a registry and register all provided commands.

        Factory method for creating a registry with commands.

        Args:
            *commands: Command instances to register

        Returns:
            CommandRegistry with all commands registered

        Example:
            registry = CommandRegistry.register(
                FlushCommand(),
                DreamCommand(),
                AgentCommand(),
            )
        """
        registry = cls()
        for command in commands:
            registry._commands[command.name] = command
            logger.debug(f"Registered command: {command.name}")
        return registry

    def get_command(self, name: str) -> Optional[Command]:
        """Get a command by name.

        Args:
            name: Command name to look up

        Returns:
            Command instance or None if not found
        """
        return self._commands.get(name)

    async def execute(
        self,
        worker: "MUDAgentWorker",
        reason: str,
        **kwargs
    ) -> CommandResult:
        """Execute a command by name.

        Args:
            reason: Command name (from turn_request.reason)
            worker: The MUDAgentWorker instance
            **kwargs: Arguments to pass to command (from turn_request)

        Returns:
            CommandResult from the executed command

        Raises:
            ValueError: If command not found
            Exception: Any exception raised by the command
        """
        command = self.get_command(reason)
        if command is None:
            raise ValueError(f"Unknown command: {reason}")

        logger.debug(f"Executing command: {reason}")
        return await command.execute(worker, reason=reason, **kwargs)
