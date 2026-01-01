# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""MUD action model for agent commands."""

from typing import Any

from pydantic import BaseModel, Field


class MUDAction(BaseModel):
    """An action the agent wants to take in the MUD.

    Actions are emitted by the agent worker to the mud:actions stream
    and consumed by Evennia to execute commands in the world.

    The to_command() method converts the structured action into an
    Evennia command string for execution.

    Attributes:
        tool: Tool name (e.g., "say", "emote", "move").
        args: Tool arguments as a dictionary.
        priority: Execution priority (lower = first). Default 5.
    """

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    priority: int = 5

    def to_command(self) -> str:
        """Convert action to Evennia command string.

        Handles all player-level and builder-level commands defined
        in the ANDIMUD specification.

        Returns:
            Command string ready for execution in Evennia.
        """
        # Player-level commands
        if self.tool == "say":
            return f"say {self.args.get('message', '')}"

        elif self.tool == "emote":
            return f"emote {self.args.get('action', '')}"

        elif self.tool == "whisper":
            target = self.args.get("target", "")
            message = self.args.get("message", "")
            return f"whisper {target} = {message}"

        elif self.tool == "look":
            target = self.args.get("target", "")
            return f"look {target}".strip()

        elif self.tool == "move":
            return self.args.get("direction", "")

        elif self.tool == "get":
            return f"get {self.args.get('object', '')}"

        elif self.tool == "drop":
            return f"drop {self.args.get('object', '')}"

        elif self.tool == "give":
            obj = self.args.get("object", "")
            target = self.args.get("target", "")
            return f"give {obj} = {target}"

        elif self.tool == "use":
            return f"use {self.args.get('object', '')}"

        # Builder-level commands (Andi only)
        elif self.tool == "dig":
            room = self.args.get("room", "")
            exits = self.args.get("exits", "")
            return f"@dig {room} = {exits}"

        elif self.tool == "create":
            return f"@create {self.args.get('object', '')}"

        elif self.tool == "describe":
            target = self.args.get("target", "")
            desc = self.args.get("description", "")
            return f"@describe {target} = {desc}"

        elif self.tool == "teleport":
            return f"@teleport {self.args.get('destination', '')}"

        elif self.tool == "destroy":
            return f"@destroy {self.args.get('object', '')}"

        elif self.tool == "link":
            exit_name = self.args.get("exit", "")
            destination = self.args.get("destination", "")
            return f"@link {exit_name} = {destination}"

        elif self.tool == "lock":
            obj = self.args.get("object", "")
            lockstring = self.args.get("lockstring", "")
            return f"@lock {obj} = {lockstring}"

        elif self.tool == "set":
            obj = self.args.get("object", "")
            attr = self.args.get("attribute", "")
            value = self.args.get("value", "")
            return f"@set {obj}/{attr} = {value}"

        else:
            # Unknown tool - use raw command if provided, else format as-is
            return self.args.get("command", f"{self.tool} {self.args}")

    def to_redis_dict(self, agent_id: str) -> dict[str, Any]:
        """Convert to dictionary for Redis stream publishing.

        Args:
            agent_id: ID of the agent emitting this action.

        Returns:
            Dictionary ready for JSON serialization to Redis.
        """
        return {
            "agent_id": agent_id,
            "command": self.to_command(),
            "tool": self.tool,
            "args": self.args,
            "priority": self.priority,
        }
