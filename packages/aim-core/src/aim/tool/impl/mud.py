# aim/tool/impl/mud.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

"""MUD world interaction tools for AI agents.

These tools generate action data that will be emitted to the mud:actions
Redis stream for execution by Evennia. The tools themselves do not execute
commands directly - they return structured data representing the intended action.
"""

from typing import Any, Dict
from .base import ToolImplementation


class MudTool(ToolImplementation):
    """Tool implementation for MUD world interactions.

    These tools generate action dictionaries that are emitted to the
    mud:actions Redis stream for execution by Evennia. Each tool returns
    a result containing the tool name, arguments, and a human-readable
    message for agent feedback.
    """

    # =========================================================================
    # Player-Level Tools
    # =========================================================================

    def say(self, message: str, **kwargs: Any) -> Dict[str, Any]:
        """Speak audibly to everyone in the current location.

        Args:
            message: What to say
            **kwargs: Additional parameters (ignored)

        Returns:
            Dictionary with action data for execution
        """
        return {
            "status": "success",
            "tool": "say",
            "args": {"message": message},
            "message": f'You say: "{message}"'
        }

    def emote(self, action: str, **kwargs: Any) -> Dict[str, Any]:
        """Perform an action or express emotion visibly.

        Args:
            action: The action to perform (e.g., "smiles warmly")
            **kwargs: Additional parameters (ignored)

        Returns:
            Dictionary with action data for execution
        """
        return {
            "status": "success",
            "tool": "emote",
            "args": {"action": action},
            "message": f"You {action}"
        }

    def whisper(self, target: str, message: str, **kwargs: Any) -> Dict[str, Any]:
        """Send a private message to a specific person.

        Args:
            target: The name of the person to whisper to
            message: The private message to send
            **kwargs: Additional parameters (ignored)

        Returns:
            Dictionary with action data for execution
        """
        return {
            "status": "success",
            "tool": "whisper",
            "args": {"target": target, "message": message},
            "message": f'You whisper to {target}: "{message}"'
        }

    def pose(self, action: str, **kwargs: Any) -> Dict[str, Any]:
        """Perform a pose in the current room.

        Args:
            action: The pose to perform
            **kwargs: Additional parameters (ignored)

        Returns:
            Dictionary with action data for execution
        """
        return {
            "status": "success",
            "tool": "pose",
            "args": {"action": action},
            "message": f"You pose: {action}"
        }

    def move(self, location: str, **kwargs: Any) -> Dict[str, Any]:
        """Move to an adjacent room through an exit.

        Args:
            location: The direction or exit name to move through
            **kwargs: Additional parameters (ignored)

        Returns:
            Dictionary with action data for execution
        """
        return {
            "status": "success",
            "tool": "move",
            "args": {"location": location},
            "message": f"You move {location}"
        }

    def get(self, object: str, **kwargs: Any) -> Dict[str, Any]:
        """Pick up an object from the room or container.

        Args:
            object: The name of the object to pick up
            **kwargs: Additional parameters (ignored)

        Returns:
            Dictionary with action data for execution
        """
        return {
            "status": "success",
            "tool": "get",
            "args": {"object": object},
            "message": f"You pick up {object}"
        }

    def drop(self, object: str, **kwargs: Any) -> Dict[str, Any]:
        """Drop an object from inventory into the room.

        Args:
            object: The name of the object to drop
            **kwargs: Additional parameters (ignored)

        Returns:
            Dictionary with action data for execution
        """
        return {
            "status": "success",
            "tool": "drop",
            "args": {"object": object},
            "message": f"You drop {object}"
        }

    def give(self, object: str, target: str, **kwargs: Any) -> Dict[str, Any]:
        """Give an object from inventory to someone.

        Args:
            object: The name of the object to give
            target: The name of the person to give it to
            **kwargs: Additional parameters (ignored)

        Returns:
            Dictionary with action data for execution
        """
        return {
            "status": "success",
            "tool": "give",
            "args": {"object": object, "target": target},
            "message": f"You give {object} to {target}"
        }

    def use(self, object: str, **kwargs: Any) -> Dict[str, Any]:
        """Use or interact with an object.

        Args:
            object: The name of the object to use
            **kwargs: Additional parameters (ignored)

        Returns:
            Dictionary with action data for execution
        """
        return {
            "status": "success",
            "tool": "use",
            "args": {"object": object},
            "message": f"You use {object}"
        }

    def home(self, **kwargs: Any) -> Dict[str, Any]:
        """Return to your home location."""
        return {
            "status": "success",
            "tool": "home",
            "args": {},
            "message": "You return home"
        }

    def setdesc(self, description: str, **kwargs: Any) -> Dict[str, Any]:
        """Set your description.

        Args:
            description: The description text to set
            **kwargs: Additional parameters (ignored)

        Returns:
            Dictionary with action data for execution
        """
        return {
            "status": "success",
            "tool": "setdesc",
            "args": {"description": description},
            "message": "You update your description"
        }

    # =========================================================================
    # Builder-Level Tools (Andi only)
    # =========================================================================

    def dig(self, room: str, exits: str, **kwargs: Any) -> Dict[str, Any]:
        """Create a new room connected via exits.

        Args:
            room: The name of the new room to create
            exits: Exit specification (e.g., "north;n, south;s")
            **kwargs: Additional parameters (ignored)

        Returns:
            Dictionary with action data for execution
        """
        return {
            "status": "success",
            "tool": "dig",
            "args": {"room": room, "exits": exits},
            "message": "You create a new room."
        }

    def create(self, object: str, **kwargs: Any) -> Dict[str, Any]:
        """Create a new object in the current location.

        Args:
            object: The name of the object to create
            **kwargs: Additional parameters (ignored)

        Returns:
            Dictionary with action data for execution
        """
        return {
            "status": "success",
            "tool": "create",
            "args": {"object": object},
            "message": "You create something."
        }

    def desc(self, target: str, description: str, **kwargs: Any) -> Dict[str, Any]:
        """Set the description of a room, object, or character via @desc.

        Args:
            target: What to describe ("here" for current room)
            description: The new description text
            **kwargs: Additional parameters (ignored)

        Returns:
            Dictionary with action data for execution
        """
        return {
            "status": "success",
            "tool": "desc",
            "args": {"target": target, "description": description},
            "message": "You update a description"
        }

    def teleport(self, destination: str, target: str = "me", **kwargs: Any) -> Dict[str, Any]:
        """Instantly transport to another location.

        Args:
            destination: The name or ID of the room to teleport to
            target: Target to teleport (defaults to "me")
            **kwargs: Additional parameters (ignored)

        Returns:
            Dictionary with action data for execution
        """
        return {
            "status": "success",
            "tool": "teleport",
            "args": {"destination": destination, "target": target},
            "message": "You teleport."
        }

    # =========================================================================
    # Execute Dispatcher
    # =========================================================================

    def execute(self, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a MUD tool function.

        Args:
            function_name: Name of the tool (say, emote, move, etc.)
            parameters: Tool parameters from LLM

        Returns:
            Dictionary with action data for execution

        Raises:
            ValueError: If function is unknown or required parameters are missing
        """
        # Player-level tools
        if function_name == "say":
            if "message" not in parameters:
                raise ValueError("Message parameter is required")
            return self.say(message=parameters["message"])

        elif function_name == "emote":
            if "action" not in parameters:
                raise ValueError("Action parameter is required")
            return self.emote(action=parameters["action"])

        elif function_name == "whisper":
            if "target" not in parameters:
                raise ValueError("Target parameter is required")
            if "message" not in parameters:
                raise ValueError("Message parameter is required")
            return self.whisper(
                target=parameters["target"],
                message=parameters["message"]
            )

        elif function_name == "pose":
            if "action" not in parameters:
                raise ValueError("Action parameter is required")
            return self.pose(action=parameters["action"])

        elif function_name == "move":
            location = parameters.get("location") or parameters.get("direction")
            if not location:
                raise ValueError("Location parameter is required")
            return self.move(location=location)

        elif function_name == "get":
            if "object" not in parameters:
                raise ValueError("Object parameter is required")
            return self.get(object=parameters["object"])

        elif function_name == "drop":
            if "object" not in parameters:
                raise ValueError("Object parameter is required")
            return self.drop(object=parameters["object"])

        elif function_name == "give":
            if "object" not in parameters:
                raise ValueError("Object parameter is required")
            if "target" not in parameters:
                raise ValueError("Target parameter is required")
            return self.give(
                object=parameters["object"],
                target=parameters["target"]
            )

        elif function_name == "use":
            if "object" not in parameters:
                raise ValueError("Object parameter is required")
            return self.use(object=parameters["object"])

        elif function_name == "home":
            return self.home()

        elif function_name == "setdesc":
            if "description" not in parameters:
                raise ValueError("Description parameter is required")
            return self.setdesc(description=parameters["description"])

        # Builder-level tools
        elif function_name == "dig":
            if "room" not in parameters:
                raise ValueError("Room parameter is required")
            if "exits" not in parameters:
                raise ValueError("Exits parameter is required")
            return self.dig(room=parameters["room"], exits=parameters["exits"])

        elif function_name == "create":
            if "object" not in parameters:
                raise ValueError("Object parameter is required")
            return self.create(object=parameters["object"])

        elif function_name == "desc":
            if "target" not in parameters:
                raise ValueError("Target parameter is required")
            if "description" not in parameters:
                raise ValueError("Description parameter is required")
            return self.desc(
                target=parameters["target"],
                description=parameters["description"]
            )

        elif function_name == "teleport":
            if "destination" not in parameters:
                raise ValueError("Destination parameter is required")
            return self.teleport(
                destination=parameters["destination"],
                target=parameters.get("target", "me")
            )

        else:
            return {
                "status": "error",
                "error": f"Unknown MUD tool: {function_name}"
            }
