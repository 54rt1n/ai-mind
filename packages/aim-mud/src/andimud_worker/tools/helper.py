# aim/app/mud/worker/turns/strategy/helpers.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Helper functions for MUD worker turn processing."""

import logging
from typing import TYPE_CHECKING
from pathlib import Path
from aim.tool.loader import ToolLoader
from aim.tool.formatting import ToolUser
from aim.utils.xml import XmlFormatter
from aim.tool.dto import Tool

if TYPE_CHECKING:
    from aim_mud_types.plan import AgentPlan

logger = logging.getLogger(__name__)


class ToolHelper:
    """Helper class for loading and formatting tools."""

    def __init__(self, tool_user: ToolUser):
        """Initialize the tool helper with a list of tools."""
        self._tool_user = tool_user
        self._base_tools = list(tool_user.tools)  # Store original tools
        self._plan_tools: list[Tool] = []  # Extra tools for plan execution

    def decorate_xml(self, xml: XmlFormatter) -> XmlFormatter:
        """Decorate the XML with the agent action tools.

        Args:
            xml: The XML to decorate

        Returns:
            The decorated XML
        """
        return self._tool_user.xml_decorator(xml)

    def add_plan_tools(self, plan: "AgentPlan", tools_path: str = "config/tools") -> None:
        """Add plan execution tools when a plan is active.

        Loads the plan.yaml tools and adds them to the tool user.
        Should be called when an active plan is detected.

        Args:
            plan: The active plan (used for context, not currently inspected).
            tools_path: Path to tools directory.
        """
        if self._plan_tools:
            # Already loaded
            return

        try:
            loader = ToolLoader(tools_path)
            plan_tool_file = Path(tools_path) / "plan.yaml"
            if plan_tool_file.exists():
                self._plan_tools = loader.load_tool_file(str(plan_tool_file))
                # Update tool user with combined tools
                all_tools = self._base_tools + self._plan_tools
                self._tool_user = ToolUser(all_tools)
                logger.info(f"Added {len(self._plan_tools)} plan tools")
            else:
                logger.warning(f"Plan tools file not found: {plan_tool_file}")
        except Exception as e:
            logger.error(f"Failed to load plan tools: {e}")

    def remove_plan_tools(self) -> None:
        """Remove plan execution tools.

        Should be called when a plan completes or is cleared.
        """
        if not self._plan_tools:
            return

        self._plan_tools = []
        # Restore base tools only
        self._tool_user = ToolUser(self._base_tools)
        logger.info("Removed plan tools")

    def has_plan_tools(self) -> bool:
        """Check if plan tools are currently loaded.

        Returns:
            True if plan tools are loaded, False otherwise.
        """
        return bool(self._plan_tools)

    @staticmethod
    def load_tools(tool_config_file: str, tools_path: str) -> ToolUser:
        """Load the tools.

        Args:
            tool_config_file: Path to tool configuration file.
            tools_path: Path to tools directory.

        Returns:
            ToolUser with loaded tools.
        """
        loader = ToolLoader(tools_path)
        _tools = loader.load_tool_file(tool_config_file)
        if not _tools:
            raise ValueError(f"No tools loaded from {tool_config_file}")
        tool_user = ToolUser(_tools)
        return tool_user

    @classmethod
    def from_file(cls, tool_config_file: str, tools_path: str) -> "ToolHelper":
        """Create a tool helper from a file.

        Args:
            tool_config_file: Path to the tool configuration file
            tools_path: Path to the tools directory

        Returns:
            ToolHelper instance with loaded tools.
        """
        tool_user = cls.load_tools(tool_config_file, tools_path)
        return cls(tool_user)