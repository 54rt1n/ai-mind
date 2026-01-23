# aim/app/mud/worker/turns/strategy/helpers.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Helper functions for MUD worker turn processing."""

import logging
from typing import TYPE_CHECKING, Optional
from pathlib import Path
from aim.tool.loader import ToolLoader
from aim.tool.formatting import ToolUser
from aim.utils.xml import XmlFormatter
from aim.tool.dto import Tool

if TYPE_CHECKING:
    from aim_mud_types.models.plan import AgentPlan

logger = logging.getLogger(__name__)


class ToolHelper:
    """Helper class for loading and formatting tools."""

    def __init__(self, tool_user: ToolUser):
        """Initialize the tool helper with a list of tools."""
        self._tool_user = tool_user
        self._base_tools = list(tool_user.tools)  # Store original tools
        self._plan_tools: list[Tool] = []  # Extra tools for plan execution
        self._aura_tools: list[Tool] = []
        self._active_auras: list[str] = []
        # Mapping from aura name to source_id for tool execution
        self._aura_source_ids: dict[str, str] = {}
        # Mapping from tool name to aura name (to look up source_id)
        self._tool_to_aura: dict[str, str] = {}

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

    def update_aura_tools(
        self,
        auras: list[str],
        tools_path: str = "config/tools",
        aura_source_ids: Optional[dict[str, str]] = None,
    ) -> None:
        """Update tools based on active room auras.

        Args:
            auras: List of aura names active in the room.
            tools_path: Path to tools directory.
            aura_source_ids: Optional mapping from aura name to source object ID.
        """
        normalized = sorted(
            {str(a).strip().lower() for a in (auras or []) if str(a).strip()}
        )
        if normalized == self._active_auras:
            return
        self._active_auras = normalized
        self._aura_tools = []
        self._aura_source_ids = aura_source_ids or {}
        self._tool_to_aura = {}

        if not normalized:
            self._refresh_tool_user()
            return

        try:
            loader = ToolLoader(tools_path)
            tools_by_name: dict[str, Tool] = {}
            for aura in normalized:
                tool_file = Path(tools_path) / "auras" / f"{aura}.yaml"
                if not tool_file.exists():
                    logger.warning("Aura tools file not found: %s", tool_file)
                    continue
                aura_tools = loader.load_tool_file(str(tool_file)) or []
                for tool in aura_tools:
                    name = getattr(tool.function, "name", None)
                    if name and name not in tools_by_name:
                        tools_by_name[name] = tool
                        # Track which aura provides this tool
                        self._tool_to_aura[name] = aura
            self._aura_tools = list(tools_by_name.values())
        except Exception as e:
            logger.error(f"Failed to load aura tools: {e}")

        self._refresh_tool_user()

    def remove_plan_tools(self) -> None:
        """Remove plan execution tools.

        Should be called when a plan completes or is cleared.
        """
        if not self._plan_tools:
            return

        self._plan_tools = []
        # Restore base tools only
        self._refresh_tool_user()
        logger.info("Removed plan tools")

    def has_plan_tools(self) -> bool:
        """Check if plan tools are currently loaded.

        Returns:
            True if plan tools are loaded, False otherwise.
        """
        return bool(self._plan_tools)

    def _refresh_tool_user(self) -> None:
        """Refresh ToolUser based on base + plan + aura tools."""
        all_tools = self._base_tools + self._plan_tools + self._aura_tools
        self._tool_user = ToolUser(all_tools)

    def filter_to_tool(self, tool_name: str) -> bool:
        """Filter ToolUser to only include the specified tool.

        Args:
            tool_name: Name of the tool to keep.

        Returns:
            True if the tool was found and filtering applied, False otherwise.
        """
        tool_name_lower = tool_name.lower()
        all_tools = self._base_tools + self._plan_tools + self._aura_tools
        matching = [
            t for t in all_tools
            if getattr(t.function, "name", "").lower() == tool_name_lower
        ]
        if matching:
            self._tool_user = ToolUser(matching)
            logger.info("Filtered tools to: %s", tool_name)
            return True
        else:
            logger.warning("Tool '%s' not found for filtering", tool_name)
            return False

    def is_aura_tool(self, tool_name: str) -> bool:
        """Check if tool_name is an available aura tool.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if the tool is an aura tool, False otherwise.
        """
        for tool in self._aura_tools:
            if getattr(tool.function, "name", None) == tool_name:
                return True
        return False

    def get_aura_source_id(self, tool_name: str) -> Optional[str]:
        """Get the source object ID for an aura tool.

        Args:
            tool_name: Name of the aura tool.

        Returns:
            Source object ID (dbref) or None if not found.
        """
        aura_name = self._tool_to_aura.get(tool_name)
        if aura_name:
            return self._aura_source_ids.get(aura_name)
        return None

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
