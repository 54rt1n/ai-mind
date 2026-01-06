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

logger = logging.getLogger(__name__)


class ToolHelper:
    """Helper class for loading and formatting tools."""

    def __init__(self, tool_user: ToolUser):
        """Initialize the tool helper with a list of tools."""
        self._tool_user = tool_user

    def decorate_xml(self, xml: XmlFormatter) -> XmlFormatter:
        """Decorate the XML with the agent action tools.

        Args:
            xml: The XML to decorate

        Returns:
            The decorated XML
        """
        return self._tool_user.xml_decorator(xml)

    @staticmethod
    def load_tools(tool_config_file: str, tools_path: str) -> ToolUser:
        """Load the tools.

        Args:
            xml: The XML to decorate
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
        """
        tool_user = cls.load_tools(tool_config_file, tools_path)
        return cls(tool_user)