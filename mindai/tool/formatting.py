# mindai/tool/formatting.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from .dto import Tool
from ..utils.xml import XmlFormatter
import json
import re
from dataclasses import dataclass

# A tool represents an instance of a function that can be called by the model.
# The inputs are the tool classes with their function parameters.
# The output is a system message piece that instructs the model how to use the tool;
#   and a response validator and parser.
# Tool calls will be expected to output a JSON object.

"""
The ToolUser class is a wrapper class around a set of Tool instances designed to generate proper formatting and validation instructions
for language models based on Tool descriptions. Key aspects:

1. Tool Description Processing
   - Parse tool definitions into model-friendly instruction formats
   - Generate clear examples of valid tool usage
   - Create validation rules for model outputs
   - Format result in to a ToolResponse object

2. Model Instruction Components
   - Input format requirements and constraints
   - Expected output structure and format
   - Parameter validation rules
   - Error handling instructions
   - Example successful and failed interactions

3. Validation Rules Generation
   - Parameter type checking instructions
   - Required field validation guidance
   - Value range and constraint checking
   - Format validation for special types (dates, URLs, etc)

4. Response Format Templates
   - JSON structure templates
   - Extracting the result based on the tool definition
   - Success response formats

5. System Message Integration
   - Tool availability and calling instructions
   - N-shot example generation

"""


@dataclass
class ToolCallResult:
    """Result of a tool call validation and parsing."""

    is_valid: bool
    tool: Optional[Tool] = None
    error: Optional[str] = None
    tool_type: Optional[str] = None
    function_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None


class ToolUser:
    """Manages tool interactions, validation, and formatting."""

    def __init__(self, tools: List[Tool]):
        """Initialize with list of available tools.

        Args:
            tools: List of available tools
        """
        self.tools = tools

    def xml_decorator(self, xml: XmlFormatter) -> XmlFormatter:
        """Format tool instructions into system message.

        Args:
            xml: XML formatter to add instructions to

        Returns:
            Updated XML formatter
        """
        if not self.tools:
            return xml

        # Add tools section with header
        xml.add_element("Tools", content="Available tools for your use:")
        xml.add_element("Tools", "Format",
            content="To use a tool, output a JSON object in the following format:",
        )

        # Document each tool with example
        for tool in self.tools:
            # Create example arguments
            example_args = {}
            if tool.function.parameters.examples and len(tool.function.parameters.examples) > 0:
                # Use the first example if available
                example_args = tool.function.parameters.examples[0]
            else:
                # Fall back to generating examples
                for (
                    param_name,
                    param_info,
                ) in tool.function.parameters.properties.items():
                    if "enum" in param_info:
                        example_args[param_name] = param_info["enum"][0]
                    elif param_info.get("type") == "number":
                        example_args[param_name] = 42.0
                    elif param_info.get("type") == "integer":
                        example_args[param_name] = 42
                    elif param_info.get("type") == "boolean":
                        example_args[param_name] = True
                    else:
                        example_args[param_name] = f"Example {param_name}"

            example = {tool.function.name: example_args}

            # Add tool documentation
            xml.add_element(
                "Tools", "functions", tool.function.name, "Name", content=tool.function.name, nowrap=True,
            )
            xml.add_element(
                "Tools", "functions", tool.function.name, "Type", content=tool.type, nowrap=True,
            )
            xml.add_element(
                "Tools", "functions", tool.function.name, "Description", content=tool.function.description, nowrap=True,
            )
            xml.add_element(
                "Tools", "functions", tool.function.name, "Example", content=f"```json\n{json.dumps(example, indent=2)}\n```", nowrap=True,
            )

            # Add parameter details
            xml.add_element(
                "Tools", "functions", tool.function.name, "Parameters", content="Required parameters and their descriptions:", nowrap=True,
            )
            for param_name, param_info in tool.function.parameters.properties.items():
                required = "[Required]" if param_name in tool.function.parameters.required else "[Optional]"
                param_type = param_info.get("type", "any")
                if "enum" in param_info:
                    param_type += f", one of: {param_info['enum']}"
                description = param_info.get("description", "")
                xml.add_element(
                    "Tools", "functions", tool.function.name, "Parameters", "Parameter",
                    content=f"{param_name} ({param_type}): {description} {required}", nowrap=True,
                )

        return xml

    def process_response(self, response: str) -> ToolCallResult:
        """Process an LLM response into a validated tool call.

        Args:
            response: Raw LLM response

        Returns:
            ToolCallResult containing validation results and parsed data
        """
        # Extract tool call
        tool_call = self._extract_tool_call(response)
        if not tool_call:
            return ToolCallResult(
                is_valid=False,
                error="Could not extract valid JSON tool call from response",
            )

        # Basic structure validation - should be a dict with exactly one key
        if not isinstance(tool_call, dict) or len(tool_call) != 1:
            return ToolCallResult(
                is_valid=False,
                error="Response must be a JSON object with exactly one function call",
            )

        # Get function name and arguments
        function_name = next(iter(tool_call))
        arguments = tool_call[function_name]

        # Find matching tool
        tool = next((t for t in self.tools if t.function.name == function_name), None)
        if not tool:
            return ToolCallResult(
                is_valid=False,
                error=f"No matching tool found for function '{function_name}'",
            )

        # Validate arguments
        if not isinstance(arguments, dict):
            return ToolCallResult(is_valid=False, tool=tool, error="Arguments must be a dictionary")

        # Check required parameters
        for required in tool.function.parameters.required:
            if required not in arguments:
                return ToolCallResult(
                    is_valid=False,
                    tool=tool,
                    error=f"Missing required parameter: {required}",
                )

        # Validate parameter types and values
        for param_name, param_value in arguments.items():
            if param_name not in tool.function.parameters.properties:
                return ToolCallResult(is_valid=False, tool=tool, error=f"Unknown parameter: {param_name}")

            param_info = tool.function.parameters.properties[param_name]

            # Type validation
            expected_type = param_info.get("type")
            if expected_type == "string" and not isinstance(param_value, str):
                return ToolCallResult(
                    is_valid=False,
                    tool=tool,
                    error=f"Parameter {param_name} must be a string",
                )
            elif expected_type == "number" and not isinstance(param_value, (int, float)):
                return ToolCallResult(
                    is_valid=False,
                    tool=tool,
                    error=f"Parameter {param_name} must be a number",
                )
            elif expected_type == "integer" and not isinstance(param_value, int):
                return ToolCallResult(
                    is_valid=False,
                    tool=tool,
                    error=f"Parameter {param_name} must be an integer",
                )
            elif expected_type == "boolean" and not isinstance(param_value, bool):
                return ToolCallResult(
                    is_valid=False,
                    tool=tool,
                    error=f"Parameter {param_name} must be a boolean",
                )

            # Enum validation
            if "enum" in param_info and param_value not in param_info["enum"]:
                return ToolCallResult(
                    is_valid=False,
                    tool=tool,
                    error=f"Parameter {param_name} must be one of: {param_info['enum']}",
                )

        # All validation passed
        return ToolCallResult(
            is_valid=True,
            tool=tool,
            tool_type=tool.type,
            function_name=function_name,
            arguments=arguments,
        )

    def _extract_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract tool call JSON from response."""
        try:
            # Try to parse entire response as JSON first
            return json.loads(response)
        except json.JSONDecodeError:
            # If that fails, try to find JSON block
            json_pattern = r"\{[\s\S]*\}"
            match = re.search(json_pattern, response)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            return None


class ToolValidator:
    """Validates and parses tool calls from LLM responses."""

    @staticmethod
    def validate_tool_call(tool_call: Dict[str, Any], available_tools: List[Tool]) -> Optional[str]:
        """Validate a tool call against available tools.

        Args:
            tool_call: The tool call to validate
            available_tools: List of available tools

        Returns:
            Error message if validation fails, None if successful
        """
        # Validate basic structure
        if not isinstance(tool_call, dict):
            return "Tool call must be a dictionary"

        if "tool_calls" not in tool_call:
            return "Response must contain 'tool_calls' array"

        if not isinstance(tool_call["tool_calls"], list) or len(tool_call["tool_calls"]) == 0:
            return "tool_calls must be a non-empty array"

        # We'll validate the first tool call for now
        call = tool_call["tool_calls"][0]

        # Validate tool call structure
        if "type" not in call:
            return "Tool call must have 'type' field"

        if "function" not in call:
            return "Tool call must have 'function' field"

        if "name" not in call["function"]:
            return "Tool function must have 'name' field"

        if "arguments" not in call["function"]:
            return "Tool function must have 'arguments' field"

        # Find matching tool
        tool = next(
            (t for t in available_tools if t.type == call["type"] and t.function.name == call["function"]["name"]),
            None,
        )

        if not tool:
            return f"No matching tool found for type '{call['type']}' and name '{call['function']['name']}'"

        # Validate arguments against tool parameters
        args = call["function"]["arguments"]
        if not isinstance(args, dict):
            return "Tool arguments must be a dictionary"

        # Check required parameters
        for required in tool.function.parameters.required:
            if required not in args:
                return f"Missing required parameter: {required}"

        # Validate parameter types and values
        for param_name, param_value in args.items():
            if param_name not in tool.function.parameters.properties:
                return f"Unknown parameter: {param_name}"

            param_info = tool.function.parameters.properties[param_name]

            # Type validation
            expected_type = param_info.get("type")
            if expected_type == "string" and not isinstance(param_value, str):
                return f"Parameter {param_name} must be a string"
            elif expected_type == "number" and not isinstance(param_value, (int, float)):
                return f"Parameter {param_name} must be a number"
            elif expected_type == "integer" and not isinstance(param_value, int):
                return f"Parameter {param_name} must be an integer"
            elif expected_type == "boolean" and not isinstance(param_value, bool):
                return f"Parameter {param_name} must be a boolean"

            # Enum validation
            if "enum" in param_info and param_value not in param_info["enum"]:
                return f"Parameter {param_name} must be one of: {param_info['enum']}"

        return None

    @staticmethod
    def extract_tool_call(response: str) -> Optional[Dict[str, Any]]:
        """Extract tool call from LLM response.

        Args:
            response: Raw response from LLM

        Returns:
            Extracted tool call if found, None otherwise
        """
        # Look for JSON block in response
        try:
            # Try to parse entire response as JSON first
            return json.loads(response)
        except json.JSONDecodeError:
            # If that fails, try to find JSON block
            import re

            json_pattern = r"\{[\s\S]*\}"
            match = re.search(json_pattern, response)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            return None

    @staticmethod
    def parse_tool_response(tool_call: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """Parse validated tool call into components.

        Args:
            tool_call: Validated tool call

        Returns:
            Tuple of (tool_type, function_name, arguments)
        """
        call = tool_call["tool_calls"][0]
        return (call["type"], call["function"]["name"], call["function"]["arguments"])
