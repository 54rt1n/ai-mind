"""Utility functions for session module."""

from typing import List
from ...agents.persona import Tool
from ...tool.dto import Tool as DtoTool, ToolFunction, ToolFunctionParameters

def convert_persona_tools_to_dto(persona_tools: List[Tool]) -> List[DtoTool]:
    """Convert persona tools to DTO format expected by ToolUser.
    
    Args:
        persona_tools: List of persona Tool objects
        
    Returns:
        List of DTO Tool objects compatible with ToolUser
    """
    formatted_tools = []
    
    for tool in persona_tools:
        # Create a basic parameter structure
        parameters = ToolFunctionParameters(
            type="object",
            properties={"input": {"type": "string", "description": "Input for the tool"}},
            required=["input"],
            examples=[{"input": "Example input"}]
        )
        
        # Create the function definition
        function = ToolFunction(
            name=tool.item,
            description=tool.description,
            parameters=parameters
        )
        
        # Create the complete tool definition
        formatted_tool = DtoTool(
            type=tool.type,
            function=function
        )
        
        formatted_tools.append(formatted_tool)
        
    return formatted_tools 