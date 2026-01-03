# aim/tool/dto.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from pydantic import BaseModel, Field
from typing import Any, Optional


class ToolFunctionParameters(BaseModel):
    type: str = Field("object", description="Type of the parameters object")
    properties: dict[str, dict[str, Any]] = Field(..., description="Properties of the function parameters")
    required: list[str] = Field(..., description="List of required parameters")
    examples: Optional[list[dict[str, Any]]] = Field(None, description="List of examples for the function")


class ToolFunction(BaseModel):
    name: str = Field(..., description="Name of the function")
    description: str = Field(..., description="Description of what the function does")
    parameters: ToolFunctionParameters = Field(..., description="Parameters schema for the function")


class Tool(BaseModel):
    type: str = Field(..., description="Type of the tool (e.g., 'web')")
    function: ToolFunction = Field(..., description="Function definition for the tool")

