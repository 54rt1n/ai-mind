# aim/server/modules/tools/dto.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from typing import List
from pydantic import BaseModel, Field

from aim.tool.dto import Tool

class CreateToolRequest(Tool):
    """Schema for creating a new tool. Inherits all fields from Tool."""
    pass

class UpdateToolRequest(BaseModel):
    """Schema for updating an existing tool. All fields are optional for partial updates."""
    type: str | None = Field(None, description="Type of the tool (e.g., 'web')")
    function: dict | None = Field(None, description="Function definition for the tool")

class ToolResponse(Tool):
    """Schema for tool response. Inherits all fields from Tool."""
    pass

class ToolListResponse(BaseModel):
    """Schema for listing tools"""
    tools: List[ToolResponse] = Field(..., description="List of all tools") 