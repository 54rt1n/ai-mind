# aim/server/modules/tools/route.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from aim.config import ChatConfig
from aim.tool.loader import ToolLoader

from .dto import CreateToolRequest, UpdateToolRequest, ToolResponse, ToolListResponse, Tool

logger = logging.getLogger(__name__)

class ToolsModule:
    def __init__(self, config: ChatConfig, security: HTTPBearer):
        self.router = APIRouter(prefix="/api/tools", tags=["tools"])
        self.security = security
        self.config = config
        self.tool_loader = ToolLoader.from_config(config)
        
        self.setup_routes()

    def setup_routes(self):
        @self.router.get("")
        async def list_tools(
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """List all available tools"""
            try:
                tools = [
                    ToolResponse(**tool.model_dump())
                    for tool in self.tool_loader.tools
                ]
                return ToolListResponse(tools=tools)
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/{tool_type}")
        async def get_tool(
            tool_type: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Get a specific tool"""
            try:
                tool = self.tool_loader.get_tool(tool_type)
                if not tool:
                    raise HTTPException(status_code=404, detail=f"Tool {tool_type} not found")
                return ToolResponse(**tool.model_dump())
            except HTTPException:
                raise
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("")
        async def create_tool(
            request: CreateToolRequest,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Create a new tool"""
            try:
                # Check if tool already exists
                if self.tool_loader.get_tool(request.type):
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Tool with type {request.type} already exists"
                    )
                
                # Save tool using loader
                self.tool_loader.save_tool(request)
                
                # Get newly created tool
                new_tool = self.tool_loader.get_tool(request.type)
                if not new_tool:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to create tool"
                    )
                
                return {
                    "status": "success",
                    "message": f"Tool {request.type} created",
                    "data": ToolResponse(**new_tool.model_dump())
                }
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.put("/{tool_type}")
        async def update_tool(
            tool_type: str,
            request: UpdateToolRequest,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Update an existing tool"""
            try:
                # Check if tool exists
                existing_tool = self.tool_loader.get_tool(tool_type)
                if not existing_tool:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Tool {tool_type} not found"
                    )
                
                # Create updated tool instance
                updated_tool = Tool(
                    type=request.type or tool_type,
                    function=request.function or existing_tool.function
                )
                
                # Update using loader
                self.tool_loader.update_tool(tool_type, updated_tool)
                
                # Get updated tool
                result_tool = self.tool_loader.get_tool(request.type or tool_type)
                if not result_tool:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to update tool"
                    )
                
                return {
                    "status": "success",
                    "message": f"Tool {tool_type} updated",
                    "data": ToolResponse(**result_tool.model_dump())
                }
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.delete("/{tool_type}")
        async def delete_tool(
            tool_type: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Delete a tool"""
            try:
                # Check if tool exists
                if not self.tool_loader.get_tool(tool_type):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Tool {tool_type} not found"
                    )
                
                # Delete using loader
                self.tool_loader.delete_tool(tool_type)
                
                return {
                    "status": "success",
                    "message": f"Tool {tool_type} deleted"
                }
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))


