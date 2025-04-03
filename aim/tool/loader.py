# aim/tool/loader.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import os
import yaml
from typing import Optional, Type
import importlib
from pathlib import Path
import logging
from .dto import Tool, ToolFunction, ToolFunctionParameters
from .impl.base import ToolImplementation
from ..config import ChatConfig

logger = logging.getLogger(__name__)

class ToolLoader:
    """Loads and manages tool configurations from YAML files."""

    def __init__(self, tools_dir: str = "config/tools"):
        """Initialize the tool loader.
        
        Args:
            tools_dir: Directory containing tool YAML configurations
        """
        self.tools_dir = tools_dir
        self._tools: dict[str, Tool] = {}
        self._implementations: dict[str, Type[ToolImplementation]] = {}
        
    def load_tools(self) -> None:
        """Load all tool configurations from the tools directory."""
        tools_path = Path(self.tools_dir)
        if not tools_path.exists():
            raise ValueError(f"Tools directory not found: {self.tools_dir}")
            
        for config_file in tools_path.glob("*.yaml"):
            try:
                config = self._read_tool_config(config_file)
                
                if "functions" not in config:
                    logger.warning(f"No functions found in config: {config_file}")
                    continue
                
                # Load implementation class if specified
                impl_class : Optional[Type[ToolImplementation]] = None
                if "implementation" in config:
                    try:
                        module = importlib.import_module(config["implementation"]["module"])
                        impl_class = getattr(module, config["implementation"]["class"])
                    except (ImportError, AttributeError) as e:
                        logger.error(f"Failed to load implementation for {config_file}: {e}")
                        continue
                
                # Create Tool instances for each function
                tool_type = config["type"]
                for function in config["functions"]:
                    try:
                        tool = Tool(
                            type=tool_type,
                            function=ToolFunction(
                                name=function["name"],
                                description=function["description"],
                                parameters=ToolFunctionParameters(
                                    type=function["parameters"].get("type", "object"),
                                    properties=function["parameters"].get("properties", {}),
                                    required=function["parameters"].get("required", []),
                                    examples=function["parameters"].get("examples", [])
                                )
                            )
                        )
                        self._tools[f"{tool_type}/{tool.function.name}"] = tool
                        self._implementations[f"{tool_type}/{tool.function.name}"] = impl_class
                    except KeyError as e:
                        logger.error(f"Invalid function config in {config_file}: {e}")
                        continue
                    
            except Exception as e:
                logger.error(f"Failed to load tool config {config_file}: {e}")
                continue

    def get_tool(self, tool_type: str, function_name: Optional[str] = None) -> Optional[Tool]:
        """Get a tool by its type and optionally function name.
        
        Args:
            tool_type: The type identifier of the tool
            function_name: Optional specific function name to get
            
        Returns:
            The Tool instance if found, None otherwise
        """
        if function_name:
            return self._tools.get(f"{tool_type}/{function_name}")
            
        # If no function specified, return first function of type
        for key, tool in self._tools.items():
            if key.startswith(f"{tool_type}/"):
                return tool
        return None
    
    def get_tools_by_type(self, tool_type: str) -> list[Tool]:
        """Get all tools of a specific type.
        
        Args:
            tool_type: The type identifier of the tools
            
        Returns:
            List of Tool instances for that type
        """
        return [
            tool for key, tool in self._tools.items() 
            if key.startswith(f"{tool_type}/")
        ]
    
    def get_implementation(self, tool_type: str, function_name: str) -> Optional[Type[ToolImplementation]]:
        """Get the implementation class for a tool type and function.
        
        Args:
            tool_type: The type identifier of the tool
            function_name: The function name
            
        Returns:
            The implementation class if found, None otherwise
        """
        return self._implementations.get(f"{tool_type}/{function_name}")
    
    @property
    def tools(self) -> list[Tool]:
        """Get all loaded tools."""
        return list(self._tools.values())
    
    def _get_tool_path(self, tool_type: str) -> Path:
        """Get the path for a tool's config file.
        
        Args:
            tool_type: The tool type identifier
            
        Returns:
            Path to the tool's config file
        """
        return Path(self.tools_dir) / f"{tool_type}.yaml"
    
    def _read_tool_config(self, path: Path) -> dict:
        """Safely read a tool config file.
        
        Args:
            path: Path to the config file
            
        Returns:
            The parsed config dict
            
        Raises:
            ValueError: If file doesn't exist or is invalid
        """
        if not path.exists():
            raise ValueError(f"Tool config not found: {path}")
            
        try:
            with open(path) as f:
                config = yaml.safe_load(f)
                
            if not isinstance(config, dict):
                raise ValueError(f"Invalid tool config format: {path}")
                
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse tool config {path}: {e}")
    
    def _write_tool_config(self, path: Path, config: dict) -> None:
        """Safely write a tool config file.
        
        Args:
            path: Path to write to
            config: Config dict to write
            
        Raises:
            ValueError: If write fails
        """
        try:
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first
            temp_path = path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                yaml.safe_dump(config, f)
                
            # Atomic rename
            temp_path.replace(path)
        except (OSError, yaml.YAMLError) as e:
            if temp_path.exists():
                temp_path.unlink()
            raise ValueError(f"Failed to write tool config {path}: {e}")
    
    def save_tool(self, tool: Tool) -> None:
        """Save a new tool function.
        
        Args:
            tool: Tool instance to save
            
        Raises:
            ValueError: If save fails
        """
        path = self._get_tool_path(tool.type)
        
        if path.exists():
            config = self._read_tool_config(path)
            functions = config.get("functions", [])
            
            # Check if function already exists
            for i, func in enumerate(functions):
                if func["name"] == tool.function.name:
                    # Update existing function
                    functions[i] = tool.function.model_dump()
                    logger.info(f"Updated existing function '{tool.function.name}' in tool '{tool.type}'")
                    break
            else:
                # Add new function
                functions.append(tool.function.model_dump())
                logger.info(f"Added new function '{tool.function.name}' to tool '{tool.type}'")
        else:
            # Create new tool config
            config = {
                "type": tool.type,
                "functions": [tool.function.model_dump()]
            }
            logger.info(f"Created new tool '{tool.type}' with function '{tool.function.name}'")
        
        self._write_tool_config(path, config)
        
        # Update internal state
        key = f"{tool.type}/{tool.function.name}"
        self._tools[key] = tool
        if key not in self._implementations:
            self._implementations[key] = PassBackImplementation
    
    def delete_tool_function(self, tool_type: str, function_name: str) -> None:
        """Delete a specific function from a tool.
        
        Args:
            tool_type: Type of tool
            function_name: Name of function to delete
            
        Raises:
            ValueError: If tool/function doesn't exist or delete fails
        """
        path = self._get_tool_path(tool_type)
        if not path.exists():
            raise ValueError(f"Tool not found: {tool_type}")
            
        config = self._read_tool_config(path)
        functions = config.get("functions", [])
        
        # Find function to delete
        new_functions = [f for f in functions if f["name"] != function_name]
        if len(new_functions) == len(functions):
            raise ValueError(f"Function '{function_name}' not found in tool '{tool_type}'")
        
        if new_functions:
            # Update config with remaining functions, preserving other fields
            config["functions"] = new_functions
            self._write_tool_config(path, config)
            logger.info(f"Removed function '{function_name}' from tool '{tool_type}'")
        else:
            # No functions left, but preserve implementation if exists
            if "implementation" in config:
                # Keep the tool with implementation but no functions
                config["functions"] = []
                self._write_tool_config(path, config)
                logger.info(f"Removed last function '{function_name}' from tool '{tool_type}', preserving implementation")
            else:
                # No implementation to preserve, delete the tool file
                try:
                    path.unlink()
                    logger.info(f"Deleted tool '{tool_type}' after removing last function")
                except OSError as e:
                    raise ValueError(f"Failed to delete tool config {path}: {e}")
        
        # Update internal state
        key = f"{tool_type}/{function_name}"
        if key in self._tools:
            del self._tools[key]
        if key in self._implementations:
            del self._implementations[key]
    
    def delete_tool(self, tool_type: str) -> None:
        """Delete an entire tool and all its functions.
        
        Args:
            tool_type: Type of tool to delete
            
        Raises:
            ValueError: If tool doesn't exist or delete fails
        """
        path = self._get_tool_path(tool_type)
        if not path.exists():
            raise ValueError(f"Tool not found: {tool_type}")
            
        try:
            path.unlink()
        except OSError as e:
            raise ValueError(f"Failed to delete tool config {path}: {e}")
            
        # Update internal state - remove all functions of this type
        keys_to_delete = [
            key for key in self._tools.keys() 
            if key.startswith(f"{tool_type}/")
        ]
        for key in keys_to_delete:
            del self._tools[key]
            if key in self._implementations:
                del self._implementations[key]
    
    @classmethod
    def from_config(cls, config: ChatConfig) -> "ToolLoader":
        """Create a ToolLoader from a ChatConfig."""
        self = cls(config.tools_path)
        self.load_tools()
        return self
