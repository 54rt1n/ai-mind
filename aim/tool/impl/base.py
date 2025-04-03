# aim/tool/impl/base.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from abc import ABC, abstractmethod
from typing import Dict, Any

class ToolImplementation(ABC):
    """Base class for tool implementations."""
    
    @abstractmethod
    def execute(self, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given parameters.
        
        Args:
            function_name: Name of the function to execute
            parameters: Dictionary of parameter names and values
            
        Returns:
            Dictionary containing the tool's response
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If execution fails
        """
        raise NotImplementedError()

