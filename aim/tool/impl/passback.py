# aim/tool/impl/passback.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from typing import Dict
from .base import ToolImplementation


class PassBackImplementation(ToolImplementation):
    """Implementation that passes parameters directly to client.
    
    This implementation is used for tools where the actual execution
    happens on the client side. It simply forwards the parameters
    in a standardized format.
    """
    
    def execute(self, parameters: Dict[str, str]) -> Dict[str, str]:
        """Pass parameters through to client.
        
        Args:
            parameters: Dictionary of parameter names and values
            
        Returns:
            Dictionary containing:
                - action: The tool type and function name
                - parameters: The original parameters
        """
        return {
            "action": "pass",
            "parameters": parameters
        } 