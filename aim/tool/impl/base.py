# aim/tool/impl/base.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 


class ToolImplementation:
    """Base class for tool implementations."""
    
    def execute(self, parameters: dict[str, str]) -> dict[str, str]:
        """Execute the tool with given parameters.
        
        Args:
            parameters: Dictionary of parameter names and values
            
        Returns:
            Dictionary containing the tool's response
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If execution fails
        """
        raise NotImplementedError()

