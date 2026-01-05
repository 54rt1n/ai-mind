import json
from typing import Dict, Any, List, Optional, Tuple

from aim.config import ChatConfig
from aim.utils.xml import XmlFormatter
from aim.tool.formatting import ToolUser
from aim.agents import Persona

from .base import BaseTurn
from ..state import SessionState, CurrentState


class ToolTurn(BaseTurn):
    """Represents a tool usage turn in the session."""
    
    def __init__(self, persona: Persona, dto_tools=None):
        super().__init__("tool", persona)
        self.parameters = {}
        self.tool_name = "unknown_tool"
        self.dto_tools = dto_tools
        
    def _update_formatter(self, formatter: XmlFormatter, session_state: SessionState, current_state: CurrentState) -> XmlFormatter:
        """Update the formatter with turn-specific elements."""
        # Add tools if they were passed in the constructor
        if self.dto_tools:
            tool_user = ToolUser(self.dto_tools)
            formatter = tool_user.xml_decorator(formatter)
            
        return formatter
    
    def _update_config(self, config: ChatConfig, session_state: SessionState, current_state: CurrentState) -> ChatConfig:
        """Update the config with tool-specific settings."""
        # Tools require JSON output format
        config.response_format = "json"
        
        # Temperature should be lower for more deterministic tool usage
        config.temperature = 0.2
        config.max_tokens = 256

        # Add tool-specific system message
        tool_system = self._get_system_message()
        config.system_message = f"{config.system_message}\n\n{tool_system}"
        
        return config
        
    def _get_system_message(self) -> str:
        """Get the system message for tool usage."""
        return (
            "<format_override>\n"
            "\t<override>You are using a tool. You must output a valid JSON response.</override>\n"
            "\t<output_mode>json</output_mode>\n"
            "\t<description>Tool Usage Requires JSON Output</description>\n"
            "</format_override>"
        )
        
    def get_prompt(self) -> str:
        """Get the prompt for the turn.
        
        Returns:
            A formatted tool usage prompt
        """
        # Format parameters for display
        formatted_params = json.dumps(self.parameters, indent=2)
        
        return (
            f"<tool_request>\n"
            f"  <tool_name>{self.tool_name}</tool_name>\n"
            f"  <parameters>\n{formatted_params}\n  </parameters>\n"
            f"</tool_request>\n\n"
            f"Please use the {self.tool_name} tool with the provided parameters. "
            f"Return a valid JSON object with the result."
        )

    def process_response(self, response: str, state: SessionState, current_state: CurrentState) -> Tuple[SessionState, CurrentState]:
        """Process the response from the model.
        
        Args:
            response: The model's response text
            state: Persistent session state
            current_state: Current transient state
            
        Returns:
            Updated state and current_state
        """
        # Extract JSON from response
        try:
            # Try to find JSON in the response
            json_start = response.find("{")
            json_end = response.rfind("}")
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end+1]
                tool_result = json.loads(json_str)
                
                # Store tool result in current state
                current_state.current_tool_result = tool_result
                
                # If tool produces workspace content, update it
                if "workspace" in tool_result:
                    current_state.current_workspace = tool_result["workspace"]
            else:
                # No valid JSON found
                current_state.current_tool_result = {"error": "No valid JSON found in response"}
                
        except json.JSONDecodeError:
            # Invalid JSON
            current_state.current_tool_result = {"error": "Invalid JSON in tool response"}
            
        except Exception as e:
            # Other errors
            current_state.current_tool_result = {"error": f"Error processing tool response: {str(e)}"}
            
        return state, current_state