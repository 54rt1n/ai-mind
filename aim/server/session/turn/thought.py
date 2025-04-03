import re
from typing import Dict, Any, List, Optional, Tuple
from .base import BaseTurn
from ....config import ChatConfig
from ....utils.xml import XmlFormatter
from ....tool.formatting import ToolUser
from ....agents import Persona
from ..state import SessionState, CurrentState


class ThoughtTurn(BaseTurn):
    """Represents an internal thought process turn for the agent."""
    
    def __init__(self, persona: Persona, dto_tools=None):
        super().__init__("thought", persona)
        self.dto_tools = dto_tools
    
    def _update_formatter(self, formatter: XmlFormatter, session_state: SessionState, current_state: CurrentState) -> XmlFormatter:
        """Update the formatter with turn-specific elements."""
        # Add tools if they were passed in the constructor
        if self.dto_tools:
            tool_user = ToolUser(self.dto_tools)
            formatter = tool_user.xml_decorator(formatter)
            
        return formatter
    
    def _update_config(self, config: ChatConfig, session_state: SessionState, current_state: CurrentState) -> ChatConfig:
        """Update the config with thought-specific settings."""
        # Adjust settings for thought generation
        # Typically want higher temperature for more creative thinking
        config.temperature = max(0.7, getattr(config, 'temperature', 0.7) + 0.1)
        config.max_tokens = 2048
        
        # Add specific system message for thought mode
        thought_system = self._get_system_message()
        config.system_message = f"{config.system_message}\n\n{thought_system}"
        
        return config
    
    def _get_system_message(self) -> str:
        """Get the system message for thought formatting."""
        return (
            "<format_override>\n"
            "\t<override>You are in your thought processes. You are only to output a thought turn.</override>\n"
            "\t<output_mode>xml</output_mode>\n"
            "\t<description>All Thought Output Is In XML Format</description>\n"
            "</format_override>"
        )
    
    def _get_default_thought(self) -> str:
        """Get default XML-formatted thought content."""
        return (
            "<think iter=\"0\">\n"
            "\t<thought>I will follow a chain of thought, reasoning through my ideas.</thought>\n"
            "\t<thought>These are the most important things I should consider:</thought>\n"
            "</think>"
        )
    
    def get_prompt(self) -> str:
        """Get the prompt for the turn.
        
        Returns:
            A formatted thought prompt
        """
        default_xml = self._get_default_thought()
        
        return (
            f"Thought Turn Format:\n\n{default_xml}\n\n"
            f"<directive>Your next turn is a thought turn. Please update your thought block appropriately, "
            f"enhancing and improving your current thoughts and reasoning. "
            f"Please output the next thoughts document. This should be an xml document.</directive>\n\n"
            f"[~~ Begin XML Output \"<think iter=\"1\">\" ~~]"
        )
        
    def process_response(self, response: str, state: SessionState, current_state: CurrentState) -> Tuple[SessionState, CurrentState]:
        """Process the response from the model.
        
        Args:
            response: The model's response text
            state: Persistent session state
            current_state: Current state to update
            
        Returns:
            Updated state and current_state
        """
        # Extract content between <think> tags
        think_pattern = r'<think.*?>(.*?)</think>'
        matches = re.findall(think_pattern, response, re.DOTALL)
        
        if matches:
            # Store the content of the think tags
            current_state.thought_content = matches[-1].strip()
            current_state.thought_iteration += 1
        else:
            # If no think tags found, store a cleaned version of the response
            lines = response.strip().split("\n")
            # Remove any lines that look like XML tags but aren't <thought> content
            filtered_lines = [line for line in lines if not (line.strip().startswith("<") and 
                                                           line.strip().endswith(">") and
                                                           "<thought>" not in line)]
            current_state.thought_content = "\n".join(filtered_lines)
            
        return state, current_state