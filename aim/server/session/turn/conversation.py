from typing import Dict, Any, List, Optional, Tuple
from .base import BaseTurn
from ....config import ChatConfig
from ....utils.xml import XmlFormatter
from ....tool.formatting import ToolUser
from ....agents import Persona
from ..state import SessionState, CurrentState


class ConversationTurn(BaseTurn):
    """Represents a conversation turn in a session."""
    
    def __init__(self, user_input: str, persona: Persona, system_message: Optional[str] = None):
        super().__init__("conversation", persona)
        self.user_input = user_input
        self.system_message = system_message
    
    def _update_formatter(self, formatter: XmlFormatter, session_state: SessionState, current_state: CurrentState) -> XmlFormatter:
        """Update the formatter with turn-specific elements."""
        # Add system message if provided to this turn
        if self.system_message:
            formatter.add_element("SystemMessage", content=self.system_message, priority=3)
        
        # Conversation turns don't add any tools to the formatter
        return formatter
        
    def _update_config(self, config: ChatConfig, session_state: SessionState, current_state: CurrentState) -> ChatConfig:
        """Update the config with conversation-specific settings."""
        # Apply any conversation-specific configuration
        if hasattr(config, 'temperature'):
            config.temperature = config.temperature
            
        if hasattr(config, 'max_tokens'):
            config.max_tokens = config.max_tokens
            
        # Set response format based on context
        if current_state.current_workspace and "code" in current_state.current_workspace.lower():
            # If workspace contains code, enable code-optimized response
            config.response_format = "code"
            
        return config
        
    def get_prompt(self) -> str:
        """Get the prompt for the turn.
        
        Returns:
            The user input as the prompt
        """
        return self.user_input
        
    def process_response(self, response: str, state: SessionState, current_state: CurrentState) -> Tuple[SessionState, CurrentState]:
        """Process the response from the model.
        
        Args:
            response: The model's response text
            state: Persistent session state
            current_state: Current state to update
            
        Returns:
            Updated state and current_state
        """
        current_state.current_conversation_response = response
        
        # Check for workspace content in the response
        workspace_start = response.find("<workspace>")
        workspace_end = response.find("</workspace>")
        
        if workspace_start != -1 and workspace_end != -1 and workspace_end > workspace_start:
            workspace_content = response[workspace_start + 11:workspace_end].strip()
            current_state.current_workspace = workspace_content
            
        # Check for scratch pad content
        scratch_start = response.find("<scratch>")
        scratch_end = response.find("</scratch>")
        
        if scratch_start != -1 and scratch_end != -1 and scratch_end > scratch_start:
            scratch_content = response[scratch_start + 9:scratch_end].strip()
            current_state.current_scratch_pad = scratch_content
            
        return state, current_state