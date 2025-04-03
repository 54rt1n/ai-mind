from typing import Dict, Any, List, Optional, Tuple
from .base import BaseTurn
from ....config import ChatConfig
from ....utils.xml import XmlFormatter
from ....tool.formatting import ToolUser
from ....agents import Persona
from ..state import SessionState, CurrentState


class ConversationTurn(BaseTurn):
    """Represents a conversation turn in a session."""
    
    def __init__(self, user_input: str, persona: Persona, system_message: Optional[str] = None, max_tokens: Optional[int] = None, temperature: Optional[float] = None):
        super().__init__("conversation", persona)
        self.user_input = user_input
        self.system_message = system_message
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def _update_formatter(self, formatter: XmlFormatter, session_state: SessionState, current_state: CurrentState) -> XmlFormatter:
        """Update the formatter with turn-specific elements."""
        # Add system message if provided to this turn
        if self.system_message:
            formatter.add_element("SystemMessage", content=self.system_message, priority=3)
        
        # Conversation turns don't add any tools to the formatter
        return formatter
        
    def _update_config(self, config: ChatConfig, session_state: SessionState, current_state: CurrentState) -> ChatConfig:
        """Update the config with conversation-specific settings."""

        if self.max_tokens is not None:
            config.max_tokens = self.max_tokens
        if self.temperature is not None:
            config.temperature = self.temperature

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
            
        return state, current_state