# aim/server/session/turn/base.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from typing import Tuple, AsyncGenerator
from abc import ABC, abstractmethod
import copy

from ....config import ChatConfig
from ....tool.formatting import ToolUser
from ....utils.xml import XmlFormatter
from ....agents import Persona
from ..state import SessionState, CurrentState

class BaseTurn(ABC):
    """Base class for all turn types in a session."""
    
    def __init__(self, turn_type: str, persona: Persona):
       self.turn_type = turn_type
       self.persona = persona

    @abstractmethod
    def _update_formatter(self, formatter: XmlFormatter, session_state: SessionState, current_state: CurrentState) -> XmlFormatter:
        """Update the formatter with the persona's data."""
        pass

    @abstractmethod
    def _update_config(self, config: ChatConfig, session_state: SessionState, current_state: CurrentState) -> ChatConfig | None:
        """Update the config for the session."""
        pass

    def get_config(self, config: ChatConfig, session_state: SessionState, current_state: CurrentState, conversation_length: int) -> ChatConfig | None:
        """Get the config for the session."""

        # We need a deep copy of the config
        local_config = copy.deepcopy(config)

        # Validate model
        selected_model = session_state.models.get(self.turn_type)
        if not selected_model:
            raise ValueError(f"Invalid model for turn type: {self.turn_type}")
            
        # Set persona and user
        local_config.user_id = session_state.user_id
        local_config.persona_id = session_state.persona_id

        local_config = self._update_config(local_config, session_state, current_state)
        
        # Prepare the common system message with persona
        system_formatter = XmlFormatter()
        
        # Let the persona decorate the formatter with its data
        system_formatter = self.persona.xml_decorator(
            system_formatter,
            mood=local_config.persona_mood,
            user_id=session_state.user_id,
            location=session_state.location or self.persona.default_location,
            conversation_length=conversation_length,
        )

        system_formatter = self._update_formatter(system_formatter, session_state, current_state)
        
        # Set the formatted system message
        local_config.system_message = system_formatter.render().replace("{{user}}", session_state.user_id)
        
        return local_config
    
    @abstractmethod
    def get_prompt(self) -> str:
        """Get the prompt for the turn."""
        pass

    @abstractmethod
    def process_response(self, response: str, state: SessionState, current_state: CurrentState) -> Tuple[SessionState, CurrentState]:
        """Handle the response from the model."""
        pass
