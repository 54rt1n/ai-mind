# aim/server/session/state.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from dataclasses import dataclass, field
from typing import Optional, List
import copy

from ...agents import Roster, Persona
from ...config import ChatConfig

@dataclass
class SessionState:
    """Persistent session state for chat interactions."""
    
    session_id: str | None = None
    user_id: str | None = None
    persona_id: str | None = None
    conversation_id: str | None = None 
    location: str | None = None
    save_conversation: bool = False
    models: dict[str, str] = field(default_factory=dict)

    def get_persona(self, config: ChatConfig) -> Persona | None:
        """Get the persona for the session.
        
        Args:
            config (ChatConfig): Config needed to initialize the roster
            
        Returns:
            Persona | None: The persona for this session, or None if not found
        """
        if not self.persona_id:
            return None
            
        # Create a new roster on each call - don't cache it
        roster = Roster.from_config(config)
        return roster.get_persona(self.persona_id)



@dataclass
class CurrentState:
    """Transient state for the current session."""

    # Thought processing state
    thought_iteration: int = 1
    thought_content: str | None = None
    current_thought_response: list[str] = field(default_factory=list)

    # Tool processing state
    current_tool_response: list[str] = field(default_factory=list)
    current_tool_result: dict | None = None
    
    # Response state
    current_conversation_response: str | None = None
    
    # Tool/workspace state
    current_workspace: str | None = None
    current_scratch_pad: str | None = None

    # Pinned messages
    pinned_messages: list[str] = field(default_factory=list)
    
    def reset_turn(self) -> None:
        """Reset the turn state to defaults."""
        self.current_conversation_response = None
        
    def reset_thought(self) -> None:
        """Reset the thought state to defaults."""
        self.thought_iteration = 1
        self.current_thought_response = []
        
    def reset_tool(self) -> None:
        """Reset the tool state to defaults."""
        self.current_tool_response = []
        self.current_tool_result = None
        
    def reset_all(self) -> None:
        """Reset all transient state to defaults."""
        self.reset_turn()
        self.reset_thought()
        self.reset_tool()
        self.current_workspace = None
        self.current_scratch_pad = None


