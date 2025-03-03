# aim/chat/strategy/base.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from abc  import ABC, abstractmethod
from typing import List, Dict, Optional

from ...agents.persona import Persona
from ..manager import ChatManager


class ChatTurnStrategy(ABC):
    """
    Generates a chat turn for the user input, augmenting the response with information from the database.
    
    This method is responsible for generating the chat turns that will be passed to the chat completion API. It retrieves relevant memories from the conversation model and inserts them into the chat history at the appropriate fold depth.
    
    Args:
        user_input (str): The current user input.
        history (List[Dict[str, str]]): The chat history up to this point.
    
    Returns:
        List[Dict[str, str]]: The updated chat turns, including the user input and any inserted memories.
    """
        
    def __init__(self, chat : ChatManager):
        self.chat = chat
        self.pinned : list[str] = []
        self.thought_content : Optional[str] = None
        self.scratch_pad : Optional[str] = None

    def pin_message(self, doc_id: str):
        if doc_id not in self.pinned:
            self.pinned.append(doc_id)
        
    def clear_pinned(self):
        self.pinned = []

    @abstractmethod
    def user_turn_for(self, persona: Persona, user_input: str, history: List[Dict[str, str]] = []) -> Dict[str, str]:
        """
        Generate a user turn for a chat session.
        
        This is what will be stored in the history.
        
        Args:
            user_input (str): The user input.
            history (List[Dict[str, str]]): The chat history.

        Returns:
            Dict[str, str]: The user turn, in the format {"role": "user", "content": user_input}.
        """
        return {"role": "user", "content": user_input}
        
    @abstractmethod
    def chat_turns_for(self, persona: Persona, user_input: str, history: List[Dict[str, str]] = [], content_len: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Generate a chat session, augmenting the response with information from the database.

        This is what will be passed to the chat complletion API.

        Args:
            user_input (str): The user input.
            history (List[Dict[str, str]]): The chat history.
            
        Returns:
            List[Dict[str, str]]: The chat turns, in the alternating format [{"role": "user", "content": user_input}, {"role": "assistant", "content": assistant_turn}].
        """
        return [*history, {"role": "user", "content": user_input + "\n\n"}]

