# aim/chat/strategy/simple.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from typing import Optional

from ...agents.persona import Persona
from ..manager import ChatManager
from .base import ChatTurnStrategy, DEFAULT_MAX_CONTEXT, DEFAULT_MAX_OUTPUT


class SimpleTurnStrategy(ChatTurnStrategy):
    def __init__(self, chat : ChatManager):
        super().__init__(chat)

    def user_turn_for(self, persona: Persona, user_input: str, history: list[dict[str, str]] = []) -> dict[str, str]:
        return {"role": "user", "content": user_input}

    def chat_turns_for(self, persona: Persona, user_input: str, history: list[dict[str, str]] = [], content_len: Optional[int] = None, max_context_tokens: int = DEFAULT_MAX_CONTEXT, max_output_tokens: int = DEFAULT_MAX_OUTPUT) -> list[dict[str, str]]:
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
