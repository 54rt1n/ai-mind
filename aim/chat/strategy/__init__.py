# aim/chat/strategy/__init__.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from ..manager import ChatManager
from .base import ChatTurnStrategy
from .simple import SimpleTurnStrategy
from .xmlmemory import XMLMemoryTurnStrategy

def chat_strategy_for(chat_strategy: str, chat: ChatManager) -> ChatTurnStrategy:
    if chat_strategy == "simple":
        return SimpleTurnStrategy(chat)
    elif chat_strategy == "xmlmemory":
        return XMLMemoryTurnStrategy(chat)
    else:
        raise ValueError(f"Unknown chat strategy: {chat_strategy}")