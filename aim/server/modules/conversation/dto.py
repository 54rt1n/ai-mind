# aim/server/modules/conversation/dto.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from pydantic import BaseModel, Field


class SimpleConversationMessage(BaseModel):
    role: str
    content: str
    timestamp: int = 0
    think: str | None = None


class SaveConversationRequest(BaseModel):
    conversation_id: str
    messages: list[SimpleConversationMessage]
