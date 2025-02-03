# mindai/server/modules/memory/dto.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from pydantic import BaseModel
from typing import Any

from ....conversation.message import ConversationMessage


class DocumentUpdate(BaseModel):
    data: dict[str, Any]

class CreateDocumentRequest(BaseModel):
    message: ConversationMessage