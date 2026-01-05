# aim/server/modules/conversation/route.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

import time
import logging
import numpy as np
from typing import Callable
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse

from aim.config import ChatConfig
from aim.chat import ChatManager
from aim.conversation.message import ConversationMessage, VISIBLE_COLUMNS
from aim.constants import DOC_CONVERSATION

from .dto import SaveConversationRequest

logger = logging.getLogger(__name__)

# Columns safe to return in API responses
API_COLUMNS = VISIBLE_COLUMNS + ['timestamp', 'speaker', 'date']


def df_to_json_safe(df) -> list:
    """Convert DataFrame to list of dicts, replacing NaN with None for JSON compatibility."""
    return df.replace({np.nan: None}).to_dict(orient='records')

class ConversationModule:
    def __init__(self, config: ChatConfig, security: HTTPBearer, get_chat_manager: Callable[[str], ChatManager]):
        self.router = APIRouter(prefix="/api/conversation", tags=["conversation"])
        self.security = security
        self.config = config
        self.get_chat_manager = get_chat_manager

        self.setup_routes()

    def setup_routes(self):
        @self.router.get("/{persona_id}")
        async def list_conversations(
            persona_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """List all conversations for a persona"""
            try:
                chat = self.get_chat_manager(persona_id)
                df = chat.cvm.get_conversation_report()
                return {
                    "status": "success",
                    "message": f"{len(df)} conversations",
                    "data": df_to_json_safe(df)
                }
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/{persona_id}")
        async def save_conversation(
            persona_id: str,
            request: SaveConversationRequest,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Save a new conversation for a persona"""
            try:
                chat = self.get_chat_manager(persona_id)
                for i, msg in enumerate(request.messages):
                    timestamp = msg.timestamp if msg.timestamp else int(time.time())
                    message = ConversationMessage(
                        doc_id=ConversationMessage.next_doc_id(),
                        document_type=DOC_CONVERSATION,
                        user_id=self.config.user_id,
                        persona_id=persona_id,
                        conversation_id=request.conversation_id,
                        branch=0,
                        sequence_no=i,
                        speaker_id=self.config.user_id if msg.role == "user" else persona_id,
                        listener_id=persona_id if msg.role == "user" else self.config.user_id,
                        role=msg.role,
                        content=msg.content,
                        timestamp=timestamp,
                        think=msg.think,
                    )
                    chat.cvm.insert(message)

                chat.cvm.refresh()

                return {"status": "success", "message": "Conversation saved successfully"}
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/{persona_id}/{conversation_id}")
        async def get_conversation(
            persona_id: str,
            conversation_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Get a specific conversation for a persona"""
            try:
                chat = self.get_chat_manager(persona_id)
                conversation = chat.cvm.get_conversation_history(conversation_id=conversation_id)
                # Filter to only public columns
                available_cols = [c for c in API_COLUMNS if c in conversation.columns]
                filtered = conversation[available_cols]
                return {
                    "status": "success",
                    "message": f"{len(conversation)} messages",
                    "data": df_to_json_safe(filtered)
                }
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/{persona_id}/{conversation_id}/remove")
        async def delete_conversation(
            persona_id: str,
            conversation_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Delete a conversation for a persona"""
            try:
                chat = self.get_chat_manager(persona_id)
                chat.cvm.delete_conversation(conversation_id)
                return {"status": "success", "message": f"Conversation {conversation_id} deleted"}
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))
