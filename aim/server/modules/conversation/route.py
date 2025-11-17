# aim/server/modules/conversation/route.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import time
import logging
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse

from ....config import ChatConfig
from ....chat import ChatManager
from ....conversation.message import ConversationMessage
from ....constants import DOC_CONVERSATION
from ....agents.roster import Roster

from .dto import SaveConversationRequest

logger = logging.getLogger(__name__)

class ConversationModule:
    def __init__(self, config: ChatConfig, security: HTTPBearer, shared_roster: Roster):
        self.router = APIRouter(prefix="/api/conversation", tags=["conversation"])
        self.security = security
        self.config = config
        self.chat = ChatManager.from_config_with_roster(config, shared_roster)
        
        self.setup_routes()

    def setup_routes(self):
        @self.router.get("")
        async def list_conversations(
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """List all conversations"""
            try:
                df = self.chat.cvm.get_conversation_report()
                return {
                    "status": "success", 
                    "message": f"{len(df)} conversations", 
                    "data": df.to_dict(orient='records')
                }
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("")
        async def save_conversation(
            request: SaveConversationRequest,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Save a new conversation"""
            try:
                for i, msg in enumerate(request.messages):
                    timestamp = msg.timestamp if msg.timestamp else int(time.time())
                    message = ConversationMessage(
                        doc_id=ConversationMessage.next_doc_id(),
                        document_type=DOC_CONVERSATION,
                        user_id=self.config.user_id,
                        persona_id=self.config.persona_id,
                        conversation_id=request.conversation_id,
                        branch=0,
                        sequence_no=i,
                        speaker_id=self.config.user_id if msg.role == "user" else self.config.persona_id,
                        listener_id=self.config.persona_id if msg.role == "user" else self.config.user_id,
                        role=msg.role,
                        content=msg.content,
                        timestamp=timestamp,
                        think=msg.think,
                    )
                    self.chat.cvm.insert(message)

                self.chat.cvm.refresh()
                
                return {"status": "success", "message": "Conversation saved successfully"}
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/{conversation_id}")
        async def get_conversation(
            conversation_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Get a specific conversation"""
            try:
                conversation = self.chat.cvm.get_conversation_history(conversation_id=conversation_id)
                return {
                    "status": "success", 
                    "message": f"{len(conversation)} messages", 
                    "data": [r.to_dict() for i, r in conversation.iterrows()]
                }
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/{conversation_id}/remove")
        async def delete_conversation(
            conversation_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Delete a conversation"""
            try:
                self.chat.cvm.delete_conversation(conversation_id)
                return {"status": "success", "message": f"Conversation {conversation_id} deleted"}
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))