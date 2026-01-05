# aim/server/modules/report/route.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

import logging
from typing import Callable
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from aim.config import ChatConfig
from aim.chat import ChatManager
from aim.utils.keywords import get_all_keywords

logger = logging.getLogger(__name__)

class ReportModule:
    def __init__(self, config: ChatConfig, security: HTTPBearer, get_chat_manager: Callable[[str], ChatManager]):
        self.router = APIRouter(prefix="/api/report", tags=["report"])
        self.security = security
        self.config = config
        self.get_chat_manager = get_chat_manager

        self.setup_routes()

    def setup_routes(self):
        @self.router.get("/{persona_id}/conversation_matrix")
        async def get_conversation_matrix(
            persona_id: str,
            #credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Get conversation analysis matrix for a persona"""
            try:
                chat = self.get_chat_manager(persona_id)
                df = chat.cvm.get_conversation_report()
                return {
                    "status": "success",
                    "message": f"{len(df)} conversations",
                    "data": df.set_index('conversation_id').T.to_dict()
                }
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/{persona_id}/symbolic_keywords")
        async def get_symbolic_keywords(
            persona_id: str,
            document_type: str = None,
            #credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Get symbolic keywords analysis for a persona"""
            try:
                chat = self.get_chat_manager(persona_id)
                keywords = get_all_keywords(chat.cvm, document_type=document_type)
                return {
                    "status": "success",
                    "message": f"{len(keywords)} keywords",
                    "data": keywords
                }
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))
