# aim/server/modules/report/route.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ....config import ChatConfig
from ....chat import ChatManager
from ....utils.keywords import get_all_keywords

logger = logging.getLogger(__name__)

class ReportModule:
    def __init__(self, config: ChatConfig, security: HTTPBearer):
        self.router = APIRouter(prefix="/api/report", tags=["report"])
        self.security = security
        self.config = config
        self.chat = ChatManager.from_config(config)
        
        self.setup_routes()

    def setup_routes(self):
        @self.router.get("/conversation_matrix")
        async def get_conversation_matrix(
            #credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Get conversation analysis matrix"""
            try:
                df = self.chat.cvm.get_conversation_report()
                return {
                    "status": "success",
                    "message": f"{len(df)} conversations",
                    "data": df.set_index('conversation_id').T.to_dict()
                }
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/symbolic_keywords")
        async def get_symbolic_keywords(
            document_type: str = None,
            #credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Get symbolic keywords analysis"""
            try:
                keywords = get_all_keywords(self.chat.cvm, document_type=document_type)
                return {
                    "status": "success",
                    "message": f"{len(keywords)} keywords",
                    "data": keywords
                }
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))