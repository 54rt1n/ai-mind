# mindai/server/modules/memory/route.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ....config import ChatConfig
from ....chat import ChatManager

from .dto import DocumentUpdate, CreateDocumentRequest

logger = logging.getLogger(__name__)


class MemoryModule:
    def __init__(self, config: ChatConfig, security: HTTPBearer):
        self.router = APIRouter(prefix="/api/memory", tags=["memory"])
        self.security = security
        self.config = config
        self.chat = ChatManager.from_config(config)
        
        self.setup_routes()

    def setup_routes(self):
        @self.router.get("/search")
        async def search_memory(
            query: str,
            top_n: int = 5,
            document_type: str = 'all',
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Search through memory documents"""
            try:
                if document_type == 'all':
                    document_type = None
                results = self.chat.cvm.query(
                    [query],
                    top_n=top_n,
                    query_document_type=document_type,
                )
                
                formatted_results = []
                for _, row in results.iterrows():
                    formatted_results.append({
                        "doc_id": row['doc_id'],
                        "document_type": row['document_type'],
                        "user_id": row['user_id'],
                        "persona_id": row['persona_id'],
                        "conversation_id": row['conversation_id'],
                        "date": row['date'],
                        "role": row['role'],
                        "content": row['content'],
                        "branch": row['branch'],
                        "sequence_no": row['sequence_no'],
                        "speaker": row['speaker'],
                        "score": row['score']
                    })

                return {"status": "success", "results": formatted_results}
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.put("/{conversation_id}/{document_id}")
        async def update_document(
            conversation_id: str,
            document_id: str,
            document: DocumentUpdate,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Update a document"""
            try:
                update_data = document.data
                self.chat.cvm.update_document(conversation_id=conversation_id, document_id=document_id, update_data=update_data)
                return {"status": "success", "message": f"Document {document_id} updated"}
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("")
        async def create_document(
            document: CreateDocumentRequest,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Create a document"""
            try:
                self.chat.cvm.insert(document.message)
                return {"status": "success", "message": f"Document {document.message.doc_id} created"}
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/{document_id}")
        async def get_document(
            document_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Get a specific document"""
            try:
                document = self.chat.cvm.get_documents(document_ids=[document_id])
                return {"status": "success", "data": document.to_dict()}
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/{conversation_id}/{document_id}/remove")
        async def delete_document(
            conversation_id: str,
            document_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Delete a document"""
            try:
                self.chat.cvm.delete_document(conversation_id, document_id)
                return {"status": "success", "message": f"Document {document_id} deleted"}
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))
