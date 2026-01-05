# aim/server/modules/memory/route.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

import logging
import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from aim.config import ChatConfig
from aim.chat import ChatManager
from aim.agents.roster import Roster
from aim.constants import CHUNK_LEVEL_FULL
from aim.conversation.message import VISIBLE_COLUMNS

from .dto import DocumentUpdate, CreateDocumentRequest

logger = logging.getLogger(__name__)

# Columns safe to return in API responses
API_COLUMNS = VISIBLE_COLUMNS + ['timestamp', 'speaker', 'date', 'score']


def safe_value(val):
    """Convert NaN to None for JSON compatibility."""
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    return val


def df_to_json_safe(df) -> list:
    """Convert DataFrame to list of dicts, replacing NaN with None for JSON compatibility."""
    return df.replace({np.nan: None}).to_dict(orient='records')


class MemoryModule:
    def __init__(self, config: ChatConfig, security: HTTPBearer, shared_roster: Roster):
        self.router = APIRouter(prefix="/api/memory", tags=["memory"])
        self.security = security
        self.config = config
        self.chat = ChatManager.from_config_with_roster(config, shared_roster)
        
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
                    chunk_level=CHUNK_LEVEL_FULL,
                )
                
                formatted_results = []
                for _, row in results.iterrows():
                    formatted_results.append({
                        "doc_id": safe_value(row['doc_id']),
                        "document_type": safe_value(row['document_type']),
                        "user_id": safe_value(row['user_id']),
                        "persona_id": safe_value(row['persona_id']),
                        "conversation_id": safe_value(row['conversation_id']),
                        "date": safe_value(row['date']),
                        "role": safe_value(row['role']),
                        "content": safe_value(row['content']),
                        "branch": safe_value(row['branch']),
                        "sequence_no": safe_value(row['sequence_no']),
                        "speaker": safe_value(row['speaker']),
                        "score": safe_value(row['score'])
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
                # Filter to only public columns
                available_cols = [c for c in API_COLUMNS if c in document.columns]
                filtered = document[available_cols]
                return {"status": "success", "data": df_to_json_safe(filtered)}
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
