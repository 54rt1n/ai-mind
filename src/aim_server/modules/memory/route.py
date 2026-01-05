# aim/server/modules/memory/route.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

import logging
import numpy as np
from typing import Callable
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from aim.config import ChatConfig
from aim.chat import ChatManager
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
    def __init__(self, config: ChatConfig, security: HTTPBearer, get_chat_manager: Callable[[str], ChatManager]):
        self.router = APIRouter(prefix="/api/memory", tags=["memory"])
        self.security = security
        self.config = config
        self.get_chat_manager = get_chat_manager

        self.setup_routes()

    def setup_routes(self):
        @self.router.get("/{persona_id}/search")
        async def search_memory(
            persona_id: str,
            query: str,
            top_n: int = 5,
            document_type: str = 'all',
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Search through persona's memory documents"""
            try:
                chat = self.get_chat_manager(persona_id)
                if document_type == 'all':
                    document_type = None
                results = chat.cvm.query(
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

        @self.router.put("/{persona_id}/{conversation_id}/{document_id}")
        async def update_document(
            persona_id: str,
            conversation_id: str,
            document_id: str,
            document: DocumentUpdate,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Update a document"""
            try:
                chat = self.get_chat_manager(persona_id)
                update_data = document.data
                chat.cvm.update_document(conversation_id=conversation_id, document_id=document_id, update_data=update_data)
                return {"status": "success", "message": f"Document {document_id} updated"}
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/{persona_id}")
        async def create_document(
            persona_id: str,
            document: CreateDocumentRequest,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Create a document"""
            try:
                chat = self.get_chat_manager(persona_id)
                chat.cvm.insert(document.message)
                return {"status": "success", "message": f"Document {document.message.doc_id} created"}
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/{persona_id}/{document_id}")
        async def get_document(
            persona_id: str,
            document_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Get a specific document"""
            try:
                chat = self.get_chat_manager(persona_id)
                document = chat.cvm.get_documents(document_ids=[document_id])
                # Filter to only public columns
                available_cols = [c for c in API_COLUMNS if c in document.columns]
                filtered = document[available_cols]
                return {"status": "success", "data": df_to_json_safe(filtered)}
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/{persona_id}/{conversation_id}/{document_id}/remove")
        async def delete_document(
            persona_id: str,
            conversation_id: str,
            document_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Delete a document"""
            try:
                chat = self.get_chat_manager(persona_id)
                chat.cvm.delete_document(conversation_id, document_id)
                return {"status": "success", "message": f"Document {document_id} deleted"}
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/{persona_id}/rebuild")
        async def rebuild_index(
            persona_id: str,
            background_tasks: BackgroundTasks,
            full: bool = False,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Rebuild memory index for a persona"""
            try:
                from aim.conversation.utils import rebuild_agent_index

                chat = self.get_chat_manager(persona_id)
                embedding_model = chat.config.embedding_model

                # Run rebuild in background
                background_tasks.add_task(
                    rebuild_agent_index,
                    agent_id=persona_id,
                    embedding_model=embedding_model,
                    full=full
                )

                return {
                    "status": "success",
                    "message": f"Rebuild started for {persona_id}",
                    "mode": "full" if full else "incremental"
                }
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))
