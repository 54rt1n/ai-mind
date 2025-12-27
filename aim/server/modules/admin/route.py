# aim/server/modules/admin/route.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ....config import ChatConfig
from ....conversation.loader import ConversationLoader
from ....conversation.index import SearchIndex
from ....utils.redis_cache import RedisCache

logger = logging.getLogger(__name__)

class AdminModule:
    def __init__(self, config: ChatConfig, security: HTTPBearer):
        self.router = APIRouter(prefix="/api/admin", tags=["admin"])
        self.security = security
        self.config = config
        self.index_path = Path("memory/indices")
        self.loader = ConversationLoader()
        
        self.setup_routes()

    async def rebuild_index_task(self) -> None:
        """Background task to rebuild the index"""
        try:
            # Create a new index
            index = SearchIndex(self.index_path, embedding_model=self.config.embedding_model)
            
            # Load all conversations
            messages = self.loader.load_all()
            
            # Convert to index documents
            documents = [
                msg.to_index_doc()
                for msg in messages
            ]
            
            # Index the documents
            logger.info("Starting indexing...")
            index.add_documents(documents)
            
            logger.info("Index rebuild complete")
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            raise

    def setup_routes(self):
        @self.router.post("/rebuild_index")
        async def rebuild_index(
            background_tasks: BackgroundTasks,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Rebuild the search index from JSONL files"""
            try:
                if self.config.server_api_key and (credentials is None or credentials.credentials != self.config.server_api_key):
                    raise HTTPException(status_code=401, detail="Invalid API key")
                
                background_tasks.add_task(self.rebuild_index_task)
                
                return {
                    "status": "success",
                    "message": "Index rebuild started in background"
                }

            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/refiner/status")
        async def get_refiner_status(
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Get the current refiner enabled/disabled status."""
            try:
                if self.config.server_api_key and (credentials is None or credentials.credentials != self.config.server_api_key):
                    raise HTTPException(status_code=401, detail="Invalid API key")

                cache = RedisCache(self.config)
                enabled = cache.is_refiner_enabled()

                return {
                    "status": "success",
                    "enabled": enabled
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/refiner/toggle")
        async def toggle_refiner(
            request: dict,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Toggle the refiner enabled/disabled status.

            Request body: {"enabled": true/false}
            """
            try:
                if self.config.server_api_key and (credentials is None or credentials.credentials != self.config.server_api_key):
                    raise HTTPException(status_code=401, detail="Invalid API key")

                enabled = request.get("enabled")
                if enabled is None:
                    raise HTTPException(status_code=400, detail="Missing 'enabled' field in request body")

                cache = RedisCache(self.config)
                success = cache.set_refiner_enabled(bool(enabled))

                if not success:
                    raise HTTPException(status_code=500, detail="Failed to update refiner status in Redis")

                return {
                    "status": "success",
                    "enabled": bool(enabled),
                    "message": f"Refiner {'enabled' if enabled else 'disabled'}"
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))