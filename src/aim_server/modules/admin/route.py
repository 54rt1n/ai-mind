# aim/server/modules/admin/route.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from aim.config import ChatConfig
from aim.conversation.loader import ConversationLoader
from aim.conversation.index import SearchIndex
from aim.utils.redis_cache import RedisCache

logger = logging.getLogger(__name__)

class AdminModule:
    def __init__(self, config: ChatConfig, security: HTTPBearer):
        self.router = APIRouter(prefix="/api/admin", tags=["admin"])
        self.security = security
        self.config = config

        self.setup_routes()

    async def rebuild_index_task(self, agent_id: Optional[str] = None) -> None:
        """Background task to rebuild the index.

        Args:
            agent_id: If provided, rebuilds index for specific agent (memory/{agent_id}/).
                     If None, rebuilds global index (memory/).
        """
        try:
            # Resolve paths based on agent_id
            if agent_id:
                index_path = Path(f"memory/{agent_id}/indices")
                conversations_dir = f"memory/{agent_id}/conversations"
                label = agent_id
            else:
                index_path = Path("memory/indices")
                conversations_dir = "memory/conversations"
                label = "global"

            loader = ConversationLoader(conversations_dir)

            # Create a new index
            index = SearchIndex(index_path, embedding_model=self.config.embedding_model)

            # Load all conversations
            messages = loader.load_all()

            # Convert to index documents
            documents = [
                msg.to_index_doc()
                for msg in messages
            ]

            # Index the documents
            logger.info(f"[{label}] Starting indexing ({len(documents)} documents)...")
            index.add_documents(documents)

            logger.info(f"[{label}] Index rebuild complete")

        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            raise

    def setup_routes(self):
        @self.router.post("/rebuild_index")
        async def rebuild_index(
            background_tasks: BackgroundTasks,
            request: Optional[dict] = None,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Rebuild the search index from JSONL files.

            Request body (optional):
                {"agent_id": "andi"}  - Rebuild index for specific agent
                {} or omitted        - Rebuild global index
            """
            try:
                if self.config.server_api_key and (credentials is None or credentials.credentials != self.config.server_api_key):
                    raise HTTPException(status_code=401, detail="Invalid API key")

                agent_id = request.get("agent_id") if request else None
                background_tasks.add_task(self.rebuild_index_task, agent_id)

                label = agent_id or "global"
                return {
                    "status": "success",
                    "message": f"Index rebuild started in background for {label}"
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

        @self.router.get("/dreamer/status")
        async def get_dreamer_status(
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Get the current dreamer paused status."""
            try:
                if self.config.server_api_key and (credentials is None or credentials.credentials != self.config.server_api_key):
                    raise HTTPException(status_code=401, detail="Invalid API key")

                cache = RedisCache(self.config)
                paused = cache.is_dreamer_paused()

                return {
                    "status": "success",
                    "paused": paused
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/dreamer/toggle")
        async def toggle_dreamer(
            request: dict,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Toggle the dreamer paused status.

            Request body: {"paused": true/false}

            When paused=true:
            - DreamerWorker stops processing new jobs from the queue
            - Dream watcher stops triggering new pipelines

            Current jobs in progress will complete before pausing takes effect.
            """
            try:
                if self.config.server_api_key and (credentials is None or credentials.credentials != self.config.server_api_key):
                    raise HTTPException(status_code=401, detail="Invalid API key")

                paused = request.get("paused")
                if paused is None:
                    raise HTTPException(status_code=400, detail="Missing 'paused' field in request body")

                cache = RedisCache(self.config)
                success = cache.set_dreamer_paused(bool(paused))

                if not success:
                    raise HTTPException(status_code=500, detail="Failed to update dreamer status in Redis")

                return {
                    "status": "success",
                    "paused": bool(paused),
                    "message": f"Dreamer {'paused' if paused else 'resumed'}"
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))