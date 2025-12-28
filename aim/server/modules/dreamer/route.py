# aim/server/modules/dreamer/route.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import redis.asyncio as redis

from ....config import ChatConfig
from ....dreamer.api import (
    start_pipeline,
    get_status,
    cancel_pipeline,
    delete_pipeline,
    resume_pipeline,
    refresh_pipeline,
    list_pipelines,
)
from ....dreamer.state import StateStore
from ....dreamer.scheduler import Scheduler
from ....dreamer.scenario import load_scenario

from .dto import (
    StartPipelineRequest,
    StartPipelineResponse,
    PipelineStatusResponse,
    ListPipelinesResponse,
    CancelPipelineResponse,
    DeletePipelineResponse,
    ResumePipelineResponse,
)

logger = logging.getLogger(__name__)


class DreamerModule:
    """FastAPI module for Dreamer pipeline management."""

    def __init__(self, config: ChatConfig, security: HTTPBearer):
        self.router = APIRouter(prefix="/api/dreamer", tags=["dreamer"])
        self.security = security
        self.config = config
        self._redis_client: Optional[redis.Redis] = None
        self._state_store: Optional[StateStore] = None
        self._scheduler: Optional[Scheduler] = None
        self._cvm = None

        self.setup_routes()

    async def get_redis(self) -> redis.Redis:
        """Lazy initialization of Redis client."""
        if self._redis_client is None:
            self._redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=False,  # We handle decoding manually
            )
        return self._redis_client

    async def get_state_store(self) -> StateStore:
        """Lazy initialization of StateStore."""
        if self._state_store is None:
            redis_client = await self.get_redis()
            self._state_store = StateStore(redis_client, key_prefix="dreamer")
        return self._state_store

    async def get_scheduler(self) -> Scheduler:
        """Lazy initialization of Scheduler."""
        if self._scheduler is None:
            redis_client = await self.get_redis()
            state_store = await self.get_state_store()
            self._scheduler = Scheduler(redis_client, state_store)
        return self._scheduler

    def get_cvm(self):
        """Lazy initialization of ConversationModel."""
        if self._cvm is None:
            from ....conversation.model import ConversationModel
            self._cvm = ConversationModel.from_config(self.config)
        return self._cvm

    async def cleanup(self):
        """Cleanup Redis connection on shutdown."""
        if self._redis_client is not None:
            await self._redis_client.close()
            self._redis_client = None
            self._state_store = None
            self._scheduler = None

    def setup_routes(self):
        @self.router.post("/pipeline", response_model=StartPipelineResponse)
        async def create_pipeline(
            request: StartPipelineRequest,
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
        ):
            """
            Start a new dreamer pipeline execution.

            Args:
                request: Pipeline configuration with scenario, conversation, and model
                credentials: API key authentication

            Returns:
                StartPipelineResponse with pipeline_id for tracking

            Raises:
                HTTPException: On validation errors or pipeline start failure
            """
            try:
                state_store = await self.get_state_store()
                scheduler = await self.get_scheduler()

                pipeline_id = await start_pipeline(
                    scenario_name=request.scenario_name,
                    conversation_id=request.conversation_id,
                    persona_id=request.persona_id,
                    user_id=request.user_id,
                    config=self.config,
                    model_name=request.model_name,
                    state_store=state_store,
                    scheduler=scheduler,
                    query_text=request.query_text,
                    guidance=request.guidance,
                    mood=request.mood,
                )

                return StartPipelineResponse(
                    status="success",
                    pipeline_id=pipeline_id,
                    message=f"Pipeline {pipeline_id} started successfully",
                )
            except FileNotFoundError as e:
                logger.error(f"Scenario not found: {e}")
                raise HTTPException(status_code=404, detail=f"Scenario not found: {e}")
            except ValueError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.exception(f"Failed to start pipeline: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to start pipeline: {e}")

        @self.router.get("/pipeline/{pipeline_id}", response_model=PipelineStatusResponse)
        async def get_pipeline_status(
            pipeline_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
        ):
            """
            Get the current status of a pipeline.

            Args:
                pipeline_id: Pipeline identifier
                credentials: API key authentication

            Returns:
                PipelineStatusResponse with detailed status information

            Raises:
                HTTPException: If pipeline not found or status retrieval fails
            """
            try:
                state_store = await self.get_state_store()

                # Load state to get scenario name
                state = await state_store.load_state(pipeline_id)
                if state is None:
                    raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")

                # Load scenario
                scenario = load_scenario(state.scenario_name)

                # Get status
                status = await get_status(pipeline_id, state_store, scenario)

                if status is None:
                    raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")

                return PipelineStatusResponse(
                    pipeline_id=status.pipeline_id,
                    scenario_name=status.scenario_name,
                    status=status.status,
                    current_step=status.current_step,
                    completed_steps=status.completed_steps,
                    failed_steps=status.failed_steps,
                    step_errors=status.step_errors,
                    progress_percent=status.progress_percent,
                    created_at=status.created_at,
                    updated_at=status.updated_at,
                )
            except HTTPException:
                raise
            except FileNotFoundError as e:
                logger.error(f"Scenario not found: {e}")
                raise HTTPException(status_code=404, detail=f"Scenario not found: {e}")
            except Exception as e:
                logger.exception(f"Failed to get pipeline status: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get pipeline status: {e}")

        @self.router.post("/pipeline/{pipeline_id}/cancel", response_model=CancelPipelineResponse)
        async def cancel_pipeline_endpoint(
            pipeline_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
        ):
            """
            Cancel a running pipeline.

            Completed steps are preserved. Sets all pending/running steps to failed.

            Args:
                pipeline_id: Pipeline identifier
                credentials: API key authentication

            Returns:
                CancelPipelineResponse confirming cancellation

            Raises:
                HTTPException: If pipeline not found or cancellation fails
            """
            try:
                state_store = await self.get_state_store()

                success = await cancel_pipeline(pipeline_id, state_store)

                if not success:
                    raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")

                return CancelPipelineResponse(
                    status="success",
                    message=f"Pipeline {pipeline_id} cancelled successfully",
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.exception(f"Failed to cancel pipeline: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to cancel pipeline: {e}")

        @self.router.post("/pipeline/{pipeline_id}/resume", response_model=ResumePipelineResponse)
        async def resume_pipeline_endpoint(
            pipeline_id: str,
            force: bool = False,
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
        ):
            """
            Resume a failed or cancelled pipeline from where it stopped.

            Re-enqueues failed steps whose dependencies are satisfied.
            When force=True, also resets stuck RUNNING steps.

            Args:
                pipeline_id: Pipeline identifier
                force: If True, also reset RUNNING steps (for stuck pipelines)
                credentials: API key authentication

            Returns:
                ResumePipelineResponse confirming resume

            Raises:
                HTTPException: If pipeline not found or resume fails
            """
            try:
                state_store = await self.get_state_store()
                scheduler = await self.get_scheduler()
                cvm = self.get_cvm()

                result = await resume_pipeline(pipeline_id, state_store, scheduler, cvm, force=force)

                if not result.found:
                    raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")

                # Build descriptive message
                parts = []
                if result.new_steps_added:
                    parts.append(f"added {len(result.new_steps_added)} new step(s): {', '.join(result.new_steps_added)}")
                if result.steps_enqueued:
                    parts.append(f"enqueued {len(result.steps_enqueued)} step(s): {', '.join(result.steps_enqueued)}")
                if result.steps_reset:
                    parts.append(f"reset {len(result.steps_reset)} step(s)")
                if result.orphaned_steps_cleaned:
                    parts.append(f"cleaned {len(result.orphaned_steps_cleaned)} orphaned step(s): {', '.join(result.orphaned_steps_cleaned)}")

                if parts:
                    message = f"Pipeline {pipeline_id}: {'; '.join(parts)}"
                else:
                    message = f"Pipeline {pipeline_id}: no changes needed"

                return ResumePipelineResponse(
                    status="success",
                    message=message,
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.exception(f"Failed to resume pipeline: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to resume pipeline: {e}")

        @self.router.post("/pipeline/{pipeline_id}/refresh", response_model=ResumePipelineResponse)
        async def refresh_pipeline_endpoint(
            pipeline_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
        ):
            """
            Refresh a complete pipeline by syncing with current scenario.

            Detects scenario changes (new/removed steps), validates documents
            exist for completed steps, and enqueues new work.

            Args:
                pipeline_id: Pipeline identifier
                credentials: API key authentication

            Returns:
                ResumePipelineResponse with details of changes

            Raises:
                HTTPException: If pipeline not found or refresh fails
            """
            try:
                state_store = await self.get_state_store()
                scheduler = await self.get_scheduler()
                cvm = self.get_cvm()

                result = await refresh_pipeline(pipeline_id, state_store, scheduler, cvm)

                if not result.found:
                    raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")

                # Build descriptive message
                parts = []
                if result.new_steps_added:
                    parts.append(f"found {len(result.new_steps_added)} new step(s): {', '.join(result.new_steps_added)}")
                if result.steps_reset:
                    parts.append(f"reset {len(result.steps_reset)} step(s) with missing documents: {', '.join(result.steps_reset)}")
                if result.orphaned_steps_cleaned:
                    parts.append(f"cleaned {len(result.orphaned_steps_cleaned)} orphaned step(s): {', '.join(result.orphaned_steps_cleaned)}")

                if parts:
                    message = f"Pipeline {pipeline_id}: {'; '.join(parts)}. Click Resume to restart."
                else:
                    message = f"Pipeline {pipeline_id}: fully synced, no changes needed"

                return ResumePipelineResponse(
                    status="success",
                    message=message,
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.exception(f"Failed to refresh pipeline: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to refresh pipeline: {e}")

        @self.router.delete("/pipeline/{pipeline_id}", response_model=DeletePipelineResponse)
        async def delete_pipeline_endpoint(
            pipeline_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
        ):
            """
            Delete a completed or failed pipeline.

            Removes all pipeline state from Redis.

            Args:
                pipeline_id: Pipeline identifier
                credentials: API key authentication

            Returns:
                DeletePipelineResponse confirming deletion

            Raises:
                HTTPException: If pipeline not found or deletion fails
            """
            try:
                state_store = await self.get_state_store()

                success = await delete_pipeline(pipeline_id, state_store)

                if not success:
                    raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")

                return DeletePipelineResponse(
                    status="success",
                    message=f"Pipeline {pipeline_id} deleted successfully",
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.exception(f"Failed to delete pipeline: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to delete pipeline: {e}")

        @self.router.get("/pipelines", response_model=ListPipelinesResponse)
        async def list_pipelines_endpoint(
            status: Optional[str] = None,
            limit: int = 100,
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
        ):
            """
            List pipelines, optionally filtered by status.

            Args:
                status: Optional status filter (running, complete, failed, pending)
                limit: Maximum number of pipelines to return
                credentials: API key authentication

            Returns:
                ListPipelinesResponse with list of pipeline statuses

            Raises:
                HTTPException: If listing fails
            """
            try:
                state_store = await self.get_state_store()

                pipelines = await list_pipelines(state_store, status=status, limit=limit)

                # Convert PipelineStatus objects to response DTOs
                pipeline_responses = [
                    PipelineStatusResponse(
                        pipeline_id=p.pipeline_id,
                        scenario_name=p.scenario_name,
                        status=p.status,
                        current_step=p.current_step,
                        completed_steps=p.completed_steps,
                        failed_steps=p.failed_steps,
                        step_errors=p.step_errors,
                        progress_percent=p.progress_percent,
                        created_at=p.created_at,
                        updated_at=p.updated_at,
                    )
                    for p in pipelines
                ]

                return ListPipelinesResponse(
                    status="success",
                    count=len(pipeline_responses),
                    pipelines=pipeline_responses,
                )
            except Exception as e:
                logger.exception(f"Failed to list pipelines: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to list pipelines: {e}")
