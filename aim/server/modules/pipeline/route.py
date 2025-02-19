# aim/server/modules/pipeline/route.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from bullmq.job import Job
import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List

from ....config import ChatConfig
from ....worker.producer import Producer

from .dto import PipelineTaskRequest

logger = logging.getLogger(__name__)

class PipelineModule:
    def __init__(self, config: ChatConfig, security: HTTPBearer):
        self.router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])
        self.security = security
        self.config = config
        self.producer = Producer.from_config(config)
        
        self.setup_routes()

    def setup_routes(self):
        @self.router.post("/task")
        async def create_task(
            task: PipelineTaskRequest,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Create a new pipeline task"""
            try:
                job = await self.producer.add_task(task.pipeline_type, task.config.model_dump())
                job_id = job.id if job is not None else -1
                return {
                    "status": "success", 
                    "message": f"{task.pipeline_type.value} pipeline task {job_id} created",
                    "config": task.config.model_dump()
                }
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/task")
        async def list_tasks(
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """List all pipeline tasks"""
            try:
                all_jobs = []
                try:
                    for job_type in ["waiting", "active", "completed", "failed"]:
                        jobs = await self.producer.queue.getJobs([job_type])
                        
                        def job_info(job: Job) -> dict:
                            return {
                                "id": job.id,
                                "job_status": job_type,
                                "progress": job.progress,
                                "name": job.name,
                                "timestamp": job.timestamp,
                                "data": job.data,
                                "finishedOn": job.finishedOn,
                            }
                        all_jobs.extend([job_info(job) for job in jobs])
                        
                    return {
                        "status": "success", 
                        "message": f"{len(all_jobs)} pipeline tasks", 
                        "jobs": all_jobs
                    }
                except ValueError as e:
                    return {
                        "status": "success",
                        "message": f"No pipeline tasks",
                        "jobs": []
                    }
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/task/{task_id}/retry")
        async def retry_task(
            task_id: int,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Retry a failed pipeline task"""
            try:
                job = await self.producer.retry_job_id(task_id)
                if job is not None:
                    return {
                        "status": "success",
                        "message": f"Pipeline task {job.id} retried",
                        "config": job.data
                    }
                return {
                    "status": "error",
                    "message": f"Pipeline task {task_id} not found",
                    "config": {}
                }
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/task/{task_id}/remove")
        async def remove_task(
            task_id: int,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Remove a pipeline task"""
            try:
                job = await self.producer.delete_task(task_id)
                if job is not None:
                    return {
                        "status": "success",
                        "message": f"Pipeline task {job.id} removed",
                        "config": {}
                    }
                return {
                    "status": "error",
                    "message": f"Pipeline task {task_id} not found",
                    "config": {}
                }
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))