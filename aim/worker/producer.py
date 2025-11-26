# aim/worker/producer.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from bullmq import Queue
from bullmq.queue import QueueBaseOptions, Job
import logging

from ..config import ChatConfig

logger = logging.getLogger(__name__)


class Producer:
    def __init__(self, queue_name: str, options : QueueBaseOptions = {}):
        self.queue = Queue(name=queue_name, opts=options)
        self.token = "token"

    async def retry_failed(self):
        jobs = await self.queue.getJobCounts()

        for j in jobs.items():
            logger.info(f"Job: {j}")

        failed = await self.queue.getJobs(["failed"])
        if failed is not None:
            logger.info(f"Failed jobs: {', '.join([str(j.id) for j in failed])}")
            for job in failed:
                logger.info(f"Failed job: {job}")
                logger.info(job.data)
                logger.info(job.stacktrace)
                await job.retry()
        else:
            logger.info("No failed jobs found.")

    async def retry_job_id(self, job_id: int) -> Job | None:
        job = await self.queue.getJobs(types=["failed"])
        if job is not None and len(job) > 0:
            logger.info(f"Failed Jobs: {', '.join([str(j.id) for j in job])}")
            myjob = next(filter(lambda x: x.id == job_id or x.id == str(job_id), job), None)
            if myjob is not None:
                logger.info(f"Failed job: {myjob}")
                logger.info(myjob.data)
                logger.info(myjob.stacktrace)
                await myjob.updateProgress(0)
                await myjob.retry()
                return myjob
            else:
                logger.info(f"No failed job {job_id} found.")
                return None
        else:
            logger.info(f"No failed job {job_id} found.")
            return None

    async def add_task(self, pipeline_type: str, config: dict) -> Job:
        task = {
            "pipeline_type": pipeline_type,
            "config": config
        }
        logger.debug(f"Adding task: {task}")
        job = await self.queue.add("pipeline_task", task, opts={"removeOnComplete": True})
        logger.info(f"Added job id: {job.id}")
        return job

    async def delete_task(self, task_id: int):
        job = await self.queue.getJobs(types=["waiting", "active", "completed", "failed"])
        if job is not None and len(job) > 0:
            logger.info(f"Current Jobs: {', '.join([str(j.id) for j in job])}")
            myjob = next(filter(lambda x: x.id == task_id or x.id == str(task_id), job), None)
            if myjob is not None:
                logger.info(f"Selected job: {myjob}")
                logger.info(myjob.data)
                logger.info(myjob.stacktrace)
                try:
                    await myjob.remove()
                except Exception as e:
                    if "locked" in str(e).lower():
                        logger.warning(f"Job {task_id} is locked, clearing lock and forcing removal")
                        # Delete the lock key directly from Redis
                        lock_key = f"{self.queue.prefix}:{self.queue.name}:{myjob.id}:lock"
                        await self.queue.client.delete(lock_key)
                        await myjob.remove()
                    else:
                        raise
                return myjob
            else:
                logger.info(f"No available job {task_id} found.")
                return None
        else:
            logger.info(f"No available job {task_id} found.")
            return None

    async def close(self):
        await self.queue.close()

    @classmethod
    def from_config(cls, config: ChatConfig) -> 'Producer':
        # Set redis connection and prefix namespace
        options : QueueBaseOptions = {}
        return Producer(queue_name=config.queue_name, options=options)