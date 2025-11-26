# aim/worker/consumer.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from bullmq import Worker
from bullmq.worker import WorkerOptions, Job
import asyncio
import logging
import signal
import sys

from ..config  import ChatConfig
from ..pipeline.factory import pipeline_factory, BasePipeline

logger = logging.getLogger(__name__)

async def process_pipeline_task(job: Job, job_token: str):
    try:
        pipeline_type = job.data['pipeline_type']
        job_config = job.data['config']

        if job.progress > 0:
            logger.info(f"Pipeline {pipeline_type} already processed.")
            return f"Pipeline {pipeline_type} already processed."

        logger.info(f"Processing {pipeline_type} pipeline with config: {job_config}")

        config = ChatConfig.from_env()
        config.update(**job_config)

        pipeline = BasePipeline.from_config(config, **job_config)
        async def progress_callback(progress: int):
            await job.updateProgress(progress)
        pipeline.progress_callback = progress_callback
        pipeline_func = pipeline_factory(pipeline_type)
        
        # Run the pipeline
        await pipeline_func(self=pipeline, **job_config)

        # Update the job status
        await job.updateProgress(100)
        
        return f"Completed {pipeline_type} pipeline"
    except Exception as e:
        logger.error(f"Error processing {pipeline_type} pipeline: {e}")
        import traceback
        traceback.print_exc()
        await job.moveToFailed(str(e), job_token)

async def run_consumer(config : ChatConfig):
    try:
        worker_options : WorkerOptions = {
            "autorun": True,
            "lockDuration": 21600000,  # 6 hours - pipelines can take a while
        }
        
        shutdown_event = asyncio.Event()
        logger.info("Starting consumer...")

        worker = Worker(name=config.queue_name, processor=process_pipeline_task, opts=worker_options)
        
        def signal_handler(signal, frame):
            logger.info("Signal received, shutting down.")
            shutdown_event.set()
            worker.cancelProcessing()
            sys.exit(0)

        #signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGHUP, signal_handler)


        logger.info("Consumer started successfully.")
        
        await shutdown_event.wait()

        logger.info("Cleaning up worker...")
        await worker.close()
        logger.info("Worker shut down successfully.")
    except Exception as e:
        logger.error("Error in run_consumer:", str(e))
        import traceback
        traceback.print_exc()
        shutdown_event.set()
        await worker.close()
