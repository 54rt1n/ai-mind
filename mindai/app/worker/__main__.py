# mindai/app/worker/__main__.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

# This is the worker for MindAI. It is used to process the messages from the queue.
# The queue is based on BullMQ and Redis.

import asyncio
import logging

from ...config import ChatConfig
from ...worker.consumer import run_consumer

async def main():
    logging.basicConfig(level=logging.INFO)
    config = ChatConfig.from_env()

    # Start the consumer
    await run_consumer(config)


asyncio.run(main())