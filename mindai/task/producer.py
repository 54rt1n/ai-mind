# mindai/task/producer.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import time
from typing import Dict, Any

from .queue import TaskQueue


class Producer:
    def __init__(self, queue: TaskQueue):
        self.queue = queue

    def create_task(self, pipeline_type: str, config: Dict[str, Any]) -> None:
        task = {
            "pipeline_type": pipeline_type,
            "config": config,
            "task_status": "pending",
            "task_time": int(time.time())
        }
        self.queue.enqueue(task)
