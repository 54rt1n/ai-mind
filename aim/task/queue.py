# aim/task/queue.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import os
import json
import time
from typing import Dict, Any

from ..config import ChatConfig


class TaskQueue:
    def __init__(self, config: ChatConfig):
        self.queue_dir = config.queue_dir
        self.queue_complete_dir = config.queue_complete_dir

    @classmethod
    def from_config(cls, config: ChatConfig):
        if not os.path.exists(config.queue_dir):
            os.makedirs(config.queue_dir)
        if not os.path.exists(config.queue_complete_dir):
            os.makedirs(config.queue_complete_dir)
        return cls(config)

    def enqueue(self, task: Dict[str, Any]) -> None:
        task_id = str(int(time.time() * 1000))  # Use timestamp as ID
        file_path = os.path.join(self.queue_dir, f"{task_id}.json")
        with open(file_path, 'w') as f:
            json.dump(task, f)

    def dequeue(self) -> Dict[str, Any]:
        files = os.listdir(self.queue_dir)
        if not files:
            return None
        oldest_file = min(files, key=lambda f: os.path.getctime(os.path.join(self.queue_dir, f)))
        file_path = os.path.join(self.queue_dir, oldest_file)
        with open(file_path, 'r') as f:
            task = json.load(f)
        os.remove(file_path)
        return task, file_path

    def complete_task(self, task: Dict[str, Any], file_path: str) -> None:
        task["task_status"] = "complete"
        task["task_time"] = int(time.time())
        new_file_path = os.path.join(self.queue_complete_dir, os.path.basename(file_path))
        with open(new_file_path, 'w') as f:
            json.dump(task, f)
