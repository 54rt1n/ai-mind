# mindai/task/consumer.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import logging
import time
from typing import Dict, Any
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .queue import TaskQueue

logger = logging.getLogger(__name__)


class Consumer(FileSystemEventHandler):
    def __init__(self, queue: TaskQueue):
        self.queue = queue

    def on_created(self, event):
        if not event.is_directory:
            task, file_path = self.queue.dequeue()
            if task:
                self.process_task(task)
                self.queue.complete_task(task, file_path)

    def process_task(self, task: Dict[str, Any]) -> None:
        pipeline_type = task['pipeline_type']
        config = task['config']
        logger.info(f"Processing {pipeline_type} pipeline with config: {config}")

def run_consumer(queue: TaskQueue):
    consumer = Consumer(queue)
    observer = Observer()
    observer.schedule(consumer, path=queue.queue_dir, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
