# aim/app/mud/worker/__init__.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Worker module for MUD agent processing.

This module provides the MUDAgentWorker class and supporting utilities,
refactored from the original monolithic worker.py (2060 lines) into a
clean module structure.

Module structure:
- main.py: Worker class, main loop, lifecycle
- events.py: Event draining and settling
- turns.py: Turn processing (decision + response phases)
- llm.py: LLM interaction with retry logic
- actions.py: Action emission to Redis
- profile.py: Agent profile management in Redis
- utils.py: Helper functions and CLI entry points
"""

from .main import MUDAgentWorker, AbortRequestedException
from .utils import run_worker, main, parse_args

__all__ = [
    "MUDAgentWorker",
    "AbortRequestedException",
    "run_worker",
    "main",
    "parse_args",
]
