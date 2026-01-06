# aim/app/mud/__main__.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""CLI entrypoint for running a MUD agent worker process.

This module provides a command-line interface for starting a MUD agent
worker that consumes events from Redis streams and processes agent turns.

Usage:
    python -m aim.app.mud --agent-id andi --persona-id andi

See aim.app.mud.worker for full documentation.
"""

from .utils import main

if __name__ == "__main__":
    main()
