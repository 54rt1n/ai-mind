# andimud_worker/commands/result.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Command result returned by command execution."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CommandResult:
    """Result returned by command execution.

    Attributes:
        complete: True if command finished (continue loop), False if needs process_turn
        flush_drain: True if command consumed events (clear worker.pending_events)
        saved_event_id: Event ID before draining (for rollback on failure)
        status: "done" or "fail" (only used if complete=True)
        message: Optional status message (for "fail" status)
    """

    complete: bool
    flush_drain: bool = False
    saved_event_id: Optional[str] = None
    status: str = "done"
    message: Optional[str] = None
