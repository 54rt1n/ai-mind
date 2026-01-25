# andimud_worker/commands/result.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Command result returned by command execution."""

from dataclasses import dataclass, field
from typing import Optional

from aim_mud_types import TurnRequestStatus


@dataclass
class CommandResult:
    """Result returned by command execution.

    Attributes:
        complete: True if command finished (continue loop), False if needs process_turn
        status: TurnRequestStatus.DONE or TurnRequestStatus.FAIL (only used if complete=True)
        message: Optional status message (for "fail" status)
        plan_guidance: Optional guidance string when plan is active (for user footer)
        turn_id: Optional turn ID override
        emitted_action_ids: List of action_ids emitted by the command (for pending action tracking)
        expects_echo: True if actions expect echo events (False for NON_PUBLISHED actions)
    """

    complete: bool
    status: TurnRequestStatus = TurnRequestStatus.DONE
    message: Optional[str] = None
    plan_guidance: Optional[str] = None
    turn_id: Optional[str] = None
    emitted_action_ids: list[str] = field(default_factory=list)
    expects_echo: bool = True
