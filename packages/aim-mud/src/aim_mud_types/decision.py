# aim_mud_types/decision.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Decision types and results for turn processing."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional


class DecisionType(Enum):
    """Decision types from Phase 1 decision processor."""
    MOVE = auto()
    TAKE = auto()
    DROP = auto()
    GIVE = auto()
    SPEAK = auto()
    EMOTE = auto()
    WAIT = auto()
    THINK = auto()          # NEW - triggers ThinkingTurnProcessor
    PLAN = auto()           # NEW - triggers plan creation (future)
    PLAN_UPDATE = auto()    # Existing - plan status update
    CONFUSED = auto()       # Fallback for invalid responses
    AURA_TOOL = auto()      # Dynamic tools from room auras


@dataclass
class DecisionResult:
    """Result from DecisionProcessor."""
    decision_type: DecisionType
    args: dict[str, Any]
    thinking: str
    raw_response: str
    cleaned_response: str
    should_flush: bool = False

    # For AURA_TOOL - store the actual tool name
    aura_tool_name: Optional[str] = None
