# aim/refiner/paradigm.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Core dataclasses for the refiner module.

Provides the ExplorationPlan dataclass used by the ExplorationEngine
to trigger pipeline scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExplorationPlan:
    """
    Result of exploration evaluation - describes what to explore.

    Attributes:
        scenario: The scenario to run ('philosopher', 'journaler', 'daydream')
        query_text: The topic or question to explore
        guidance: Optional guidance text for the pipeline
        reasoning: Optional LLM reasoning for why this was chosen
    """

    scenario: str
    query_text: str
    guidance: Optional[str] = None
    reasoning: Optional[str] = None
