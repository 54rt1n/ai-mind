# aim/refiner/__init__.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Refiner module - Autonomous exploration engine for AI-Mind.

This module provides context-aware exploration when the API is idle,
allowing the persona to autonomously explore knowledge and generate insights.
"""

from aim.refiner.engine import ExplorationEngine
from aim.refiner.context import ContextGatherer, GatheredContext
from aim.refiner.paradigm import ExplorationPlan
from aim.refiner.prompts import (
    build_topic_selection_prompt,
    build_validation_prompt,
    build_brainstorm_selection_prompt,
    build_daydream_selection_prompt,
    build_knowledge_selection_prompt,
)

__all__ = [
    "ExplorationEngine",
    "ContextGatherer",
    "GatheredContext",
    "ExplorationPlan",
    "build_topic_selection_prompt",
    "build_validation_prompt",
    "build_brainstorm_selection_prompt",
    "build_daydream_selection_prompt",
    "build_knowledge_selection_prompt",
]
