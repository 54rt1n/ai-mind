# aim/refiner/__init__.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Refiner module - Autonomous exploration engine for AI-Mind.

This module provides context-aware exploration when the API is idle,
allowing the persona to autonomously explore knowledge and generate insights.

Architecture:
    Config (YAML) → Paradigm (strategy) → Engine (context)

The Paradigm class is the domain object that encapsulates all paradigm-specific
behavior: prompt building, scenario routing, document gathering, and tool definitions.
"""

from .engine import ExplorationEngine
from .context import ContextGatherer, GatheredContext
from .paradigm import Paradigm

__all__ = [
    "ExplorationEngine",
    "ContextGatherer",
    "GatheredContext",
    "Paradigm",
]
