# aim_code/strategy/__init__.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Code-focused turn strategies for building code consciousness.

Provides XMLCodeTurnStrategy for code agents like blip, with:
- Focused code retrieval (files, line ranges)
- Call graph traversal (height/depth neighborhood)
- Semantic code search
- Module spec integration
"""

from .base import FocusRequest, XMLCodeTurnStrategy, DEFAULT_CONSCIOUSNESS_BUDGET

__all__ = [
    "DEFAULT_CONSCIOUSNESS_BUDGET",
    "FocusRequest",
    "XMLCodeTurnStrategy",
]
