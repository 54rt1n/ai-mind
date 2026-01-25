# aim_code/tools/__init__.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Code-focused tools for CODE_RAG.

Provides the focus tool for explicit file/line range targeting with call graph context.
"""

from .focus import FocusTool

__all__ = [
    "FocusTool",
]
