# aim_code/__init__.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
CODE_RAG: Code-aware retrieval augmented generation for AI-Mind.

This package provides code-focused consciousness building for the blip agent,
including:
- XMLCodeTurnStrategy: Fresh strategy implementation for code-focused prompts
- CodeGraph: Bidirectional call graph with height/depth traversal
- Focus tool: Explicit focus on files/line ranges with call graph context
- Language parsers: Tree-sitter based parsing for symbol extraction
"""

from aim.constants import DOC_SOURCE_CODE, DOC_SPEC
from aim_code.documents import SourceDoc, SourceDocMetadata, SpecDoc
from aim_code.strategy import FocusRequest, XMLCodeTurnStrategy
from aim_code.tools import FocusTool
from aim_code.graph import CodeGraph, SymbolRef, generate_mermaid

__all__ = [
    # Constants
    "DOC_SOURCE_CODE",
    "DOC_SPEC",
    # Documents
    "SourceDoc",
    "SourceDocMetadata",
    "SpecDoc",
    # Strategy
    "FocusRequest",
    "XMLCodeTurnStrategy",
    # Tools
    "FocusTool",
    # Graph
    "CodeGraph",
    "SymbolRef",
    "generate_mermaid",
]
