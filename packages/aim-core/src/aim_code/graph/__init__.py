# aim_code/graph/__init__.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Call graph module for CODE_RAG.

Provides bidirectional call graph traversal with height/depth neighborhood queries.

Components:
- CodeGraph: Bidirectional call graph with save/load and neighborhood traversal
- ModuleRegistry: Maps Python module names to file paths
- SymbolTable: Maps symbol names to SymbolRef tuples
- ImportResolver: Resolves call targets using imports and symbol table
- generate_mermaid: Creates visual diagrams from call graph edges
"""

from .models import ParsedFile, Symbol, SymbolRef
from .code_graph import CodeGraph
from .module_registry import ModuleRegistry
from .symbol_table import SymbolTable
from .resolver import ImportResolver
from .mermaid import generate_mermaid

__all__ = [
    # Core types
    "SymbolRef",
    "Symbol",
    "ParsedFile",
    # Graph
    "CodeGraph",
    # Resolution
    "ModuleRegistry",
    "SymbolTable",
    "ImportResolver",
    # Visualization
    "generate_mermaid",
]
