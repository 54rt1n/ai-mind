# aim_code/parsers/__init__.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Language parsers for CODE_RAG.

Tree-sitter based parsers adapted from vendor/data/parsers for symbol extraction.
"""

from .base import BaseParser, ExtractedSymbol, ExtractedImport, ParsedFile
from .python_parser import PythonParser
from .registry import ParserRegistry

# Optional parser imports - may not exist yet
try:
    from .typescript_parser import TypeScriptParser
except ImportError:
    TypeScriptParser = None  # type: ignore[misc, assignment]

try:
    from .bash_parser import BashParser
except ImportError:
    BashParser = None  # type: ignore[misc, assignment]


__all__ = [
    # Base types
    "BaseParser",
    "ExtractedSymbol",
    "ExtractedImport",
    "ParsedFile",
    # Registry
    "ParserRegistry",
    # Parsers
    "PythonParser",
    "TypeScriptParser",
    "BashParser",
]
