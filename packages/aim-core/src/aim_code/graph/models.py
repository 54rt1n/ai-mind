# aim_code/graph/models.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Data models for CODE_RAG call graph.

These models represent the graph structure for call relationships.
Symbol differs from parsers.ExtractedSymbol in that it doesn't store content
(content lives in indexed DOC_SOURCE_CODE documents), making it lighter
for graph traversal and storage.
"""

from dataclasses import dataclass, field
from typing import Optional


# Symbol reference: (file_path, symbol_name, line_start)
# - Internal refs: ("packages/aim-core/src/aim/config.py", "ChatConfig.from_env", 192)
# - External refs: ("package:json", "loads", -1) where line=-1 marks untraversable
SymbolRef = tuple[str, str, int]


@dataclass
class Symbol:
    """Extracted symbol for call graph construction.

    Unlike ExtractedSymbol from parsers, this does not include the actual source
    content - that's stored separately in DOC_SOURCE_CODE documents. This keeps
    the graph data structure lightweight for traversal and persistence.
    """

    name: str  # "from_env"
    symbol_type: str  # "function" | "method" | "class"
    line_start: int
    line_end: int
    parent: Optional[str] = None  # For methods: "ChatConfig"
    signature: Optional[str] = None
    raw_calls: list[str] = field(default_factory=list)  # ["get_env", "cls"]


@dataclass
class ParsedFile:
    """Result of parsing a source file for call graph construction.

    Contains the import mappings needed for call resolution and the symbols
    with their raw (unresolved) call targets. The resolver uses this data
    in pass 2 to build the actual call graph edges.
    """

    imports: dict[str, str]  # short_name -> full_module: {"ChatConfig": "aim.config"}
    symbols: list[Symbol]
    attribute_types: dict[str, str]  # self.attr -> type: {"self.cvm": "ConversationModel"}
