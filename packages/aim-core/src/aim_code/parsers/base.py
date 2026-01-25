# aim_code/parsers/base.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Base parser interface for CODE_RAG language parsers.

Enhanced from vendor/data/parsers/base.py with symbol extraction support.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass
class ExtractedSymbol:
    """A symbol extracted from source code."""

    name: str
    symbol_type: str  # "class" | "function" | "method"
    line_start: int
    line_end: int
    content: str  # Source code for this symbol
    parent: Optional[str] = None  # For methods: owning class name
    signature: Optional[str] = None  # Function/method signature
    raw_calls: list[str] = field(default_factory=list)  # Raw call targets
    docstring: Optional[str] = None


@dataclass
class ExtractedImport:
    """An import statement extracted from source code."""

    short_name: str  # The name used in code (e.g., "ChatConfig")
    full_module: str  # Full module path (e.g., "aim.config")
    line: int


@dataclass
class ParsedFile:
    """Result of parsing a source file for CODE_RAG."""

    imports: dict[str, str]  # short_name -> full_module
    symbols: list[ExtractedSymbol]
    attribute_types: dict[str, str]  # self.attr -> type name


class BaseParser(ABC):
    """Base class for CODE_RAG language parsers.

    Extends the vendor parser pattern with symbol extraction capabilities
    for building call graphs and indexing code documents.
    """

    def __init__(self):
        """Initialize parser with language-specific tree-sitter."""
        self.parser = None
        self.language = None
        self._load_parser()

    @abstractmethod
    def _load_parser(self) -> None:
        """Load the tree-sitter parser for this language."""
        pass

    def is_available(self) -> bool:
        """Check if parser loaded successfully."""
        return self.parser is not None and self.language is not None

    def get_language_name(self) -> str:
        """Get the language name for this parser."""
        return self.__class__.__name__.replace("Parser", "").lower()

    def parse(self, content: str) -> Optional["tree_sitter.Tree"]:
        """Parse source code and return the AST.

        Args:
            content: Source code as string.

        Returns:
            Tree-sitter Tree or None if parser unavailable.
        """
        if not self.is_available():
            return None
        return self.parser.parse(content.encode())

    @abstractmethod
    def extract_symbols(
        self, content: str, file_path: str
    ) -> Iterator[ExtractedSymbol]:
        """Extract all symbols (classes, functions, methods) from file content.

        Args:
            content: Source code as string.
            file_path: Path to the source file (for context).

        Yields:
            ExtractedSymbol for each symbol found.
        """
        pass

    @abstractmethod
    def extract_imports(self, content: str) -> list[ExtractedImport]:
        """Extract all import statements from file content.

        Args:
            content: Source code as string.

        Returns:
            List of ExtractedImport objects.
        """
        pass

    @abstractmethod
    def extract_attribute_types(
        self, content: str, class_name: Optional[str] = None
    ) -> dict[str, str]:
        """Extract attribute type mappings from class definitions.

        For Python, this extracts types from __init__ and class annotations.
        Maps "self.attr" -> "TypeName".

        Args:
            content: Source code as string.
            class_name: Optional class to focus on.

        Returns:
            Dict mapping attribute names to type names.
        """
        pass

    def parse_file(self, content: str, file_path: str) -> ParsedFile:
        """Parse a source file and extract all CODE_RAG relevant information.

        Args:
            content: Source code as string.
            file_path: Path to the source file.

        Returns:
            ParsedFile with imports, symbols, and attribute types.
        """
        imports_list = self.extract_imports(content)
        imports = {imp.short_name: imp.full_module for imp in imports_list}
        symbols = list(self.extract_symbols(content, file_path))
        attribute_types = self.extract_attribute_types(content)

        return ParsedFile(
            imports=imports, symbols=symbols, attribute_types=attribute_types
        )

    def _get_node_text(self, node, content: str) -> str:
        """Get the text content of a node."""
        return content[node.start_byte : node.end_byte]

    def _get_node_line(self, node) -> int:
        """Get the 1-indexed line number of a node."""
        return node.start_point[0] + 1

    def _get_node_end_line(self, node) -> int:
        """Get the 1-indexed end line number of a node."""
        return node.end_point[0] + 1

    def _walk_tree(self, node) -> Iterator:
        """Walk all nodes in a tree using a generator."""
        yield node
        for child in node.children:
            yield from self._walk_tree(child)
