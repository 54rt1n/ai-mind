# aim_code/symbol_extractor.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Symbol extraction from source files using tree-sitter parsers.

SymbolExtractor bridges the parser layer (ExtractedSymbol with content) and the
graph layer (Symbol without content). It provides two interfaces:

- extract_from_file(): Yields Symbol objects for graph construction
- parse_file(): Returns ParsedFile for full resolution context

The content from ExtractedSymbol is used elsewhere (DOC_SOURCE_CODE indexing)
but is not needed for the call graph itself.
"""

from typing import Iterator

from aim_code.parsers import ParserRegistry
from aim_code.graph.models import Symbol, ParsedFile


class SymbolExtractor:
    """Extract symbols from source files using tree-sitter parsers.

    This class wraps the parser registry and converts the content-rich
    ExtractedSymbol objects into the leaner Symbol objects used for
    call graph construction.
    """

    def __init__(self):
        self.registry = ParserRegistry()

    def extract_from_file(
        self, content: str, language: str, file_path: str
    ) -> Iterator[Symbol]:
        """Extract symbols from file content.

        Converts ExtractedSymbol (from parser) to Symbol (for graph).
        The content field is intentionally dropped - it's stored separately
        in DOC_SOURCE_CODE documents.

        Args:
            content: Source code as string.
            language: Language name (e.g., 'python', 'typescript').
            file_path: Path to source file (for context in parser).

        Yields:
            Symbol objects suitable for call graph construction.
        """
        parser = self.registry.get_parser(language)
        if not parser or not parser.is_available():
            return

        for extracted in parser.extract_symbols(content, file_path):
            yield Symbol(
                name=extracted.name,
                symbol_type=extracted.symbol_type,
                line_start=extracted.line_start,
                line_end=extracted.line_end,
                parent=extracted.parent,
                signature=extracted.signature,
                raw_calls=extracted.raw_calls,
            )

    def parse_file(self, content: str, language: str, file_path: str) -> ParsedFile:
        """Parse a file and return full ParsedFile for graph construction.

        Returns ParsedFile containing:
        - imports: Mapping of short names to full module paths
        - symbols: List of Symbol objects
        - attribute_types: Mapping of self.attr to type names

        This provides all the context needed for call resolution in pass 2
        of graph building.

        Args:
            content: Source code as string.
            language: Language name (e.g., 'python', 'typescript').
            file_path: Path to source file.

        Returns:
            ParsedFile with imports, symbols, and attribute types.
            Returns empty ParsedFile if parser unavailable.
        """
        parser = self.registry.get_parser(language)
        if not parser or not parser.is_available():
            return ParsedFile(imports={}, symbols=[], attribute_types={})

        parsed = parser.parse_file(content, file_path)

        # Convert ExtractedSymbol to Symbol (drop content, docstring)
        symbols = [
            Symbol(
                name=s.name,
                symbol_type=s.symbol_type,
                line_start=s.line_start,
                line_end=s.line_end,
                parent=s.parent,
                signature=s.signature,
                raw_calls=s.raw_calls,
            )
            for s in parsed.symbols
        ]

        return ParsedFile(
            imports=parsed.imports,
            symbols=symbols,
            attribute_types=parsed.attribute_types,
        )
