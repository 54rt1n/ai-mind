# tests/unit/aim_code/test_symbol_extractor.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Unit tests for aim_code.symbol_extractor.

Tests the SymbolExtractor which bridges the parser layer to the graph layer,
converting ExtractedSymbol (with content) to Symbol (without content).
"""

import pytest
from unittest.mock import MagicMock, patch

from aim_code.symbol_extractor import SymbolExtractor
from aim_code.graph.models import Symbol, ParsedFile
from aim_code.parsers.base import ExtractedSymbol, ParsedFile as ParserParsedFile


@pytest.fixture
def mock_parser():
    """Create a mock parser with controllable behavior."""
    parser = MagicMock()
    parser.is_available.return_value = True
    return parser


@pytest.fixture
def sample_extracted_symbols():
    """Sample ExtractedSymbol objects as a parser would produce."""
    return [
        ExtractedSymbol(
            name="MyClass",
            symbol_type="class",
            line_start=10,
            line_end=50,
            content="class MyClass:\n    pass",
            parent=None,
            signature="class MyClass",
            raw_calls=[],
            docstring="A sample class.",
        ),
        ExtractedSymbol(
            name="__init__",
            symbol_type="method",
            line_start=12,
            line_end=20,
            content="def __init__(self):\n    self.x = 1",
            parent="MyClass",
            signature="def __init__(self)",
            raw_calls=[],
            docstring=None,
        ),
        ExtractedSymbol(
            name="process",
            symbol_type="method",
            line_start=22,
            line_end=35,
            content="def process(self, data):\n    return helper(data)",
            parent="MyClass",
            signature="def process(self, data) -> dict",
            raw_calls=["helper"],
            docstring="Process the data.",
        ),
    ]


class TestSymbolExtractorInit:
    """Tests for SymbolExtractor initialization."""

    def test_creates_registry(self):
        """SymbolExtractor should create a ParserRegistry on init."""
        extractor = SymbolExtractor()
        assert extractor.registry is not None

    def test_registry_type(self):
        """Registry should be a ParserRegistry instance."""
        from aim_code.parsers import ParserRegistry

        extractor = SymbolExtractor()
        assert isinstance(extractor.registry, ParserRegistry)


class TestExtractFromFile:
    """Tests for extract_from_file method."""

    def test_no_parser_yields_nothing(self):
        """When no parser available for language, yields nothing."""
        extractor = SymbolExtractor()
        # Use a language that has no parser
        symbols = list(extractor.extract_from_file("x = 1", "cobol", "test.cob"))
        assert symbols == []

    def test_unavailable_parser_yields_nothing(self):
        """When parser exists but is not available, yields nothing."""
        extractor = SymbolExtractor()

        # Mock the registry to return a parser that isn't available
        mock_parser = MagicMock()
        mock_parser.is_available.return_value = False

        with patch.object(extractor.registry, "get_parser", return_value=mock_parser):
            symbols = list(extractor.extract_from_file("x = 1", "python", "test.py"))

        assert symbols == []

    def test_converts_extracted_to_symbol(
        self, mock_parser, sample_extracted_symbols
    ):
        """Should convert ExtractedSymbol to Symbol, dropping content/docstring."""
        mock_parser.extract_symbols.return_value = iter(sample_extracted_symbols)

        extractor = SymbolExtractor()

        with patch.object(extractor.registry, "get_parser", return_value=mock_parser):
            symbols = list(
                extractor.extract_from_file("...", "python", "test.py")
            )

        assert len(symbols) == 3

        # Check first symbol (class)
        cls = symbols[0]
        assert isinstance(cls, Symbol)
        assert cls.name == "MyClass"
        assert cls.symbol_type == "class"
        assert cls.line_start == 10
        assert cls.line_end == 50
        assert cls.parent is None
        assert cls.signature == "class MyClass"
        assert cls.raw_calls == []

        # Check second symbol (method with parent)
        init = symbols[1]
        assert init.name == "__init__"
        assert init.symbol_type == "method"
        assert init.parent == "MyClass"

        # Check third symbol (method with calls)
        process = symbols[2]
        assert process.name == "process"
        assert process.raw_calls == ["helper"]

    def test_symbol_has_no_content_field(
        self, mock_parser, sample_extracted_symbols
    ):
        """Symbol should not have content or docstring fields."""
        mock_parser.extract_symbols.return_value = iter(sample_extracted_symbols)

        extractor = SymbolExtractor()

        with patch.object(extractor.registry, "get_parser", return_value=mock_parser):
            symbols = list(
                extractor.extract_from_file("...", "python", "test.py")
            )

        for sym in symbols:
            assert not hasattr(sym, "content")
            assert not hasattr(sym, "docstring")


class TestParseFile:
    """Tests for parse_file method."""

    def test_no_parser_returns_empty(self):
        """When no parser available, returns empty ParsedFile."""
        extractor = SymbolExtractor()

        result = extractor.parse_file("x = 1", "cobol", "test.cob")

        assert isinstance(result, ParsedFile)
        assert result.imports == {}
        assert result.symbols == []
        assert result.attribute_types == {}

    def test_unavailable_parser_returns_empty(self):
        """When parser unavailable, returns empty ParsedFile."""
        extractor = SymbolExtractor()

        mock_parser = MagicMock()
        mock_parser.is_available.return_value = False

        with patch.object(extractor.registry, "get_parser", return_value=mock_parser):
            result = extractor.parse_file("x = 1", "python", "test.py")

        assert result.imports == {}
        assert result.symbols == []

    def test_returns_parsed_file_with_all_data(
        self, mock_parser, sample_extracted_symbols
    ):
        """Should return ParsedFile with imports, symbols, and attribute_types."""
        # Mock the parser.parse_file to return a ParserParsedFile
        mock_parsed = ParserParsedFile(
            imports={"Helper": "utils.helper", "Optional": "typing"},
            symbols=sample_extracted_symbols,
            attribute_types={"self.x": "int", "self.helper": "Helper"},
        )
        mock_parser.parse_file.return_value = mock_parsed

        extractor = SymbolExtractor()

        with patch.object(extractor.registry, "get_parser", return_value=mock_parser):
            result = extractor.parse_file("...", "python", "test.py")

        assert isinstance(result, ParsedFile)

        # Imports preserved
        assert result.imports == {"Helper": "utils.helper", "Optional": "typing"}

        # Symbols converted
        assert len(result.symbols) == 3
        assert all(isinstance(s, Symbol) for s in result.symbols)
        assert result.symbols[0].name == "MyClass"

        # Attribute types preserved
        assert result.attribute_types == {"self.x": "int", "self.helper": "Helper"}

    def test_symbols_converted_correctly(
        self, mock_parser, sample_extracted_symbols
    ):
        """Symbols in ParsedFile should be graph.Symbol, not parser.ExtractedSymbol."""
        mock_parsed = ParserParsedFile(
            imports={},
            symbols=sample_extracted_symbols,
            attribute_types={},
        )
        mock_parser.parse_file.return_value = mock_parsed

        extractor = SymbolExtractor()

        with patch.object(extractor.registry, "get_parser", return_value=mock_parser):
            result = extractor.parse_file("...", "python", "test.py")

        for sym in result.symbols:
            # Should be graph.Symbol
            assert isinstance(sym, Symbol)
            # Should NOT be ExtractedSymbol
            assert not isinstance(sym, ExtractedSymbol)
            # Should not have content
            assert not hasattr(sym, "content")


class TestIntegrationWithRealParser:
    """Integration tests using actual parsers (when available)."""

    @pytest.fixture
    def extractor(self):
        """Create a real SymbolExtractor."""
        return SymbolExtractor()

    def test_extract_from_typescript(self, extractor):
        """Test extraction from TypeScript code (if parser available)."""
        if not extractor.registry.has_parser("typescript"):
            pytest.skip("TypeScript parser not available")

        content = '''
function greet(name: string): string {
    return `Hello, ${name}!`;
}

class Greeter {
    constructor(private prefix: string) {}

    greet(name: string): string {
        return greet(this.prefix + name);
    }
}
'''
        symbols = list(extractor.extract_from_file(content, "typescript", "test.ts"))

        assert len(symbols) > 0

        function_names = {s.name for s in symbols if s.symbol_type == "function"}
        assert "greet" in function_names

        class_names = {s.name for s in symbols if s.symbol_type == "class"}
        assert "Greeter" in class_names

    def test_parse_file_typescript(self, extractor):
        """Test full parse_file on TypeScript (if parser available)."""
        if not extractor.registry.has_parser("typescript"):
            pytest.skip("TypeScript parser not available")

        content = '''
import { Helper } from './helper';

class Service {
    helper: Helper;

    constructor(h: Helper) {
        this.helper = h;
    }
}
'''
        result = extractor.parse_file(content, "typescript", "test.ts")

        assert "Helper" in result.imports
        assert len(result.symbols) > 0
        # TypeScript parser should extract attribute types
        assert "this.helper" in result.attribute_types
