# tests/unit/aim_code/graph/test_models.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Unit tests for aim_code.graph.models.

Tests the data models used for call graph construction: Symbol, ParsedFile,
and the SymbolRef type alias.
"""

import pytest

from aim_code.graph.models import Symbol, ParsedFile, SymbolRef


class TestSymbolRef:
    """Tests for SymbolRef type alias."""

    def test_symbolref_is_tuple(self):
        """SymbolRef should be a tuple of (file_path, symbol_name, line_start)."""
        ref: SymbolRef = ("path/to/file.py", "ClassName.method_name", 42)
        assert ref[0] == "path/to/file.py"
        assert ref[1] == "ClassName.method_name"
        assert ref[2] == 42

    def test_symbolref_internal_ref(self):
        """Internal refs have positive line numbers."""
        ref: SymbolRef = ("packages/aim-core/src/aim/config.py", "ChatConfig.from_env", 192)
        assert ref[2] > 0

    def test_symbolref_external_ref(self):
        """External refs have line_start of -1."""
        ref: SymbolRef = ("package:json", "loads", -1)
        assert ref[2] == -1

    def test_symbolref_hashable(self):
        """SymbolRef should be hashable for use in sets and dicts."""
        ref1: SymbolRef = ("file.py", "func", 10)
        ref2: SymbolRef = ("file.py", "func", 10)

        # Can be added to sets
        ref_set = {ref1, ref2}
        assert len(ref_set) == 1

        # Can be dict keys
        ref_dict = {ref1: "value"}
        assert ref_dict[ref2] == "value"


class TestSymbol:
    """Tests for Symbol dataclass."""

    def test_symbol_minimal(self):
        """Symbol can be created with required fields only."""
        sym = Symbol(
            name="my_function",
            symbol_type="function",
            line_start=10,
            line_end=20,
        )
        assert sym.name == "my_function"
        assert sym.symbol_type == "function"
        assert sym.line_start == 10
        assert sym.line_end == 20
        assert sym.parent is None
        assert sym.signature is None
        assert sym.raw_calls == []

    def test_symbol_function(self):
        """Symbol for a standalone function."""
        sym = Symbol(
            name="process_data",
            symbol_type="function",
            line_start=15,
            line_end=30,
            signature="def process_data(data: dict) -> str",
            raw_calls=["validate", "transform", "json.dumps"],
        )
        assert sym.name == "process_data"
        assert sym.symbol_type == "function"
        assert sym.parent is None
        assert "validate" in sym.raw_calls

    def test_symbol_method(self):
        """Symbol for a class method."""
        sym = Symbol(
            name="from_env",
            symbol_type="method",
            line_start=192,
            line_end=196,
            parent="ChatConfig",
            signature="def from_env(cls, dotenv_file: Optional[str] = None) -> ChatConfig",
            raw_calls=["get_env", "cls"],
        )
        assert sym.name == "from_env"
        assert sym.symbol_type == "method"
        assert sym.parent == "ChatConfig"
        assert sym.signature is not None

    def test_symbol_class(self):
        """Symbol for a class definition."""
        sym = Symbol(
            name="ChatConfig",
            symbol_type="class",
            line_start=100,
            line_end=200,
            signature="class ChatConfig(BaseModel)",
            raw_calls=[],  # Classes typically don't have direct calls
        )
        assert sym.name == "ChatConfig"
        assert sym.symbol_type == "class"
        assert sym.parent is None

    def test_symbol_raw_calls_mutable(self):
        """raw_calls default should be a fresh list per instance."""
        sym1 = Symbol(name="a", symbol_type="function", line_start=1, line_end=2)
        sym2 = Symbol(name="b", symbol_type="function", line_start=3, line_end=4)

        sym1.raw_calls.append("call1")

        # sym2's raw_calls should be unaffected
        assert sym2.raw_calls == []


class TestParsedFile:
    """Tests for ParsedFile dataclass."""

    def test_parsed_file_empty(self):
        """ParsedFile can be created with empty collections."""
        pf = ParsedFile(imports={}, symbols=[], attribute_types={})
        assert pf.imports == {}
        assert pf.symbols == []
        assert pf.attribute_types == {}

    def test_parsed_file_with_imports(self):
        """ParsedFile stores import mappings."""
        pf = ParsedFile(
            imports={
                "ChatConfig": "aim.config",
                "Path": "pathlib",
                "Optional": "typing",
            },
            symbols=[],
            attribute_types={},
        )
        assert pf.imports["ChatConfig"] == "aim.config"
        assert "Path" in pf.imports

    def test_parsed_file_with_symbols(self):
        """ParsedFile stores list of Symbol objects."""
        func = Symbol(
            name="helper",
            symbol_type="function",
            line_start=5,
            line_end=10,
        )
        method = Symbol(
            name="process",
            symbol_type="method",
            line_start=20,
            line_end=35,
            parent="MyClass",
        )

        pf = ParsedFile(
            imports={},
            symbols=[func, method],
            attribute_types={},
        )

        assert len(pf.symbols) == 2
        assert pf.symbols[0].name == "helper"
        assert pf.symbols[1].parent == "MyClass"

    def test_parsed_file_with_attribute_types(self):
        """ParsedFile stores attribute type mappings."""
        pf = ParsedFile(
            imports={"ConversationModel": "aim.conversation.model"},
            symbols=[],
            attribute_types={
                "self.cvm": "ConversationModel",
                "self.config": "ChatConfig",
                "self.name": "str",
            },
        )

        assert pf.attribute_types["self.cvm"] == "ConversationModel"
        assert pf.attribute_types["self.config"] == "ChatConfig"

    def test_parsed_file_full_example(self):
        """ParsedFile with realistic content."""
        pf = ParsedFile(
            imports={
                "ChatConfig": "aim.config",
                "ConversationModel": "aim.conversation.model",
                "Optional": "typing",
            },
            symbols=[
                Symbol(
                    name="ChatManager",
                    symbol_type="class",
                    line_start=20,
                    line_end=150,
                    signature="class ChatManager",
                ),
                Symbol(
                    name="__init__",
                    symbol_type="method",
                    line_start=25,
                    line_end=40,
                    parent="ChatManager",
                    signature="def __init__(self, config: ChatConfig)",
                    raw_calls=["ConversationModel"],
                ),
                Symbol(
                    name="query",
                    symbol_type="method",
                    line_start=45,
                    line_end=80,
                    parent="ChatManager",
                    signature="def query(self, text: str) -> list",
                    raw_calls=["self.cvm.search", "self._process"],
                ),
            ],
            attribute_types={
                "self.config": "ChatConfig",
                "self.cvm": "ConversationModel",
            },
        )

        assert len(pf.imports) == 3
        assert len(pf.symbols) == 3
        assert len(pf.attribute_types) == 2

        # Verify we can find methods by parent
        manager_methods = [s for s in pf.symbols if s.parent == "ChatManager"]
        assert len(manager_methods) == 2
