# tests/unit/aim_code/parsers/test_typescript_parser.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Unit tests for TypeScript parser.

Tests symbol extraction, import parsing, and attribute type extraction.
"""

import pytest

from aim_code.parsers.typescript_parser import TypeScriptParser, AVAILABLE


@pytest.fixture
def parser() -> TypeScriptParser:
    """Create a TypeScript parser instance."""
    return TypeScriptParser()


@pytest.fixture
def sample_class() -> str:
    """Sample TypeScript class with various constructs."""
    return '''
/**
 * A sample class for testing.
 */
class MyClass extends BaseClass implements IInterface {
    name: string;
    private count: number = 0;

    constructor(public id: string, private config: Config) {
        super();
        this.name = id;
    }

    async fetchData(url: string): Promise<Data> {
        const response = await fetch(url);
        return response.json();
    }

    static create(id: string): MyClass {
        return new MyClass(id, defaultConfig);
    }
}
'''


@pytest.fixture
def sample_functions() -> str:
    """Sample TypeScript functions."""
    return '''
function greet(name: string): string {
    return `Hello, ${name}!`;
}

async function loadData(id: number): Promise<Data> {
    const data = await fetchById(id);
    return processData(data);
}

const multiply = (a: number, b: number): number => {
    return a * b;
}

const add = (a: number, b: number) => a + b;

export const subtract = (a: number, b: number): number => {
    return a - b;
}

export function divide(a: number, b: number): number {
    return a / b;
}
'''


@pytest.fixture
def sample_imports() -> str:
    """Sample TypeScript imports."""
    return '''
import { writable, derived, get } from 'svelte/store';
import { browser } from '$app/environment';
import type { ChatConfig, ChatMessage } from '$lib';
import React from 'react';
import * as lodash from 'lodash';
import { Foo as F, Bar as B } from './utils';
'''


@pytest.fixture
def sample_store() -> str:
    """Sample Svelte store pattern."""
    return '''
import { writable, derived } from 'svelte/store';

function createStore() {
    const { subscribe, set, update } = writable<{ count: number }>({
        count: 0,
    });

    function increment() {
        update(store => ({ ...store, count: store.count + 1 }));
    }

    function decrement() {
        update(store => ({ ...store, count: store.count - 1 }));
    }

    return {
        subscribe,
        increment,
        decrement,
    };
}

export const counterStore = createStore();
'''


@pytest.mark.skipif(not AVAILABLE, reason="tree-sitter-typescript not available")
class TestTypeScriptParserAvailability:
    """Tests for parser availability."""

    def test_parser_loads(self, parser: TypeScriptParser):
        """Parser should load successfully when tree-sitter is available."""
        assert parser.is_available()
        assert parser.parser is not None
        assert parser.language is not None

    def test_get_language_name(self, parser: TypeScriptParser):
        """Parser should report its language name."""
        assert parser.get_language_name() == "typescript"


@pytest.mark.skipif(not AVAILABLE, reason="tree-sitter-typescript not available")
class TestExtractSymbols:
    """Tests for symbol extraction."""

    def test_extract_class(self, parser: TypeScriptParser, sample_class: str):
        """Should extract class declarations."""
        symbols = list(parser.extract_symbols(sample_class, "test.ts"))

        # Find the class symbol
        class_symbols = [s for s in symbols if s.symbol_type == "class"]
        assert len(class_symbols) == 1

        cls = class_symbols[0]
        assert cls.name == "MyClass"
        assert cls.parent is None
        assert "extends BaseClass" in cls.signature
        assert "implements IInterface" in cls.signature
        assert cls.docstring is not None
        assert "sample class" in cls.docstring

    def test_extract_methods(self, parser: TypeScriptParser, sample_class: str):
        """Should extract method definitions from a class."""
        symbols = list(parser.extract_symbols(sample_class, "test.ts"))

        # Find method symbols
        methods = [s for s in symbols if s.symbol_type == "method"]
        method_names = {m.name for m in methods}

        assert "constructor" in method_names
        assert "fetchData" in method_names
        assert "create" in method_names

    def test_method_parent_class(self, parser: TypeScriptParser, sample_class: str):
        """Methods should reference their parent class."""
        symbols = list(parser.extract_symbols(sample_class, "test.ts"))

        methods = [s for s in symbols if s.symbol_type == "method"]
        for method in methods:
            assert method.parent == "MyClass"

    def test_extract_function_declaration(
        self, parser: TypeScriptParser, sample_functions: str
    ):
        """Should extract function declarations."""
        symbols = list(parser.extract_symbols(sample_functions, "test.ts"))

        functions = [s for s in symbols if s.symbol_type == "function"]
        function_names = {f.name for f in functions}

        assert "greet" in function_names
        assert "loadData" in function_names
        assert "divide" in function_names  # exported function

    def test_extract_arrow_functions(
        self, parser: TypeScriptParser, sample_functions: str
    ):
        """Should extract arrow functions assigned to const."""
        symbols = list(parser.extract_symbols(sample_functions, "test.ts"))

        functions = [s for s in symbols if s.symbol_type == "function"]
        function_names = {f.name for f in functions}

        assert "multiply" in function_names
        assert "add" in function_names
        assert "subtract" in function_names  # exported const arrow

    def test_function_signature(
        self, parser: TypeScriptParser, sample_functions: str
    ):
        """Should extract function signatures with types."""
        symbols = list(parser.extract_symbols(sample_functions, "test.ts"))

        greet = next(s for s in symbols if s.name == "greet")
        assert "string" in greet.signature
        assert "name" in greet.signature

    def test_extract_raw_calls(
        self, parser: TypeScriptParser, sample_functions: str
    ):
        """Should extract function calls from bodies."""
        symbols = list(parser.extract_symbols(sample_functions, "test.ts"))

        load_data = next(s for s in symbols if s.name == "loadData")
        assert "fetchById" in load_data.raw_calls
        assert "processData" in load_data.raw_calls

    def test_svelte_store_functions(self, parser: TypeScriptParser, sample_store: str):
        """Should extract functions within factory functions."""
        symbols = list(parser.extract_symbols(sample_store, "test.ts"))

        function_names = {s.name for s in symbols if s.symbol_type == "function"}

        assert "createStore" in function_names
        # Note: inner functions may or may not be extracted depending on
        # how we want to handle nested scopes. Currently we do extract them.

    def test_export_const_assignment(self, parser: TypeScriptParser, sample_store: str):
        """Should extract exported const assignments."""
        symbols = list(parser.extract_symbols(sample_store, "test.ts"))

        # counterStore is const = createStore(), not a function itself
        # so it should NOT be extracted as a function
        function_names = {s.name for s in symbols if s.symbol_type == "function"}
        assert "counterStore" not in function_names


@pytest.mark.skipif(not AVAILABLE, reason="tree-sitter-typescript not available")
class TestExtractImports:
    """Tests for import extraction."""

    def test_named_imports(self, parser: TypeScriptParser, sample_imports: str):
        """Should extract named imports."""
        imports = parser.extract_imports(sample_imports)

        import_map = {i.short_name: i.full_module for i in imports}

        assert "writable" in import_map
        assert import_map["writable"] == "svelte/store.writable"

        assert "derived" in import_map
        assert "get" in import_map

    def test_default_import(self, parser: TypeScriptParser, sample_imports: str):
        """Should extract default imports."""
        imports = parser.extract_imports(sample_imports)

        import_map = {i.short_name: i.full_module for i in imports}

        assert "React" in import_map
        assert import_map["React"] == "react"

    def test_namespace_import(self, parser: TypeScriptParser, sample_imports: str):
        """Should extract namespace imports (import * as name)."""
        imports = parser.extract_imports(sample_imports)

        import_map = {i.short_name: i.full_module for i in imports}

        assert "lodash" in import_map
        assert import_map["lodash"] == "lodash"

    def test_aliased_imports(self, parser: TypeScriptParser, sample_imports: str):
        """Should extract aliased imports with the alias as short_name."""
        imports = parser.extract_imports(sample_imports)

        import_map = {i.short_name: i.full_module for i in imports}

        # Foo as F should be stored with short_name "F"
        assert "F" in import_map
        assert import_map["F"] == "./utils.Foo"

        assert "B" in import_map
        assert import_map["B"] == "./utils.Bar"

    def test_type_imports(self, parser: TypeScriptParser, sample_imports: str):
        """Should extract type-only imports."""
        imports = parser.extract_imports(sample_imports)

        import_map = {i.short_name: i.full_module for i in imports}

        # type imports are still imports
        assert "ChatConfig" in import_map or "ChatMessage" in import_map

    def test_import_line_numbers(self, parser: TypeScriptParser, sample_imports: str):
        """Should track import line numbers."""
        imports = parser.extract_imports(sample_imports)

        # All imports should have valid line numbers (1-indexed)
        for imp in imports:
            assert imp.line >= 1


@pytest.mark.skipif(not AVAILABLE, reason="tree-sitter-typescript not available")
class TestExtractAttributeTypes:
    """Tests for attribute type extraction."""

    def test_class_property_types(self, parser: TypeScriptParser, sample_class: str):
        """Should extract class property types."""
        types = parser.extract_attribute_types(sample_class)

        assert "this.name" in types
        assert types["this.name"] == "string"

        assert "this.count" in types
        assert types["this.count"] == "number"

    def test_constructor_parameter_properties(
        self, parser: TypeScriptParser, sample_class: str
    ):
        """Should extract types from constructor parameter properties."""
        types = parser.extract_attribute_types(sample_class)

        # constructor(public id: string, private config: Config)
        assert "this.id" in types
        assert types["this.id"] == "string"

        assert "this.config" in types
        assert types["this.config"] == "Config"

    def test_filter_by_class_name(self, parser: TypeScriptParser):
        """Should filter attribute extraction by class name."""
        content = '''
class Foo {
    x: number;
}

class Bar {
    y: string;
}
'''
        types = parser.extract_attribute_types(content, class_name="Foo")

        assert "this.x" in types
        assert "this.y" not in types

    def test_generic_type_extraction(self, parser: TypeScriptParser):
        """Should extract base types from generics."""
        content = '''
class Container {
    items: Array<Item>;
    pending: Promise<Result>;
    mapping: Map<string, Value>;
}
'''
        types = parser.extract_attribute_types(content)

        assert types.get("this.items") == "Item"
        assert types.get("this.pending") == "Result"
        assert types.get("this.mapping") == "Value"

    def test_union_type_extraction(self, parser: TypeScriptParser):
        """Should extract first non-null type from unions."""
        content = '''
class NullableProps {
    name: string | null;
    value: undefined | number;
}
'''
        types = parser.extract_attribute_types(content)

        assert types.get("this.name") == "string"
        assert types.get("this.value") == "number"


@pytest.mark.skipif(not AVAILABLE, reason="tree-sitter-typescript not available")
class TestParseFile:
    """Tests for the complete parse_file method."""

    def test_parse_file_returns_parsed_file(
        self, parser: TypeScriptParser, sample_class: str
    ):
        """parse_file should return a ParsedFile with all components."""
        content = sample_class + '''
import { Config } from './config';
'''
        result = parser.parse_file(content, "test.ts")

        assert hasattr(result, "imports")
        assert hasattr(result, "symbols")
        assert hasattr(result, "attribute_types")

        assert "Config" in result.imports
        assert len(result.symbols) > 0
        assert len(result.attribute_types) > 0


@pytest.mark.skipif(not AVAILABLE, reason="tree-sitter-typescript not available")
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_content(self, parser: TypeScriptParser):
        """Should handle empty content gracefully."""
        symbols = list(parser.extract_symbols("", "test.ts"))
        imports = parser.extract_imports("")
        types = parser.extract_attribute_types("")

        assert symbols == []
        assert imports == []
        assert types == {}

    def test_syntax_error_resilience(self, parser: TypeScriptParser):
        """Parser should handle syntax errors gracefully."""
        # Incomplete code with syntax errors
        content = '''
class Broken {
    method( {
        // unclosed
'''
        # Should not raise an exception
        symbols = list(parser.extract_symbols(content, "test.ts"))
        # May or may not extract partial symbols, but should not crash
        assert isinstance(symbols, list)

    def test_jsx_content(self, parser: TypeScriptParser):
        """Should handle JSX/TSX content."""
        content = '''
function Component(props: Props): JSX.Element {
    return <div>Hello</div>;
}
'''
        symbols = list(parser.extract_symbols(content, "test.tsx"))

        # JSX might not parse perfectly with the TypeScript parser,
        # but it should not crash
        assert isinstance(symbols, list)

    def test_decorators_in_class(self, parser: TypeScriptParser):
        """Should handle decorated class members."""
        content = '''
class Controller {
    @Get('/users')
    getUsers(): User[] {
        return this.userService.findAll();
    }
}
'''
        symbols = list(parser.extract_symbols(content, "test.ts"))

        methods = [s for s in symbols if s.symbol_type == "method"]
        assert len(methods) == 1
        assert methods[0].name == "getUsers"
