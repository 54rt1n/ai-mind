# tests/core_tests/unit/aim_code/graph/test_resolver.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for ImportResolver - call target resolution."""

import pytest


class TestResolverSelfCalls:
    """Tests for self.method() resolution."""

    @pytest.fixture
    def resolver_context(self):
        """Create resolver with basic context for testing."""
        from aim_code.graph import ImportResolver, SymbolTable, ModuleRegistry

        symbol_table = SymbolTable()
        symbol_table.add("current.py", "helper", parent="MyClass", line_start=10)
        symbol_table.add("current.py", "other_method", parent="MyClass", line_start=20)
        symbol_table.add("current.py", "standalone", parent=None, line_start=30)

        module_registry = ModuleRegistry()

        resolver = ImportResolver(
            imports={},
            attribute_types={},
            symbol_table=symbol_table,
            module_registry=module_registry,
            current_file="current.py",
        )
        return resolver

    def test_resolve_self_method_same_class(self, resolver_context):
        """self.method() should resolve to method in same class."""
        result = resolver_context.resolve("self.helper", parent_class="MyClass")

        assert result == ("current.py", "MyClass.helper", 10)

    def test_resolve_self_method_without_parent_class(self, resolver_context):
        """self.method() without parent_class should use bare name."""
        result = resolver_context.resolve("self.standalone", parent_class=None)

        # Falls back to looking for just "standalone" in same file
        assert result == ("current.py", "standalone", 30)

    def test_resolve_self_method_not_found(self, resolver_context):
        """self.method() for unknown method should return None."""
        result = resolver_context.resolve("self.nonexistent", parent_class="MyClass")

        assert result is None


class TestResolverAttributeCalls:
    """Tests for self.attr.method() resolution."""

    @pytest.fixture
    def resolver_with_types(self):
        """Create resolver with attribute types and imports."""
        from aim_code.graph import ImportResolver, SymbolTable, ModuleRegistry

        symbol_table = SymbolTable()
        symbol_table.add("other.py", "query", parent="DataStore", line_start=50)

        module_registry = ModuleRegistry()
        module_registry.add("app.data", "other.py")

        resolver = ImportResolver(
            imports={"DataStore": "app.data"},
            attribute_types={"self.store": "DataStore"},
            symbol_table=symbol_table,
            module_registry=module_registry,
            current_file="current.py",
        )
        return resolver

    def test_resolve_self_attr_method(self, resolver_with_types):
        """self.attr.method() should resolve using attribute type."""
        result = resolver_with_types.resolve(
            "self.store.query", parent_class="MyClass"
        )

        assert result == ("other.py", "DataStore.query", 50)

    def test_resolve_self_attr_method_unknown_attr(self, resolver_with_types):
        """self.unknown_attr.method() should return None."""
        result = resolver_with_types.resolve(
            "self.unknown.method", parent_class="MyClass"
        )

        assert result is None

    def test_resolve_self_attr_method_unknown_type(self):
        """self.attr.method() with unknown type in imports returns None."""
        from aim_code.graph import ImportResolver, SymbolTable, ModuleRegistry

        resolver = ImportResolver(
            imports={},  # Type not in imports
            attribute_types={"self.store": "UnknownType"},
            symbol_table=SymbolTable(),
            module_registry=ModuleRegistry(),
            current_file="current.py",
        )

        result = resolver.resolve("self.store.method", parent_class="MyClass")

        assert result is None


class TestResolverImportedCalls:
    """Tests for ImportedClass.method() resolution."""

    @pytest.fixture
    def resolver_with_imports(self):
        """Create resolver with imports and module registry."""
        from aim_code.graph import ImportResolver, SymbolTable, ModuleRegistry

        symbol_table = SymbolTable()
        symbol_table.add("utils.py", "from_env", parent="Config", line_start=100)

        module_registry = ModuleRegistry()
        module_registry.add("app.config", "utils.py")

        resolver = ImportResolver(
            imports={"Config": "app.config"},
            attribute_types={},
            symbol_table=symbol_table,
            module_registry=module_registry,
            current_file="current.py",
        )
        return resolver

    def test_resolve_imported_class_method(self, resolver_with_imports):
        """ImportedClass.method() should resolve via imports."""
        result = resolver_with_imports.resolve("Config.from_env")

        assert result == ("utils.py", "Config.from_env", 100)

    def test_resolve_imported_external_package(self):
        """Imported class from external package should return external ref."""
        from aim_code.graph import ImportResolver, SymbolTable, ModuleRegistry

        resolver = ImportResolver(
            imports={"json": "json"},  # stdlib import
            attribute_types={},
            symbol_table=SymbolTable(),
            module_registry=ModuleRegistry(),  # Not in registry -> external
            current_file="current.py",
        )

        result = resolver.resolve("json.loads")

        assert result == ("package:json", "json.loads", -1)


class TestResolverBareCalls:
    """Tests for bare function() resolution."""

    def test_resolve_bare_function_same_file(self):
        """bare_function() should resolve in same file."""
        from aim_code.graph import ImportResolver, SymbolTable, ModuleRegistry

        symbol_table = SymbolTable()
        symbol_table.add("current.py", "helper", parent=None, line_start=5)

        resolver = ImportResolver(
            imports={},
            attribute_types={},
            symbol_table=symbol_table,
            module_registry=ModuleRegistry(),
            current_file="current.py",
        )

        result = resolver.resolve("helper")

        assert result == ("current.py", "helper", 5)

    def test_resolve_bare_function_builtin(self):
        """Unknown bare function should resolve to builtins external."""
        from aim_code.graph import ImportResolver, SymbolTable, ModuleRegistry

        resolver = ImportResolver(
            imports={},
            attribute_types={},
            symbol_table=SymbolTable(),
            module_registry=ModuleRegistry(),
            current_file="current.py",
        )

        result = resolver.resolve("print")

        assert result == ("package:builtins", "print", -1)


class TestResolverUnknownPatterns:
    """Tests for unknown call patterns."""

    def test_resolve_unknown_pattern(self):
        """Unknown multi-part call should resolve to package:unknown."""
        from aim_code.graph import ImportResolver, SymbolTable, ModuleRegistry

        resolver = ImportResolver(
            imports={},
            attribute_types={},
            symbol_table=SymbolTable(),
            module_registry=ModuleRegistry(),
            current_file="current.py",
        )

        result = resolver.resolve("some.unknown.chain.call")

        assert result == ("package:unknown", "some.unknown.chain.call", -1)
