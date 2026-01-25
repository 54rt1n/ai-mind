# tests/core_tests/unit/aim_code/graph/test_symbol_table.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for SymbolTable - symbol name to SymbolRef mapping."""


class TestSymbolTableBasics:
    """Tests for basic add/lookup operations."""

    def test_empty_table_lookup_qualified_returns_none(self):
        """lookup_qualified should return None for unknown symbols."""
        from aim_code.graph import SymbolTable

        table = SymbolTable()

        assert table.lookup_qualified("file.py", "unknown") is None

    def test_empty_table_lookup_name_returns_empty(self):
        """lookup_name should return empty list for unknown names."""
        from aim_code.graph import SymbolTable

        table = SymbolTable()

        assert table.lookup_name("unknown") == []

    def test_add_and_lookup_qualified_function(self):
        """Should store and retrieve a function symbol."""
        from aim_code.graph import SymbolTable

        table = SymbolTable()
        table.add("file.py", "my_func", parent=None, line_start=10)

        result = table.lookup_qualified("file.py", "my_func")
        assert result == ("file.py", "my_func", 10)

    def test_add_and_lookup_qualified_method(self):
        """Should store and retrieve a method symbol with qualified name."""
        from aim_code.graph import SymbolTable

        table = SymbolTable()
        table.add("file.py", "process", parent="DataLoader", line_start=42)

        result = table.lookup_qualified("file.py", "DataLoader.process")
        assert result == ("file.py", "DataLoader.process", 42)

    def test_lookup_name_returns_all_matches(self):
        """lookup_name should return all symbols with that name across files."""
        from aim_code.graph import SymbolTable

        table = SymbolTable()
        table.add("file1.py", "helper", parent=None, line_start=10)
        table.add("file2.py", "helper", parent=None, line_start=20)
        table.add("file3.py", "helper", parent="Utils", line_start=30)

        results = table.lookup_name("helper")

        assert len(results) == 3
        assert ("file1.py", "helper", 10) in results
        assert ("file2.py", "helper", 20) in results
        assert ("file3.py", "Utils.helper", 30) in results


class TestSymbolTableQualifiedNames:
    """Tests for qualified name construction."""

    def test_function_has_simple_qualified_name(self):
        """Functions should have their name as qualified name."""
        from aim_code.graph import SymbolTable

        table = SymbolTable()
        table.add("file.py", "process_data", parent=None, line_start=5)

        result = table.lookup_qualified("file.py", "process_data")
        assert result[1] == "process_data"

    def test_method_has_dotted_qualified_name(self):
        """Methods should have Parent.name as qualified name."""
        from aim_code.graph import SymbolTable

        table = SymbolTable()
        table.add("file.py", "run", parent="Engine", line_start=100)

        result = table.lookup_qualified("file.py", "Engine.run")
        assert result[1] == "Engine.run"

    def test_same_name_different_parents(self):
        """Same method name under different classes should be distinct."""
        from aim_code.graph import SymbolTable

        table = SymbolTable()
        table.add("file.py", "process", parent="ClassA", line_start=10)
        table.add("file.py", "process", parent="ClassB", line_start=50)

        result_a = table.lookup_qualified("file.py", "ClassA.process")
        result_b = table.lookup_qualified("file.py", "ClassB.process")

        assert result_a == ("file.py", "ClassA.process", 10)
        assert result_b == ("file.py", "ClassB.process", 50)


class TestSymbolTableFileScoping:
    """Tests for file-scoped lookups."""

    def test_same_symbol_different_files(self):
        """Same qualified name in different files should be separate."""
        from aim_code.graph import SymbolTable

        table = SymbolTable()
        table.add("file1.py", "Config", parent=None, line_start=1)
        table.add("file2.py", "Config", parent=None, line_start=1)

        result1 = table.lookup_qualified("file1.py", "Config")
        result2 = table.lookup_qualified("file2.py", "Config")

        assert result1 == ("file1.py", "Config", 1)
        assert result2 == ("file2.py", "Config", 1)

    def test_lookup_qualified_wrong_file_returns_none(self):
        """lookup_qualified should return None if file doesn't match."""
        from aim_code.graph import SymbolTable

        table = SymbolTable()
        table.add("file1.py", "func", parent=None, line_start=10)

        result = table.lookup_qualified("other_file.py", "func")
        assert result is None
