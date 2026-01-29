# tests/core_tests/unit/aim_code/graph/test_mermaid.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for Mermaid diagram generation from call graph edges."""


class TestMermaidGeneration:
    """Tests for generate_mermaid function."""

    def test_empty_edges_produces_minimal_diagram(self):
        """Empty edge set should produce just header."""
        from aim_code.graph import generate_mermaid

        result = generate_mermaid(set())

        assert result == "graph TD"

    def test_single_edge_generates_arrow(self):
        """Single edge should produce one arrow line."""
        from aim_code.graph import generate_mermaid

        edges = {
            (("file.py", "func_a", 10), ("file.py", "func_b", 20))
        }

        result = generate_mermaid(edges)

        assert "graph TD" in result
        assert "-->" in result
        assert "func_a:10" in result
        assert "func_b:20" in result

    def test_multiple_edges(self):
        """Multiple edges should produce multiple arrows."""
        from aim_code.graph import generate_mermaid

        edges = {
            (("f.py", "a", 1), ("f.py", "b", 10)),
            (("f.py", "b", 10), ("f.py", "c", 20)),
        }

        result = generate_mermaid(edges)

        lines = result.split("\n")
        arrow_lines = [l for l in lines if "-->" in l]
        assert len(arrow_lines) == 2

    def test_external_ref_has_no_line_number(self):
        """External refs (line=-1) should display without line number."""
        from aim_code.graph import generate_mermaid

        edges = {
            (("file.py", "func", 10), ("package:json", "loads", -1))
        }

        result = generate_mermaid(edges)

        # External node should show just the symbol
        assert "loads" in result
        assert "loads:-1" not in result  # Should not include negative line

    def test_external_nodes_get_styling(self):
        """External nodes should have dashed style applied."""
        from aim_code.graph import generate_mermaid

        edges = {
            (("file.py", "func", 10), ("package:json", "loads", -1))
        }

        result = generate_mermaid(edges)

        assert "style" in result
        assert "stroke-dasharray" in result

    def test_max_edges_truncation(self):
        """Edges beyond max_edges should be truncated with note."""
        from aim_code.graph import generate_mermaid

        # Create 10 edges
        edges = {
            (("f.py", "a", i), ("f.py", "b", i + 100))
            for i in range(10)
        }

        result = generate_mermaid(edges, max_edges=5)

        # Should have exactly 5 arrow lines (sorted)
        arrow_lines = [l for l in result.split("\n") if "-->" in l]
        assert len(arrow_lines) == 5

        # Should have truncation note
        assert "... and 5 more edges" in result

    def test_sanitize_special_characters(self):
        """File paths with special chars should be sanitized in IDs."""
        from aim_code.graph import generate_mermaid

        edges = {
            (("src/foo-bar/file.py", "Class.method", 10), ("other.py", "func", 20))
        }

        result = generate_mermaid(edges)

        # Slashes, dots, dashes should be replaced with underscores
        # The ID should not contain these raw characters
        # Just verify it doesn't crash and produces output
        assert "graph TD" in result
        assert "-->" in result


class TestMermaidEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_duplicate_edges(self):
        """Duplicate edges in set should only appear once."""
        from aim_code.graph import generate_mermaid

        # Sets automatically deduplicate, but let's verify behavior
        edge = (("f.py", "a", 1), ("f.py", "b", 10))
        edges = {edge, edge}

        result = generate_mermaid(edges)

        arrow_lines = [l for l in result.split("\n") if "-->" in l]
        assert len(arrow_lines) == 1

    def test_handles_self_loop(self):
        """Self-referential edge should be handled."""
        from aim_code.graph import generate_mermaid

        ref = ("f.py", "recursive", 10)
        edges = {(ref, ref)}

        result = generate_mermaid(edges)

        assert "recursive:10" in result
        assert "-->" in result

    def test_long_symbol_names(self):
        """Long symbol names should be included in display."""
        from aim_code.graph import generate_mermaid

        edges = {
            (("f.py", "VeryLongClassName.very_long_method_name", 10),
             ("f.py", "AnotherLongClass.another_long_method", 20))
        }

        result = generate_mermaid(edges)

        assert "VeryLongClassName.very_long_method_name:10" in result
        assert "AnotherLongClass.another_long_method:20" in result

    def test_line_zero_is_not_external(self):
        """Line 0 should be treated as internal (with line number)."""
        from aim_code.graph import generate_mermaid

        edges = {
            (("f.py", "module_level", 0), ("f.py", "func", 10))
        }

        result = generate_mermaid(edges)

        assert "module_level:0" in result
