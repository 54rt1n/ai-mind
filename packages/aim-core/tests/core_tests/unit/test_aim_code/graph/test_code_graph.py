# tests/core_tests/unit/aim_code/graph/test_code_graph.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for CodeGraph - bidirectional call graph with traversal.

Tests verify graph construction, persistence, and neighborhood queries.
"""

import json
import pytest
from pathlib import Path


class TestCodeGraphConstruction:
    """Tests for CodeGraph construction and edge management."""

    def test_empty_graph_initialization(self):
        """New CodeGraph should have empty calls and callers."""
        from aim_code.graph import CodeGraph

        graph = CodeGraph()
        assert len(graph.calls) == 0
        assert len(graph.callers) == 0

    def test_add_edge_creates_bidirectional_relationship(self):
        """add_edge should update both calls and callers."""
        from aim_code.graph import CodeGraph

        graph = CodeGraph()
        caller = ("file.py", "func_a", 10)
        callee = ("file.py", "func_b", 20)

        graph.add_edge(caller, callee)

        assert callee in graph.calls[caller]
        assert caller in graph.callers[callee]

    def test_add_multiple_edges_from_same_caller(self):
        """A symbol can call multiple other symbols."""
        from aim_code.graph import CodeGraph

        graph = CodeGraph()
        caller = ("file.py", "main", 1)
        callee1 = ("file.py", "helper1", 10)
        callee2 = ("file.py", "helper2", 20)

        graph.add_edge(caller, callee1)
        graph.add_edge(caller, callee2)

        assert len(graph.calls[caller]) == 2
        assert callee1 in graph.calls[caller]
        assert callee2 in graph.calls[caller]

    def test_add_multiple_edges_to_same_callee(self):
        """A symbol can be called by multiple other symbols."""
        from aim_code.graph import CodeGraph

        graph = CodeGraph()
        caller1 = ("file.py", "test1", 1)
        caller2 = ("file.py", "test2", 10)
        callee = ("file.py", "helper", 50)

        graph.add_edge(caller1, callee)
        graph.add_edge(caller2, callee)

        assert len(graph.callers[callee]) == 2
        assert caller1 in graph.callers[callee]
        assert caller2 in graph.callers[callee]


class TestCodeGraphLookup:
    """Tests for get_callees and get_callers."""

    def test_get_callees_returns_empty_set_for_unknown_ref(self):
        """get_callees should return empty set for refs not in graph."""
        from aim_code.graph import CodeGraph

        graph = CodeGraph()
        unknown = ("file.py", "unknown", 999)

        result = graph.get_callees(unknown)
        assert result == set()

    def test_get_callers_returns_empty_set_for_unknown_ref(self):
        """get_callers should return empty set for refs not in graph."""
        from aim_code.graph import CodeGraph

        graph = CodeGraph()
        unknown = ("file.py", "unknown", 999)

        result = graph.get_callers(unknown)
        assert result == set()

    def test_get_callees_returns_correct_refs(self):
        """get_callees should return all symbols a ref calls."""
        from aim_code.graph import CodeGraph

        graph = CodeGraph()
        caller = ("file.py", "main", 1)
        callee1 = ("file.py", "func1", 10)
        callee2 = ("file.py", "func2", 20)
        graph.add_edge(caller, callee1)
        graph.add_edge(caller, callee2)

        result = graph.get_callees(caller)
        assert result == {callee1, callee2}

    def test_get_callers_returns_correct_refs(self):
        """get_callers should return all symbols that call a ref."""
        from aim_code.graph import CodeGraph

        graph = CodeGraph()
        caller1 = ("file.py", "test1", 1)
        caller2 = ("file.py", "test2", 10)
        callee = ("file.py", "shared_util", 50)
        graph.add_edge(caller1, callee)
        graph.add_edge(caller2, callee)

        result = graph.get_callers(callee)
        assert result == {caller1, caller2}


class TestCodeGraphExternalRefs:
    """Tests for external reference handling."""

    def test_is_external_true_for_negative_line(self):
        """is_external should return True for line_start == -1."""
        from aim_code.graph import CodeGraph

        graph = CodeGraph()
        external_ref = ("package:json", "loads", -1)

        assert graph.is_external(external_ref) is True

    def test_is_external_false_for_positive_line(self):
        """is_external should return False for line_start >= 0."""
        from aim_code.graph import CodeGraph

        graph = CodeGraph()
        internal_ref = ("file.py", "func", 10)

        assert graph.is_external(internal_ref) is False

    def test_is_external_false_for_line_zero(self):
        """is_external should return False for line_start == 0."""
        from aim_code.graph import CodeGraph

        graph = CodeGraph()
        ref_at_zero = ("file.py", "module_level", 0)

        assert graph.is_external(ref_at_zero) is False


class TestCodeGraphPersistence:
    """Tests for save/load to edges.json."""

    def test_save_creates_edges_file(self, tmp_path):
        """save should create edges.json in the given directory."""
        from aim_code.graph import CodeGraph

        graph = CodeGraph()
        graph.add_edge(("a.py", "f1", 1), ("a.py", "f2", 10))

        graph.save(tmp_path)

        assert (tmp_path / "edges.json").exists()

    def test_save_creates_directory_if_missing(self, tmp_path):
        """save should create the directory if it doesn't exist."""
        from aim_code.graph import CodeGraph

        graph = CodeGraph()
        graph.add_edge(("a.py", "f1", 1), ("a.py", "f2", 10))
        nested_path = tmp_path / "nested" / "graph"

        graph.save(nested_path)

        assert (nested_path / "edges.json").exists()

    def test_save_format_is_valid_json(self, tmp_path):
        """saved edges.json should be valid JSON with expected structure."""
        from aim_code.graph import CodeGraph

        graph = CodeGraph()
        graph.add_edge(("a.py", "f1", 1), ("a.py", "f2", 10))
        graph.save(tmp_path)

        with open(tmp_path / "edges.json") as f:
            data = json.load(f)

        assert "edges" in data
        assert len(data["edges"]) == 1
        assert data["edges"][0] == [["a.py", "f1", 1], ["a.py", "f2", 10]]

    def test_load_returns_empty_graph_if_file_missing(self, tmp_path):
        """load should return empty graph if edges.json doesn't exist."""
        from aim_code.graph import CodeGraph

        graph = CodeGraph.load(tmp_path)

        assert len(graph.calls) == 0
        assert len(graph.callers) == 0

    def test_load_restores_graph_correctly(self, tmp_path):
        """load should restore graph with same edges as saved."""
        from aim_code.graph import CodeGraph

        original = CodeGraph()
        caller = ("src/main.py", "process", 42)
        callee = ("src/utils.py", "helper", 15)
        original.add_edge(caller, callee)
        original.save(tmp_path)

        loaded = CodeGraph.load(tmp_path)

        assert callee in loaded.calls[caller]
        assert caller in loaded.callers[callee]

    def test_round_trip_preserves_external_refs(self, tmp_path):
        """save/load should preserve external refs with line=-1."""
        from aim_code.graph import CodeGraph

        original = CodeGraph()
        caller = ("src/main.py", "process", 42)
        external = ("package:json", "loads", -1)
        original.add_edge(caller, external)
        original.save(tmp_path)

        loaded = CodeGraph.load(tmp_path)

        assert external in loaded.calls[caller]
        assert loaded.is_external(external)


class TestCodeGraphNeighborhood:
    """Tests for get_neighborhood traversal."""

    @pytest.fixture
    def chain_graph(self):
        """Create a linear call chain: a -> b -> c -> d."""
        from aim_code.graph import CodeGraph

        graph = CodeGraph()
        graph.add_edge(("f.py", "a", 1), ("f.py", "b", 10))
        graph.add_edge(("f.py", "b", 10), ("f.py", "c", 20))
        graph.add_edge(("f.py", "c", 20), ("f.py", "d", 30))
        return graph

    def test_neighborhood_depth_one(self, chain_graph):
        """depth=1 should return direct callees only."""
        ref_b = ("f.py", "b", 10)

        edges = chain_graph.get_neighborhood([ref_b], height=0, depth=1)

        # b -> c
        assert (ref_b, ("f.py", "c", 20)) in edges
        # Should not include c -> d
        assert (("f.py", "c", 20), ("f.py", "d", 30)) not in edges

    def test_neighborhood_depth_two(self, chain_graph):
        """depth=2 should return callees and their callees."""
        ref_b = ("f.py", "b", 10)

        edges = chain_graph.get_neighborhood([ref_b], height=0, depth=2)

        # b -> c and c -> d
        assert (ref_b, ("f.py", "c", 20)) in edges
        assert (("f.py", "c", 20), ("f.py", "d", 30)) in edges

    def test_neighborhood_height_one(self, chain_graph):
        """height=1 should return direct callers only."""
        ref_c = ("f.py", "c", 20)

        edges = chain_graph.get_neighborhood([ref_c], height=1, depth=0)

        # b -> c
        assert (("f.py", "b", 10), ref_c) in edges
        # Should not include a -> b
        assert (("f.py", "a", 1), ("f.py", "b", 10)) not in edges

    def test_neighborhood_height_two(self, chain_graph):
        """height=2 should return callers and their callers."""
        ref_c = ("f.py", "c", 20)

        edges = chain_graph.get_neighborhood([ref_c], height=2, depth=0)

        # b -> c and a -> b
        assert (("f.py", "b", 10), ref_c) in edges
        assert (("f.py", "a", 1), ("f.py", "b", 10)) in edges

    def test_neighborhood_both_directions(self, chain_graph):
        """height=1, depth=1 should traverse both directions."""
        ref_b = ("f.py", "b", 10)

        edges = chain_graph.get_neighborhood([ref_b], height=1, depth=1)

        # Upward: a -> b
        assert (("f.py", "a", 1), ref_b) in edges
        # Downward: b -> c
        assert (ref_b, ("f.py", "c", 20)) in edges

    def test_neighborhood_stops_at_external_refs(self):
        """Traversal should not continue into external refs."""
        from aim_code.graph import CodeGraph

        graph = CodeGraph()
        internal = ("f.py", "func", 10)
        external = ("package:json", "loads", -1)
        beyond = ("package:json", "dumps", -1)
        graph.add_edge(internal, external)
        graph.add_edge(external, beyond)

        edges = graph.get_neighborhood([internal], height=0, depth=2)

        # Should include internal -> external
        assert (internal, external) in edges
        # Should NOT include external -> beyond (external is untraversable)
        assert (external, beyond) not in edges

    def test_neighborhood_multiple_start_symbols(self):
        """get_neighborhood should work with multiple starting symbols."""
        from aim_code.graph import CodeGraph

        graph = CodeGraph()
        a = ("f.py", "a", 1)
        b = ("f.py", "b", 10)
        c = ("f.py", "c", 20)
        graph.add_edge(a, c)
        graph.add_edge(b, c)

        edges = graph.get_neighborhood([a, b], height=0, depth=1)

        assert (a, c) in edges
        assert (b, c) in edges

    def test_neighborhood_empty_for_zero_levels(self):
        """height=0, depth=0 should return empty set."""
        from aim_code.graph import CodeGraph

        graph = CodeGraph()
        graph.add_edge(("f.py", "a", 1), ("f.py", "b", 10))

        edges = graph.get_neighborhood([("f.py", "a", 1)], height=0, depth=0)

        assert edges == set()

    def test_neighborhood_handles_cycles(self):
        """Traversal should handle cycles without infinite loop."""
        from aim_code.graph import CodeGraph

        graph = CodeGraph()
        a = ("f.py", "a", 1)
        b = ("f.py", "b", 10)
        graph.add_edge(a, b)
        graph.add_edge(b, a)  # Cycle!

        # Should complete without hanging
        edges = graph.get_neighborhood([a], height=5, depth=5)

        assert (a, b) in edges
        assert (b, a) in edges
