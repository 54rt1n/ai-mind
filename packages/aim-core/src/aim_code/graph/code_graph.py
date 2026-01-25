# aim_code/graph/code_graph.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Bidirectional call graph with height/depth traversal.

The CodeGraph stores call relationships between symbols and supports
neighborhood queries for consciousness building in XMLCodeTurnStrategy.
"""

import json
from collections import defaultdict
from pathlib import Path

from .models import SymbolRef


class CodeGraph:
    """Bidirectional call graph with height/depth traversal.

    Key: (file_path, symbol, line_start) - matches doc_id for direct CVM lookup.
    Stores both forward (calls) and inverse (callers) for traversal.

    External refs are marked with line_start == -1 and are recorded as
    endpoints but never traversed INTO.
    """

    def __init__(self):
        """Initialize empty graph with bidirectional edge storage."""
        # Full refs as keys for direct lookup
        self.calls: dict[SymbolRef, set[SymbolRef]] = defaultdict(set)
        self.callers: dict[SymbolRef, set[SymbolRef]] = defaultdict(set)

    @classmethod
    def load(cls, graph_path: Path) -> "CodeGraph":
        """Load graph from edges.json file.

        Args:
            graph_path: Directory containing edges.json

        Returns:
            CodeGraph populated from file, or empty graph if file missing.
        """
        graph = cls()
        edges_file = graph_path / "edges.json"

        if not edges_file.exists():
            return graph

        with open(edges_file) as f:
            data = json.load(f)

        for caller_ref, callee_ref in data.get("edges", []):
            # Convert lists back to tuples for use as dict keys
            caller = (caller_ref[0], caller_ref[1], caller_ref[2])
            callee = (callee_ref[0], callee_ref[1], callee_ref[2])
            graph.calls[caller].add(callee)
            graph.callers[callee].add(caller)

        return graph

    def save(self, graph_path: Path) -> None:
        """Save graph to edges.json file.

        Args:
            graph_path: Directory to write edges.json into.
        """
        graph_path.mkdir(parents=True, exist_ok=True)
        edges_file = graph_path / "edges.json"

        edges = []
        for caller, callees in self.calls.items():
            for callee in callees:
                edges.append([list(caller), list(callee)])

        with open(edges_file, "w") as f:
            json.dump({"edges": edges}, f, indent=2)

    def add_edge(self, caller: SymbolRef, callee: SymbolRef) -> None:
        """Add a call relationship.

        Maintains bidirectional consistency: adds to both calls and callers.

        Args:
            caller: Symbol making the call
            callee: Symbol being called
        """
        self.calls[caller].add(callee)
        self.callers[callee].add(caller)

    def get_callees(self, ref: SymbolRef) -> set[SymbolRef]:
        """Get symbols this symbol calls (forward/down).

        Args:
            ref: Symbol to look up

        Returns:
            Set of symbols called by ref, empty set if none.
        """
        return self.calls.get(ref, set())

    def get_callers(self, ref: SymbolRef) -> set[SymbolRef]:
        """Get symbols that call this symbol (inverse/up).

        Args:
            ref: Symbol to look up

        Returns:
            Set of symbols that call ref, empty set if none.
        """
        return self.callers.get(ref, set())

    def is_external(self, ref: SymbolRef) -> bool:
        """Check if ref is external (line_start == -1).

        External refs represent calls to packages outside the indexed codebase.
        They are recorded as endpoints but not traversed into.

        Args:
            ref: Symbol reference to check

        Returns:
            True if external (line_start == -1)
        """
        return ref[2] == -1

    def get_neighborhood(
        self,
        symbols: list[SymbolRef],
        height: int = 1,
        depth: int = 1,
    ) -> set[tuple[SymbolRef, SymbolRef]]:
        """Get call graph neighborhood around symbols.

        Traverses up (callers) and down (callees) from the given symbols
        to build a subgraph of the call relationships.

        Args:
            symbols: Center symbols (file_path, symbol, line_start)
            height: Levels UP (callers of callers of...)
            depth: Levels DOWN (callees of callees of...)

        Returns:
            Set of (caller_ref, callee_ref) edges in the neighborhood.
        """
        edges: set[tuple[SymbolRef, SymbolRef]] = set()

        def traverse_down(ref: SymbolRef, remaining: int) -> None:
            if remaining <= 0:
                return
            for callee in self.get_callees(ref):
                edges.add((ref, callee))
                # Don't traverse INTO external refs (line=-1)
                if not self.is_external(callee):
                    traverse_down(callee, remaining - 1)

        def traverse_up(ref: SymbolRef, remaining: int) -> None:
            if remaining <= 0:
                return
            for caller in self.get_callers(ref):
                edges.add((caller, ref))
                # Don't traverse INTO external refs (line=-1)
                if not self.is_external(caller):
                    traverse_up(caller, remaining - 1)

        for ref in symbols:
            traverse_down(ref, depth)
            traverse_up(ref, height)

        return edges
