# aim_code/graph/mermaid.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Mermaid diagram generation from call graph edges.

Produces graph TD format diagrams for visualization in consciousness blocks.
"""

from typing import Set, Tuple

from .models import SymbolRef


def generate_mermaid(
    edges: Set[Tuple[SymbolRef, SymbolRef]], max_edges: int = 50
) -> str:
    """Generate mermaid diagram from call graph edges.

    Args:
        edges: Set of (caller_ref, callee_ref) tuples
        max_edges: Limit to prevent huge diagrams

    Returns:
        Mermaid graph definition string suitable for rendering.
    """
    lines = ["graph TD"]
    external_nodes: set[str] = set()

    # Sort and limit edges
    edge_list = sorted(edges)[:max_edges]

    for caller_ref, callee_ref in edge_list:
        caller_file, caller_symbol, caller_line = caller_ref
        callee_file, callee_symbol, callee_line = callee_ref

        # Create unique IDs (sanitize for mermaid)
        caller_id = _sanitize_id(f"{caller_file}_{caller_symbol}")
        callee_id = _sanitize_id(f"{callee_file}_{callee_symbol}")

        # Display: symbol:line (or just symbol for externals)
        caller_display = (
            f"{caller_symbol}:{caller_line}" if caller_line >= 0 else caller_symbol
        )
        callee_display = (
            f"{callee_symbol}:{callee_line}" if callee_line >= 0 else callee_symbol
        )

        # Track external nodes for styling
        if callee_line == -1:
            external_nodes.add(callee_id)

        lines.append(
            f'    {caller_id}["{caller_display}"] --> {callee_id}["{callee_display}"]'
        )

    # Style external nodes differently
    for ext_id in sorted(external_nodes):
        lines.append(f"    style {ext_id} fill:#ccc,stroke:#999,stroke-dasharray: 5 5")

    if len(edges) > max_edges:
        lines.append(f'    note["... and {len(edges) - max_edges} more edges"]')

    return "\n".join(lines)


def _sanitize_id(s: str) -> str:
    """Sanitize string for mermaid node ID.

    Replaces characters that are invalid in mermaid IDs.

    Args:
        s: Raw string to sanitize

    Returns:
        Sanitized string safe for use as mermaid node ID.
    """
    return s.replace("/", "_").replace(".", "_").replace("-", "_").replace(":", "_")
