# aim_code/graph/symbol_table.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Symbol name to SymbolRef mapping.

Maps symbol names to their full references for call resolution.
Supports both qualified lookup (file + name) and name-only lookup.
"""

from collections import defaultdict
from typing import Optional

from .models import SymbolRef


class SymbolTable:
    """Maps symbol names to their refs across the codebase.

    Provides two lookup strategies:
    1. Qualified: (file_path, name) -> exact SymbolRef
    2. Name-only: name -> list of all SymbolRefs with that name

    The qualified lookup is used for same-file resolution.
    The name-only lookup supports finding symbols across files.
    """

    def __init__(self):
        """Initialize empty symbol table."""
        # name -> list of refs (for overloaded names across files)
        self._by_name: dict[str, list[SymbolRef]] = defaultdict(list)
        # (file_path, qualified_name) -> ref (for exact lookup)
        self._by_qualified: dict[tuple[str, str], SymbolRef] = {}

    def add(self, file_path: str, name: str, parent: Optional[str], line_start: int) -> None:
        """Add a symbol to the table.

        Args:
            file_path: Path to the file containing this symbol
            name: Symbol name (e.g., "from_env")
            parent: Parent class name for methods (e.g., "ChatConfig")
            line_start: Line number where symbol starts
        """
        qualified_name = f"{parent}.{name}" if parent else name
        ref: SymbolRef = (file_path, qualified_name, line_start)

        self._by_name[name].append(ref)
        self._by_qualified[(file_path, qualified_name)] = ref

    def lookup_qualified(self, file_path: str, name: str) -> Optional[SymbolRef]:
        """Exact lookup by file_path + name.

        Args:
            file_path: Path to the file
            name: Qualified symbol name (e.g., "ChatConfig.from_env")

        Returns:
            SymbolRef if found, None otherwise.
        """
        return self._by_qualified.get((file_path, name))

    def lookup_name(self, name: str) -> list[SymbolRef]:
        """Get all refs for a name (may be multiple across files).

        Args:
            name: Symbol name to look up

        Returns:
            List of all SymbolRefs with this name, empty list if none.
        """
        return self._by_name.get(name, [])
