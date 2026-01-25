# aim_code/graph/resolver.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Import and call target resolution.

Resolves raw call strings (e.g., "self.cvm.query") to SymbolRefs
using imports, attribute types, module registry, and symbol table.
"""

from typing import Optional

from .models import SymbolRef
from .module_registry import ModuleRegistry
from .symbol_table import SymbolTable


class ImportResolver:
    """Resolves call targets using file's imports, module registry, and symbol table.

    Handles several call patterns:
    - self.method() - method on same class, same file
    - self.attr.method() - method on typed attribute
    - ImportedClass.method() - method on imported class
    - bare_function() - function in same file or builtin
    """

    def __init__(
        self,
        imports: dict[str, str],
        attribute_types: dict[str, str],
        symbol_table: SymbolTable,
        module_registry: ModuleRegistry,
        current_file: str,
    ):
        """Initialize resolver with context for a specific file.

        Args:
            imports: short_name -> full_module (e.g., {"ChatConfig": "aim.config"})
            attribute_types: self.attr -> type (e.g., {"self.cvm": "ConversationModel"})
            symbol_table: Global symbol table for lookup
            module_registry: Global module -> file mapping
            current_file: Path to the file being processed
        """
        self.imports = imports
        self.attribute_types = attribute_types
        self.symbol_table = symbol_table
        self.module_registry = module_registry
        self.current_file = current_file

    def resolve(
        self, raw_call: str, parent_class: Optional[str] = None
    ) -> Optional[SymbolRef]:
        """Resolve a raw call string to a SymbolRef.

        Args:
            raw_call: Call target as extracted from AST (e.g., "get_env",
                      "ChatConfig.from_env", "self.helper", "self.cvm.query")
            parent_class: Name of enclosing class (for self.* resolution)

        Returns:
            SymbolRef if resolved, None if cannot resolve. External refs
            (packages outside the indexed codebase) return refs with line=-1.

        Examples:
            "get_env"              -> lookup in same file
            "ChatConfig.from_env"  -> use imports to find file
            "self.helper"          -> method on same class
            "self.cvm.query"       -> use attribute_types to find type
        """
        parts = raw_call.split(".")

        # Case 1: self.method() - same class, same file
        if parts[0] == "self" and len(parts) == 2:
            method_name = parts[1]
            qualified = f"{parent_class}.{method_name}" if parent_class else method_name
            return self.symbol_table.lookup_qualified(self.current_file, qualified)

        # Case 2: self.attr.method() - need attribute type
        if parts[0] == "self" and len(parts) == 3:
            attr_key = f"self.{parts[1]}"
            type_name = self.attribute_types.get(attr_key)
            if type_name and type_name in self.imports:
                target_module = self.imports[type_name]
                target_file = self.module_registry.get_file_path(target_module)
                if target_file:
                    qualified = f"{type_name}.{parts[2]}"
                    return self.symbol_table.lookup_qualified(target_file, qualified)
            return None  # Can't resolve

        # Case 3: ImportedClass.method()
        if parts[0] in self.imports:
            target_module = self.imports[parts[0]]
            target_file = self.module_registry.get_file_path(target_module)
            if target_file:
                qualified = ".".join(parts)
                return self.symbol_table.lookup_qualified(target_file, qualified)
            # External module - return external ref
            return (f"package:{target_module}", ".".join(parts), -1)

        # Case 4: bare function() - same file
        if len(parts) == 1:
            ref = self.symbol_table.lookup_qualified(self.current_file, parts[0])
            if ref:
                return ref
            # Could be builtin - return external ref
            return (f"package:builtins", parts[0], -1)

        # Unknown pattern - return external ref with best guess
        return (f"package:unknown", raw_call, -1)
