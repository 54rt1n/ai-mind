# aim_code/parsers/python_parser.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Python parser for CODE_RAG using tree-sitter.

Extracts symbols, imports, calls, and attribute types from Python source code.
"""

import warnings
from typing import Iterator, Optional

from .base import BaseParser, ExtractedSymbol, ExtractedImport

try:
    import tree_sitter_python
    from tree_sitter import Language, Parser

    AVAILABLE = True
except ImportError:
    AVAILABLE = False


class PythonParser(BaseParser):
    """Parser for Python code with symbol extraction."""

    def _load_parser(self) -> None:
        """Load Python tree-sitter parser."""
        if not AVAILABLE:
            warnings.warn("tree-sitter-python not available")
            return

        try:
            self.language = Language(tree_sitter_python.language())
            self.parser = Parser(self.language)
        except Exception as e:
            warnings.warn(f"Failed to load Python parser: {e}")

    def extract_symbols(
        self, content: str, file_path: str
    ) -> Iterator[ExtractedSymbol]:
        """Extract all symbols from Python source code.

        Yields class definitions and function/method definitions.
        """
        tree = self.parse(content)
        if tree is None:
            return

        yield from self._extract_from_node(tree.root_node, content, parent_class=None)

    def _extract_from_node(
        self, node, content: str, parent_class: Optional[str] = None
    ) -> Iterator[ExtractedSymbol]:
        """Recursively extract symbols from AST nodes."""
        if node.type == "class_definition":
            class_name = self._get_class_name(node)
            if class_name:
                # Extract docstring
                docstring = self._extract_docstring(node, content)

                yield ExtractedSymbol(
                    name=class_name,
                    symbol_type="class",
                    line_start=self._get_node_line(node),
                    line_end=self._get_node_end_line(node),
                    content=self._get_node_text(node, content),
                    parent=None,
                    signature=self._get_class_signature(node, content),
                    raw_calls=[],  # Classes don't have calls at definition level
                    docstring=docstring,
                )

                # Recurse into class body for methods
                body = node.child_by_field_name("body")
                if body:
                    for child in body.children:
                        yield from self._extract_from_node(
                            child, content, parent_class=class_name
                        )

        elif node.type == "function_definition":
            func_name = self._get_function_name(node)
            if func_name:
                # Determine if method or function
                symbol_type = "method" if parent_class else "function"

                # Skip private/dunder methods except __init__
                if parent_class and func_name.startswith("_") and func_name != "__init__":
                    # Still include but mark as internal
                    pass

                # Extract docstring
                docstring = self._extract_docstring(node, content)

                # Extract raw calls from function body
                raw_calls = self._extract_raw_calls(node, content)

                yield ExtractedSymbol(
                    name=func_name,
                    symbol_type=symbol_type,
                    line_start=self._get_node_line(node),
                    line_end=self._get_node_end_line(node),
                    content=self._get_node_text(node, content),
                    parent=parent_class,
                    signature=self._get_function_signature(node, content),
                    raw_calls=raw_calls,
                    docstring=docstring,
                )

        else:
            # Recurse into other nodes (module level)
            for child in node.children:
                yield from self._extract_from_node(child, content, parent_class)

    def _get_class_name(self, node) -> Optional[str]:
        """Extract class name from class_definition node."""
        name_node = node.child_by_field_name("name")
        if name_node:
            return name_node.text.decode()
        return None

    def _get_function_name(self, node) -> Optional[str]:
        """Extract function name from function_definition node."""
        name_node = node.child_by_field_name("name")
        if name_node:
            return name_node.text.decode()
        return None

    def _get_class_signature(self, node, content: str) -> str:
        """Extract class signature (class Name(bases):)."""
        # Get text up to the colon
        text = self._get_node_text(node, content)
        colon_idx = text.find(":")
        if colon_idx > 0:
            return text[:colon_idx].strip()
        return text.split("\n")[0].strip()

    def _get_function_signature(self, node, content: str) -> str:
        """Extract function signature (def name(params) -> return:)."""
        # Build signature from parts
        parts = []

        # Check for decorators
        for child in node.children:
            if child.type == "decorator":
                parts.append(self._get_node_text(child, content))

        # Get def line
        name_node = node.child_by_field_name("name")
        params_node = node.child_by_field_name("parameters")
        return_node = node.child_by_field_name("return_type")

        sig = "def "
        if name_node:
            sig += name_node.text.decode()
        if params_node:
            sig += self._get_node_text(params_node, content)
        if return_node:
            sig += " -> " + self._get_node_text(return_node, content)

        if parts:
            return "\n".join(parts) + "\n" + sig
        return sig

    def _extract_docstring(self, node, content: str) -> Optional[str]:
        """Extract docstring from class or function definition."""
        body = node.child_by_field_name("body")
        if body and body.children:
            first_stmt = body.children[0]
            if first_stmt.type == "expression_statement":
                expr = first_stmt.children[0] if first_stmt.children else None
                if expr and expr.type == "string":
                    docstring = self._get_node_text(expr, content)
                    # Remove quotes
                    if docstring.startswith('"""') or docstring.startswith("'''"):
                        return docstring[3:-3].strip()
                    elif docstring.startswith('"') or docstring.startswith("'"):
                        return docstring[1:-1].strip()
        return None

    def _extract_raw_calls(self, node, content: str) -> list[str]:
        """Extract raw call targets from function body.

        These are unresolved - just the text of what's being called.
        Resolution happens later using imports and symbol table.
        """
        calls = []
        body = node.child_by_field_name("body")
        if not body:
            return calls

        for child in self._walk_tree(body):
            if child.type == "call":
                func_node = child.child_by_field_name("function")
                if func_node:
                    call_text = self._get_node_text(func_node, content)
                    # Skip common builtins that aren't useful for call graphs
                    if call_text not in (
                        "print",
                        "len",
                        "str",
                        "int",
                        "float",
                        "bool",
                        "list",
                        "dict",
                        "set",
                        "tuple",
                        "range",
                        "enumerate",
                        "zip",
                        "map",
                        "filter",
                        "sorted",
                        "reversed",
                        "type",
                        "isinstance",
                        "hasattr",
                        "getattr",
                        "setattr",
                        "super",
                    ):
                        calls.append(call_text)

        return list(set(calls))  # Deduplicate

    def extract_imports(self, content: str) -> list[ExtractedImport]:
        """Extract all import statements from Python source code."""
        tree = self.parse(content)
        if tree is None:
            return []

        imports = []

        for node in self._walk_tree(tree.root_node):
            if node.type == "import_statement":
                # import foo, bar
                for child in node.children:
                    if child.type == "dotted_name":
                        module = child.text.decode()
                        imports.append(
                            ExtractedImport(
                                short_name=module.split(".")[-1],
                                full_module=module,
                                line=self._get_node_line(node),
                            )
                        )
                    elif child.type == "aliased_import":
                        # import foo as f
                        name_node = child.child_by_field_name("name")
                        alias_node = child.child_by_field_name("alias")
                        if name_node:
                            module = name_node.text.decode()
                            alias = (
                                alias_node.text.decode()
                                if alias_node
                                else module.split(".")[-1]
                            )
                            imports.append(
                                ExtractedImport(
                                    short_name=alias,
                                    full_module=module,
                                    line=self._get_node_line(node),
                                )
                            )

            elif node.type == "import_from_statement":
                # from foo import bar, baz
                module_node = node.child_by_field_name("module_name")
                module = module_node.text.decode() if module_node else ""

                for child in node.children:
                    if child.type == "dotted_name" and child != module_node:
                        name = child.text.decode()
                        imports.append(
                            ExtractedImport(
                                short_name=name,
                                full_module=f"{module}.{name}" if module else name,
                                line=self._get_node_line(node),
                            )
                        )
                    elif child.type == "aliased_import":
                        name_node = child.child_by_field_name("name")
                        alias_node = child.child_by_field_name("alias")
                        if name_node:
                            name = name_node.text.decode()
                            alias = alias_node.text.decode() if alias_node else name
                            imports.append(
                                ExtractedImport(
                                    short_name=alias,
                                    full_module=f"{module}.{name}" if module else name,
                                    line=self._get_node_line(node),
                                )
                            )

        return imports

    def extract_attribute_types(
        self, content: str, class_name: Optional[str] = None
    ) -> dict[str, str]:
        """Extract attribute type mappings from Python class definitions.

        Extracts types from:
        1. Class-level annotations (for dataclasses/Pydantic)
        2. __init__ parameter types assigned to self
        3. Direct instantiation in __init__ (self.attr = Type())
        """
        tree = self.parse(content)
        if tree is None:
            return {}

        types = {}

        for node in self._walk_tree(tree.root_node):
            if node.type == "class_definition":
                current_class = self._get_class_name(node)
                if class_name and current_class != class_name:
                    continue

                # Get __init__ method
                init_node = self._find_init_method(node)
                param_types = {}
                if init_node:
                    param_types = self._extract_param_types(init_node, content)

                # Extract from class body
                body = node.child_by_field_name("body")
                if body:
                    for child in body.children:
                        # Class-level annotations
                        if child.type == "expression_statement":
                            expr = child.children[0] if child.children else None
                            if expr and expr.type == "assignment":
                                self._extract_assignment_type(
                                    expr, content, types, param_types, is_init=False
                                )

                        # Check __init__ assignments
                        if child.type == "function_definition":
                            func_name = self._get_function_name(child)
                            if func_name == "__init__":
                                self._extract_init_types(
                                    child, content, types, param_types
                                )

        return types

    def _find_init_method(self, class_node):
        """Find __init__ method in class."""
        body = class_node.child_by_field_name("body")
        if not body:
            return None

        for child in body.children:
            if child.type == "function_definition":
                name = self._get_function_name(child)
                if name == "__init__":
                    return child
        return None

    def _extract_param_types(self, func_node, content: str) -> dict[str, str]:
        """Extract parameter types from function signature."""
        param_types = {}
        params = func_node.child_by_field_name("parameters")
        if not params:
            return param_types

        for param in params.children:
            if param.type == "typed_parameter":
                name_node = None
                type_node = None
                for child in param.children:
                    if child.type == "identifier":
                        name_node = child
                    elif child.type == "type":
                        type_node = child

                if name_node and type_node:
                    param_name = name_node.text.decode()
                    type_text = self._get_node_text(type_node, content)
                    base_type = self._extract_base_type(type_text)
                    if base_type and param_name != "self":
                        param_types[param_name] = base_type

        return param_types

    def _extract_init_types(
        self, init_node, content: str, types: dict, param_types: dict
    ):
        """Extract attribute types from __init__ method."""
        body = init_node.child_by_field_name("body")
        if not body:
            return

        for node in self._walk_tree(body):
            if node.type in ("assignment", "augmented_assignment"):
                self._extract_assignment_type(
                    node, content, types, param_types, is_init=True
                )

    def _extract_assignment_type(
        self, node, content: str, types: dict, param_types: dict, is_init: bool
    ):
        """Extract type from an assignment statement."""
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")

        if not left or not right:
            return

        left_text = self._get_node_text(left, content)

        # Only process self.attr assignments
        if not left_text.startswith("self."):
            return

        attr_name = left_text  # Keep full "self.attr"

        # Check if RHS is a typed parameter
        if right.type == "identifier":
            param_name = right.text.decode()
            if param_name in param_types:
                types[attr_name] = param_types[param_name]
                return

        # Check if RHS is a call (instantiation)
        if right.type == "call":
            func = right.child_by_field_name("function")
            if func:
                if func.type == "identifier":
                    # self.attr = Type(...)
                    types[attr_name] = func.text.decode()
                elif func.type == "attribute":
                    # self.attr = Type.factory(...)
                    obj = func.child_by_field_name("object")
                    if obj and obj.type == "identifier":
                        types[attr_name] = obj.text.decode()

    def _extract_base_type(self, type_text: str) -> Optional[str]:
        """Extract base type name from type annotation.

        Handles Optional[Type], List[Type], etc.
        """
        text = type_text.strip()

        # Strip common wrappers
        for wrapper in ("Optional[", "List[", "Dict[", "Set[", "Tuple["):
            if text.startswith(wrapper):
                text = text[len(wrapper) : -1]
                # For Dict[K, V], take V
                if "," in text:
                    text = text.split(",")[-1].strip()
                break

        # Handle Union - take first non-None
        if text.startswith("Union["):
            inner = text[6:-1]
            for part in inner.split(","):
                part = part.strip()
                if part != "None":
                    text = part
                    break

        # Return base identifier
        return text.split("[")[0].split(",")[0].strip() or None
