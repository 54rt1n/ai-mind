# aim_code/parsers/typescript_parser.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
TypeScript parser for CODE_RAG using tree-sitter.

Extracts symbols, imports, calls, and attribute types from TypeScript source code.
Handles both TypeScript (.ts, .tsx) and JavaScript (.js, .jsx) files.
"""

import warnings
from typing import Iterator, Optional

from .base import BaseParser, ExtractedSymbol, ExtractedImport

try:
    import tree_sitter_typescript
    from tree_sitter import Language, Parser

    AVAILABLE = True
except ImportError:
    AVAILABLE = False


class TypeScriptParser(BaseParser):
    """Parser for TypeScript/JavaScript code with symbol extraction."""

    def _load_parser(self) -> None:
        """Load TypeScript tree-sitter parser."""
        if not AVAILABLE:
            warnings.warn("tree-sitter-typescript not available")
            return

        try:
            self.language = Language(tree_sitter_typescript.language_typescript())
            self.parser = Parser(self.language)
        except Exception as e:
            warnings.warn(f"Failed to load TypeScript parser: {e}")

    def extract_symbols(
        self, content: str, file_path: str
    ) -> Iterator[ExtractedSymbol]:
        """Extract all symbols from TypeScript source code.

        Yields class definitions, function declarations, methods, and arrow functions.
        """
        tree = self.parse(content)
        if tree is None:
            return

        yield from self._extract_from_node(tree.root_node, content, parent_class=None)

    def _extract_from_node(
        self, node, content: str, parent_class: Optional[str] = None
    ) -> Iterator[ExtractedSymbol]:
        """Recursively extract symbols from AST nodes."""
        if node.type == "class_declaration":
            class_name = self._get_class_name(node)
            if class_name:
                docstring = self._extract_jsdoc(node, content)

                yield ExtractedSymbol(
                    name=class_name,
                    symbol_type="class",
                    line_start=self._get_node_line(node),
                    line_end=self._get_node_end_line(node),
                    content=self._get_node_text(node, content),
                    parent=None,
                    signature=self._get_class_signature(node, content),
                    raw_calls=[],
                    docstring=docstring,
                )

                # Recurse into class body for methods
                body = node.child_by_field_name("body")
                if body:
                    for child in body.children:
                        yield from self._extract_from_node(
                            child, content, parent_class=class_name
                        )

        elif node.type == "function_declaration":
            func_name = self._get_function_name(node)
            if func_name:
                docstring = self._extract_jsdoc(node, content)
                raw_calls = self._extract_raw_calls(node, content)

                yield ExtractedSymbol(
                    name=func_name,
                    symbol_type="function",
                    line_start=self._get_node_line(node),
                    line_end=self._get_node_end_line(node),
                    content=self._get_node_text(node, content),
                    parent=None,
                    signature=self._get_function_signature(node, content),
                    raw_calls=raw_calls,
                    docstring=docstring,
                )

        elif node.type == "method_definition":
            method_name = self._get_method_name(node)
            if method_name:
                docstring = self._extract_jsdoc(node, content)
                raw_calls = self._extract_raw_calls(node, content)

                yield ExtractedSymbol(
                    name=method_name,
                    symbol_type="method",
                    line_start=self._get_node_line(node),
                    line_end=self._get_node_end_line(node),
                    content=self._get_node_text(node, content),
                    parent=parent_class,
                    signature=self._get_method_signature(node, content),
                    raw_calls=raw_calls,
                    docstring=docstring,
                )

        elif node.type in ("lexical_declaration", "variable_declaration"):
            # Handle: const foo = () => {} or const foo = function() {}
            yield from self._extract_from_variable_declaration(
                node, content, parent_class
            )

        elif node.type == "export_statement":
            # Handle: export function foo() {} or export const foo = ...
            declaration = node.child_by_field_name("declaration")
            if declaration:
                yield from self._extract_from_node(declaration, content, parent_class)
            else:
                # export { ... } or export default - recurse into children
                for child in node.children:
                    if child.type in (
                        "function_declaration",
                        "class_declaration",
                        "lexical_declaration",
                    ):
                        yield from self._extract_from_node(child, content, parent_class)

        else:
            # Recurse into other nodes (module level)
            for child in node.children:
                yield from self._extract_from_node(child, content, parent_class)

    def _extract_from_variable_declaration(
        self, node, content: str, parent_class: Optional[str]
    ) -> Iterator[ExtractedSymbol]:
        """Extract function symbols from variable declarations.

        Handles patterns like:
        - const foo = () => {}
        - const foo = function() {}
        - const foo = async () => {}
        """
        for child in node.children:
            if child.type == "variable_declarator":
                name_node = child.child_by_field_name("name")
                value_node = child.child_by_field_name("value")

                if not name_node or not value_node:
                    continue

                name = name_node.text.decode()

                # Check if the value is a function expression or arrow function
                if value_node.type in ("arrow_function", "function_expression"):
                    docstring = self._extract_jsdoc(node, content)
                    raw_calls = self._extract_raw_calls(value_node, content)

                    yield ExtractedSymbol(
                        name=name,
                        symbol_type="function",
                        line_start=self._get_node_line(node),
                        line_end=self._get_node_end_line(node),
                        content=self._get_node_text(node, content),
                        parent=parent_class,
                        signature=self._get_arrow_function_signature(
                            name, value_node, content
                        ),
                        raw_calls=raw_calls,
                        docstring=docstring,
                    )

    def _get_class_name(self, node) -> Optional[str]:
        """Extract class name from class_declaration node."""
        name_node = node.child_by_field_name("name")
        if name_node:
            return name_node.text.decode()
        return None

    def _get_function_name(self, node) -> Optional[str]:
        """Extract function name from function_declaration node."""
        name_node = node.child_by_field_name("name")
        if name_node:
            return name_node.text.decode()
        return None

    def _get_method_name(self, node) -> Optional[str]:
        """Extract method name from method_definition node."""
        name_node = node.child_by_field_name("name")
        if name_node:
            return name_node.text.decode()
        return None

    def _get_class_signature(self, node, content: str) -> str:
        """Extract class signature (class Name extends Base implements I {)."""
        text = self._get_node_text(node, content)
        # Find the opening brace
        brace_idx = text.find("{")
        if brace_idx > 0:
            return text[:brace_idx].strip()
        return text.split("\n")[0].strip()

    def _get_function_signature(self, node, content: str) -> str:
        """Extract function signature."""
        parts = []

        # Check for async keyword
        for child in node.children:
            if child.type == "async":
                parts.append("async")
                break

        parts.append("function")

        name_node = node.child_by_field_name("name")
        if name_node:
            parts.append(name_node.text.decode())

        params_node = node.child_by_field_name("parameters")
        if params_node:
            parts.append(self._get_node_text(params_node, content))

        return_node = node.child_by_field_name("return_type")
        if return_node:
            parts.append(": " + self._get_node_text(return_node, content))

        return " ".join(parts)

    def _get_method_signature(self, node, content: str) -> str:
        """Extract method signature."""
        parts = []

        # Check for modifiers (static, async, get, set)
        for child in node.children:
            if child.type in ("accessibility_modifier", "static"):
                parts.append(child.text.decode())
            elif child.type == "async":
                parts.append("async")

        name_node = node.child_by_field_name("name")
        if name_node:
            parts.append(name_node.text.decode())

        params_node = node.child_by_field_name("parameters")
        if params_node:
            parts.append(self._get_node_text(params_node, content))

        return_node = node.child_by_field_name("return_type")
        if return_node:
            parts.append(": " + self._get_node_text(return_node, content))

        return " ".join(parts)

    def _get_arrow_function_signature(
        self, name: str, node, content: str
    ) -> str:
        """Extract signature for an arrow function or function expression."""
        parts = []

        # Check for async
        for child in node.children:
            if child.type == "async":
                parts.append("async")
                break

        parts.append(f"const {name} =")

        params_node = node.child_by_field_name("parameters")
        if params_node:
            parts.append(self._get_node_text(params_node, content))
        else:
            # Single parameter without parens: x => x + 1
            param_node = node.child_by_field_name("parameter")
            if param_node:
                parts.append(f"({param_node.text.decode()})")

        return_node = node.child_by_field_name("return_type")
        if return_node:
            parts.append(": " + self._get_node_text(return_node, content))

        parts.append("=>")

        return " ".join(parts)

    def _extract_jsdoc(self, node, content: str) -> Optional[str]:
        """Extract JSDoc comment preceding a node."""
        # Look for a comment node before this node
        # In tree-sitter, comments are usually siblings, not children
        parent = node.parent
        if not parent:
            return None

        # Find this node's index in parent
        node_idx = None
        for i, child in enumerate(parent.children):
            if child == node:
                node_idx = i
                break

        if node_idx is None or node_idx == 0:
            return None

        # Check the previous sibling for a comment
        prev_sibling = parent.children[node_idx - 1]
        if prev_sibling.type == "comment":
            comment_text = self._get_node_text(prev_sibling, content)
            # Check if it's a JSDoc comment (starts with /**)
            if comment_text.startswith("/**"):
                # Strip the comment delimiters and clean up
                text = comment_text[3:-2] if comment_text.endswith("*/") else comment_text[3:]
                # Remove leading asterisks from each line
                lines = []
                for line in text.split("\n"):
                    line = line.strip()
                    if line.startswith("*"):
                        line = line[1:].strip()
                    lines.append(line)
                return "\n".join(lines).strip()

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

        # Common JS/TS builtins to skip
        skip_builtins = {
            "console.log",
            "console.error",
            "console.warn",
            "console.info",
            "console.debug",
            "parseInt",
            "parseFloat",
            "String",
            "Number",
            "Boolean",
            "Array",
            "Object",
            "JSON.parse",
            "JSON.stringify",
            "Math.floor",
            "Math.ceil",
            "Math.round",
            "Math.random",
            "Math.max",
            "Math.min",
            "Date.now",
            "Promise.resolve",
            "Promise.reject",
            "Promise.all",
            "setTimeout",
            "setInterval",
            "clearTimeout",
            "clearInterval",
            "fetch",
            "require",
        }

        for child in self._walk_tree(body):
            if child.type == "call_expression":
                func_node = child.child_by_field_name("function")
                if func_node:
                    call_text = self._get_node_text(func_node, content)
                    if call_text not in skip_builtins:
                        calls.append(call_text)

        return list(set(calls))  # Deduplicate

    def extract_imports(self, content: str) -> list[ExtractedImport]:
        """Extract all import statements from TypeScript source code."""
        tree = self.parse(content)
        if tree is None:
            return []

        imports = []

        for node in self._walk_tree(tree.root_node):
            if node.type == "import_statement":
                imports.extend(self._extract_import_statement(node, content))

        return imports

    def _extract_import_statement(
        self, node, content: str
    ) -> list[ExtractedImport]:
        """Extract imports from an import statement node.

        Handles:
        - import { Foo, Bar } from 'module'
        - import { Foo as F } from 'module'
        - import Foo from 'module'
        - import * as foo from 'module'
        - import 'module' (side-effect import)
        - import type { Foo } from 'module'
        """
        imports = []
        line = self._get_node_line(node)

        # Get the module source (the string after 'from')
        source_node = node.child_by_field_name("source")
        if not source_node:
            return imports

        # Module is the string value without quotes
        module = source_node.text.decode().strip("'\"")

        # Look for import clause components
        for child in node.children:
            if child.type == "import_clause":
                imports.extend(
                    self._extract_from_import_clause(child, module, line, content)
                )

        return imports

    def _extract_from_import_clause(
        self, clause_node, module: str, line: int, content: str
    ) -> list[ExtractedImport]:
        """Extract imports from an import clause."""
        imports = []

        for child in clause_node.children:
            if child.type == "identifier":
                # Default import: import Foo from 'module'
                name = child.text.decode()
                imports.append(
                    ExtractedImport(
                        short_name=name,
                        full_module=module,
                        line=line,
                    )
                )

            elif child.type == "named_imports":
                # Named imports: import { Foo, Bar } from 'module'
                imports.extend(
                    self._extract_named_imports(child, module, line, content)
                )

            elif child.type == "namespace_import":
                # Namespace import: import * as foo from 'module'
                for identifier in child.children:
                    if identifier.type == "identifier":
                        name = identifier.text.decode()
                        imports.append(
                            ExtractedImport(
                                short_name=name,
                                full_module=module,
                                line=line,
                            )
                        )
                        break

        return imports

    def _extract_named_imports(
        self, node, module: str, line: int, content: str
    ) -> list[ExtractedImport]:
        """Extract from named_imports node: { Foo, Bar as B }."""
        imports = []

        for child in node.children:
            if child.type == "import_specifier":
                name_node = child.child_by_field_name("name")
                alias_node = child.child_by_field_name("alias")

                if name_node:
                    original_name = name_node.text.decode()
                    # Use alias if present, otherwise use original name
                    short_name = (
                        alias_node.text.decode() if alias_node else original_name
                    )
                    imports.append(
                        ExtractedImport(
                            short_name=short_name,
                            full_module=f"{module}.{original_name}",
                            line=line,
                        )
                    )

        return imports

    def extract_attribute_types(
        self, content: str, class_name: Optional[str] = None
    ) -> dict[str, str]:
        """Extract attribute type mappings from TypeScript class definitions.

        Extracts types from:
        1. Class property declarations with type annotations
        2. Constructor parameter properties (public/private/protected params)
        """
        tree = self.parse(content)
        if tree is None:
            return {}

        types = {}

        for node in self._walk_tree(tree.root_node):
            if node.type == "class_declaration":
                current_class = self._get_class_name(node)
                if class_name and current_class != class_name:
                    continue

                body = node.child_by_field_name("body")
                if body:
                    self._extract_class_property_types(body, content, types)
                    self._extract_constructor_param_types(body, content, types)

        return types

    def _extract_class_property_types(
        self, class_body, content: str, types: dict
    ) -> None:
        """Extract types from class property declarations.

        Handles: name: Type or name: Type = value
        """
        for child in class_body.children:
            if child.type in ("public_field_definition", "property_definition"):
                name_node = child.child_by_field_name("name")
                type_annotation = child.child_by_field_name("type")

                if name_node and type_annotation:
                    prop_name = name_node.text.decode()
                    # Extract actual type from type_annotation (skip the ':' child)
                    type_text = self._get_type_from_annotation(type_annotation, content)
                    if type_text:
                        base_type = self._extract_base_type(type_text)
                        if base_type:
                            types[f"this.{prop_name}"] = base_type

    def _get_type_from_annotation(self, type_annotation, content: str) -> Optional[str]:
        """Extract the actual type text from a type_annotation node.

        Type annotation nodes have the form: `: TypeName`
        We want just the TypeName part.
        """
        for child in type_annotation.children:
            # Skip the colon punctuation
            if child.type != ":":
                return self._get_node_text(child, content)
        return None

    def _extract_constructor_param_types(
        self, class_body, content: str, types: dict
    ) -> None:
        """Extract types from constructor parameter properties.

        Handles: constructor(public name: Type, private age: Type)
        """
        for child in class_body.children:
            if child.type == "method_definition":
                name_node = child.child_by_field_name("name")
                if name_node and name_node.text.decode() == "constructor":
                    params_node = child.child_by_field_name("parameters")
                    if params_node:
                        self._extract_param_property_types(params_node, content, types)

    def _extract_param_property_types(
        self, params_node, content: str, types: dict
    ) -> None:
        """Extract types from constructor parameters with accessibility modifiers."""
        for param in params_node.children:
            # Look for parameters with public/private/protected
            if param.type in ("required_parameter", "optional_parameter"):
                has_modifier = False
                param_name = None
                param_type = None

                for child in param.children:
                    if child.type == "accessibility_modifier":
                        has_modifier = True
                    elif child.type == "identifier":
                        param_name = child.text.decode()
                    elif child.type == "type_annotation":
                        # Get the actual type from the annotation
                        for type_child in child.children:
                            if type_child.type != ":":
                                param_type = self._get_node_text(type_child, content)
                                break

                if has_modifier and param_name and param_type:
                    base_type = self._extract_base_type(param_type)
                    if base_type:
                        types[f"this.{param_name}"] = base_type

    def _extract_base_type(self, type_text: str) -> Optional[str]:
        """Extract base type name from type annotation.

        Handles Array<Type>, Type[], Optional types, etc.
        """
        text = type_text.strip()

        # Handle union types - take first non-null/undefined
        if "|" in text:
            for part in text.split("|"):
                part = part.strip()
                if part.lower() not in ("null", "undefined"):
                    text = part
                    break

        # Handle Array<Type> -> Type
        if text.startswith("Array<") and text.endswith(">"):
            text = text[6:-1]

        # Handle Type[] -> Type
        if text.endswith("[]"):
            text = text[:-2]

        # Handle Promise<Type> -> Type
        if text.startswith("Promise<") and text.endswith(">"):
            text = text[8:-1]

        # Handle Map<K, V> -> take V
        if text.startswith("Map<") and text.endswith(">"):
            inner = text[4:-1]
            if "," in inner:
                text = inner.split(",")[-1].strip()

        # Handle Set<Type> -> Type
        if text.startswith("Set<") and text.endswith(">"):
            text = text[4:-1]

        # Handle generics: Type<Args> -> Type
        if "<" in text:
            text = text.split("<")[0]

        return text.strip() or None
