# aim_code/parsers/bash_parser.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Bash parser for CODE_RAG using tree-sitter.

Extracts function definitions from Bash/Shell scripts for code indexing.
Bash parsing is intentionally simple - mainly for completeness in multi-language
codebases. It extracts function definitions and the commands they call.
"""

import warnings
from typing import Iterator, Optional

from .base import BaseParser, ExtractedSymbol, ExtractedImport

try:
    from tree_sitter_language_pack import get_language
    from tree_sitter import Parser

    AVAILABLE = True
except ImportError:
    AVAILABLE = False


# Shell builtins to skip when extracting raw_calls
SHELL_BUILTINS = frozenset({
    # I/O
    "echo",
    "printf",
    "read",
    # Navigation/listing
    "cd",
    "ls",
    "pwd",
    "pushd",
    "popd",
    # Variable handling
    "export",
    "local",
    "declare",
    "readonly",
    "unset",
    "set",
    "shift",
    # Sourcing
    "source",
    ".",
    # Testing
    "test",
    "[",
    "[[",
    # Control flow keywords (shouldn't appear as command_name but just in case)
    "if",
    "then",
    "else",
    "elif",
    "fi",
    "for",
    "while",
    "until",
    "do",
    "done",
    "case",
    "esac",
    "in",
    "select",
    # Flow control
    "break",
    "continue",
    "return",
    "exit",
    # Boolean
    "true",
    "false",
    # Execution
    "eval",
    "exec",
    # Common utilities that aren't interesting for call graphs
    "cat",
    "grep",
    "sed",
    "awk",
    "cut",
    "head",
    "tail",
    "sort",
    "uniq",
    "wc",
    "tr",
    "tee",
    "xargs",
    "find",
    "mkdir",
    "rm",
    "cp",
    "mv",
    "touch",
    "chmod",
    "chown",
})


class BashParser(BaseParser):
    """Parser for Bash/Shell scripts with function extraction."""

    def _load_parser(self) -> None:
        """Load Bash tree-sitter parser via language pack."""
        if not AVAILABLE:
            warnings.warn("tree-sitter-language-pack not available")
            return

        try:
            self.language = get_language("bash")
            self.parser = Parser(self.language)
        except Exception as e:
            warnings.warn(f"Failed to load Bash parser: {e}")

    def get_language_name(self) -> str:
        """Get the language name for this parser."""
        return "bash"

    def extract_symbols(
        self, content: str, file_path: str
    ) -> Iterator[ExtractedSymbol]:
        """Extract function definitions from Bash source code.

        Bash has only one symbol type: function definitions.
        Both `function name() { ... }` and `name() { ... }` syntaxes are handled.
        """
        tree = self.parse(content)
        if tree is None:
            return

        yield from self._extract_from_node(tree.root_node, content)

    def _extract_from_node(self, node, content: str) -> Iterator[ExtractedSymbol]:
        """Recursively extract function definitions from AST nodes."""
        if node.type == "function_definition":
            func_name = self._get_function_name(node, content)
            if func_name:
                # Extract raw calls from function body
                raw_calls = self._extract_raw_calls(node, content)

                yield ExtractedSymbol(
                    name=func_name,
                    symbol_type="function",
                    line_start=self._get_node_line(node),
                    line_end=self._get_node_end_line(node),
                    content=self._get_node_text(node, content),
                    parent=None,  # Bash doesn't have classes
                    signature=self._get_function_signature(node, content),
                    raw_calls=raw_calls,
                    docstring=None,  # Bash doesn't have docstrings
                )
        else:
            # Recurse into children
            for child in node.children:
                yield from self._extract_from_node(child, content)

    def _get_function_name(self, node, content: str) -> Optional[str]:
        """Extract function name from function_definition node.

        Handles both:
        - function name() { ... }
        - name() { ... }
        """
        # tree-sitter-bash uses 'name' field for function name
        name_node = node.child_by_field_name("name")
        if name_node:
            return name_node.text.decode()
        return None

    def _get_function_signature(self, node, content: str) -> str:
        """Extract function signature (first line before body).

        Returns something like 'function_name()' or 'function function_name()'.
        """
        text = self._get_node_text(node, content)
        # Get first line up to opening brace
        first_line = text.split("\n")[0]
        # Remove the opening brace if present
        if "{" in first_line:
            first_line = first_line.split("{")[0].strip()
        return first_line

    def _extract_raw_calls(self, node, content: str) -> list[str]:
        """Extract command names called within a function body.

        Looks for command_name nodes and filters out common builtins.
        """
        calls = []

        # Find the compound_statement (body) of the function
        body = node.child_by_field_name("body")
        if not body:
            return calls

        for child in self._walk_tree(body):
            if child.type == "command_name":
                cmd_text = self._get_node_text(child, content)
                # Skip builtins and common utilities
                if cmd_text not in SHELL_BUILTINS:
                    calls.append(cmd_text)

        return list(set(calls))  # Deduplicate

    def extract_imports(self, content: str) -> list[ExtractedImport]:
        """Extract imports from Bash source code.

        Bash doesn't have imports in the Python sense. Source statements
        could be considered imports, but they're typically not useful for
        CODE_RAG purposes since they reference files, not symbols.

        Returns empty list for simplicity.
        """
        return []

    def extract_attribute_types(
        self, content: str, class_name: Optional[str] = None
    ) -> dict[str, str]:
        """Extract attribute types from Bash source code.

        Bash has no type system and no classes, so this returns empty dict.
        """
        return {}
