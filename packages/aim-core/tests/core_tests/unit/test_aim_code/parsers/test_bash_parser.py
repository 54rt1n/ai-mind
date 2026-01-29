# tests/core_tests/unit/aim_code/parsers/test_bash_parser.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for BashParser - tree-sitter based Bash/Shell script parser.

Tests verify symbol extraction from Bash function definitions.
These tests require tree_sitter_language_pack to be installed.
"""

import pytest

# Check if tree-sitter is available for conditional skipping
try:
    from tree_sitter_language_pack import get_language
    from tree_sitter import Parser

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

# Skip marker for tests requiring tree-sitter
requires_tree_sitter = pytest.mark.skipif(
    not TREE_SITTER_AVAILABLE,
    reason="tree_sitter_language_pack not installed",
)


# Sample Bash scripts for testing
SIMPLE_FUNCTION = """#!/bin/bash

my_function() {
    echo "Hello, world!"
}
"""

FUNCTION_WITH_FUNCTION_KEYWORD = """#!/bin/bash

function another_func() {
    echo "Using function keyword"
}
"""

FUNCTION_CALLING_OTHER_FUNCTIONS = """#!/bin/bash

helper_func() {
    echo "I'm a helper"
}

main_func() {
    helper_func
    do_something
    echo "Done"
}
"""

MULTIPLE_FUNCTIONS = """#!/bin/bash

setup() {
    echo "Setting up"
}

run_tests() {
    setup
    pytest tests/
}

cleanup() {
    rm -rf /tmp/test
}
"""

FUNCTION_WITH_BUILTINS_ONLY = """#!/bin/bash

builtin_only() {
    cd /tmp
    ls -la
    echo "hello"
    export FOO=bar
    local x=1
}
"""

NESTED_SCRIPT = """#!/bin/bash
# Complex script with various structures

set -e

GLOBAL_VAR="value"

init_config() {
    local config_file="$1"
    if [ -f "$config_file" ]; then
        source "$config_file"
    fi
}

process_item() {
    local item="$1"
    validate_item "$item"
    transform_item "$item"
}

main() {
    init_config "/etc/app.conf"
    for item in "$@"; do
        process_item "$item"
    done
}

main "$@"
"""


class TestBashParserAvailability:
    """Tests for parser availability and loading."""

    @requires_tree_sitter
    def test_parser_loads_successfully(self):
        """BashParser should load tree-sitter-bash via language pack."""
        from aim_code.parsers.bash_parser import BashParser

        parser = BashParser()
        assert parser.is_available()

    def test_language_name(self):
        """get_language_name() should return 'bash'."""
        from aim_code.parsers.bash_parser import BashParser

        parser = BashParser()
        assert parser.get_language_name() == "bash"

    @requires_tree_sitter
    def test_parser_can_parse_bash(self):
        """Parser should successfully parse valid Bash code."""
        from aim_code.parsers.bash_parser import BashParser

        parser = BashParser()
        tree = parser.parse(SIMPLE_FUNCTION)
        assert tree is not None
        assert tree.root_node is not None

    def test_parser_unavailable_returns_none_on_parse(self):
        """When parser unavailable, parse() should return None."""
        from aim_code.parsers.bash_parser import BashParser

        parser = BashParser()
        if not parser.is_available():
            tree = parser.parse(SIMPLE_FUNCTION)
            assert tree is None


@requires_tree_sitter
class TestBashFunctionExtraction:
    """Tests for extracting function definitions."""

    def test_extract_simple_function(self):
        """Should extract function defined with name() syntax."""
        from aim_code.parsers.bash_parser import BashParser

        parser = BashParser()
        symbols = list(parser.extract_symbols(SIMPLE_FUNCTION, "test.sh"))

        assert len(symbols) == 1
        sym = symbols[0]
        assert sym.name == "my_function"
        assert sym.symbol_type == "function"
        assert sym.parent is None
        assert sym.docstring is None

    def test_extract_function_with_keyword(self):
        """Should extract function defined with 'function' keyword."""
        from aim_code.parsers.bash_parser import BashParser

        parser = BashParser()
        symbols = list(parser.extract_symbols(FUNCTION_WITH_FUNCTION_KEYWORD, "test.sh"))

        assert len(symbols) == 1
        sym = symbols[0]
        assert sym.name == "another_func"
        assert sym.symbol_type == "function"

    def test_extract_multiple_functions(self):
        """Should extract all functions from script."""
        from aim_code.parsers.bash_parser import BashParser

        parser = BashParser()
        symbols = list(parser.extract_symbols(MULTIPLE_FUNCTIONS, "test.sh"))

        assert len(symbols) == 3
        names = {s.name for s in symbols}
        assert names == {"setup", "run_tests", "cleanup"}

    def test_function_line_numbers(self):
        """Should capture correct line numbers for functions."""
        from aim_code.parsers.bash_parser import BashParser

        parser = BashParser()
        symbols = list(parser.extract_symbols(SIMPLE_FUNCTION, "test.sh"))

        sym = symbols[0]
        # Function starts on line 3 (after shebang and blank line)
        assert sym.line_start == 3
        # Function ends on line 5 (closing brace)
        assert sym.line_end == 5

    def test_function_content_captured(self):
        """Should capture full function content."""
        from aim_code.parsers.bash_parser import BashParser

        parser = BashParser()
        symbols = list(parser.extract_symbols(SIMPLE_FUNCTION, "test.sh"))

        sym = symbols[0]
        assert "my_function()" in sym.content
        assert 'echo "Hello, world!"' in sym.content

    def test_function_signature(self):
        """Should extract function signature."""
        from aim_code.parsers.bash_parser import BashParser

        parser = BashParser()
        symbols = list(parser.extract_symbols(SIMPLE_FUNCTION, "test.sh"))

        sym = symbols[0]
        assert "my_function()" in sym.signature


@requires_tree_sitter
class TestBashRawCallsExtraction:
    """Tests for extracting command calls from function bodies."""

    def test_extract_calls_to_other_functions(self):
        """Should extract calls to user-defined functions."""
        from aim_code.parsers.bash_parser import BashParser

        parser = BashParser()
        symbols = list(parser.extract_symbols(FUNCTION_CALLING_OTHER_FUNCTIONS, "test.sh"))

        # Find main_func
        main_func = next(s for s in symbols if s.name == "main_func")
        assert "helper_func" in main_func.raw_calls
        assert "do_something" in main_func.raw_calls

    def test_builtins_filtered_from_raw_calls(self):
        """Should not include shell builtins in raw_calls."""
        from aim_code.parsers.bash_parser import BashParser

        parser = BashParser()
        symbols = list(parser.extract_symbols(FUNCTION_WITH_BUILTINS_ONLY, "test.sh"))

        sym = symbols[0]
        # All commands in this function are builtins
        assert len(sym.raw_calls) == 0

    def test_echo_filtered(self):
        """echo should be filtered from raw_calls."""
        from aim_code.parsers.bash_parser import BashParser

        parser = BashParser()
        symbols = list(parser.extract_symbols(FUNCTION_CALLING_OTHER_FUNCTIONS, "test.sh"))

        main_func = next(s for s in symbols if s.name == "main_func")
        assert "echo" not in main_func.raw_calls

    def test_raw_calls_deduplicated(self):
        """raw_calls should be deduplicated."""
        from aim_code.parsers.bash_parser import BashParser

        script = """#!/bin/bash
repeat_caller() {
    helper
    helper
    helper
}
"""
        parser = BashParser()
        symbols = list(parser.extract_symbols(script, "test.sh"))

        sym = symbols[0]
        assert sym.raw_calls.count("helper") == 1

    def test_complex_script_raw_calls(self):
        """Should extract raw_calls from complex script correctly."""
        from aim_code.parsers.bash_parser import BashParser

        parser = BashParser()
        symbols = list(parser.extract_symbols(NESTED_SCRIPT, "test.sh"))

        # Find process_item function
        process_item = next(s for s in symbols if s.name == "process_item")
        assert "validate_item" in process_item.raw_calls
        assert "transform_item" in process_item.raw_calls

        # Find main function
        main_func = next(s for s in symbols if s.name == "main")
        assert "init_config" in main_func.raw_calls
        assert "process_item" in main_func.raw_calls


class TestBashImportsAndTypes:
    """Tests for imports and attribute types (which return empty for Bash)."""

    def test_extract_imports_returns_empty(self):
        """extract_imports() should return empty list for Bash."""
        from aim_code.parsers.bash_parser import BashParser

        parser = BashParser()
        imports = parser.extract_imports(NESTED_SCRIPT)

        assert imports == []

    def test_extract_attribute_types_returns_empty(self):
        """extract_attribute_types() should return empty dict for Bash."""
        from aim_code.parsers.bash_parser import BashParser

        parser = BashParser()
        types = parser.extract_attribute_types(NESTED_SCRIPT)

        assert types == {}


@requires_tree_sitter
class TestBashParseFile:
    """Tests for the parse_file() convenience method."""

    def test_parse_file_returns_parsed_file(self):
        """parse_file() should return ParsedFile with symbols."""
        from aim_code.parsers.bash_parser import BashParser
        from aim_code.parsers.base import ParsedFile

        parser = BashParser()
        result = parser.parse_file(MULTIPLE_FUNCTIONS, "test.sh")

        assert isinstance(result, ParsedFile)
        assert len(result.symbols) == 3
        assert result.imports == {}
        assert result.attribute_types == {}


class TestBashEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_script(self):
        """Should handle empty script gracefully."""
        from aim_code.parsers.bash_parser import BashParser

        parser = BashParser()
        symbols = list(parser.extract_symbols("", "empty.sh"))

        assert symbols == []

    def test_script_without_functions(self):
        """Should return empty list for script with no functions."""
        from aim_code.parsers.bash_parser import BashParser

        parser = BashParser()
        symbols = list(parser.extract_symbols(
            "#!/bin/bash\necho 'Hello'\nls -la\n",
            "no_funcs.sh"
        ))

        assert symbols == []

    def test_shebang_only(self):
        """Should handle script with only shebang."""
        from aim_code.parsers.bash_parser import BashParser

        parser = BashParser()
        symbols = list(parser.extract_symbols("#!/bin/bash\n", "shebang.sh"))

        assert symbols == []

    def test_comment_only(self):
        """Should handle script with only comments."""
        from aim_code.parsers.bash_parser import BashParser

        script = """#!/bin/bash
# This is a comment
# Another comment
"""
        parser = BashParser()
        symbols = list(parser.extract_symbols(script, "comments.sh"))

        assert symbols == []
