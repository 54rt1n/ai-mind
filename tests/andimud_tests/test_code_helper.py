# tests/andimud_tests/test_code_helper.py
"""Tests for CodeRoom helper functions.

These tests verify the list_directories and resolve_cd_path functions
used for CodeRoom navigation in ANDIMUD.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import os

# Import the code helper module
import sys

andimud_path = Path(__file__).parent.parent.parent / "andimud"
if str(andimud_path) not in sys.path:
    sys.path.insert(0, str(andimud_path))

CODE_HELPER_AVAILABLE = False
list_directories = None
resolve_cd_path = None
run_command = None
CommandResult = None

try:
    from typeclasses.code.code_helper import (
        list_directories,
        resolve_cd_path,
        run_command,
        CommandResult,
    )
    CODE_HELPER_AVAILABLE = True
except Exception:
    pass

if not CODE_HELPER_AVAILABLE:
    pytestmark = pytest.mark.skip(
        reason="code_helper module not available"
    )


@pytest.mark.skipif(not CODE_HELPER_AVAILABLE, reason="code_helper not available")
class TestResolveClPath:
    """Tests for the resolve_cd_path function."""

    def test_empty_target_returns_root(self):
        """Empty target means go to root."""
        assert resolve_cd_path("", "src/foo") == ""
        assert resolve_cd_path("", "") == ""
        assert resolve_cd_path("", "deep/nested/path") == ""

    def test_slash_target_returns_root(self):
        """Single slash means go to root."""
        assert resolve_cd_path("/", "src/foo") == ""
        assert resolve_cd_path("/", "") == ""
        assert resolve_cd_path("/", "a/b/c") == ""

    def test_dot_stays_in_current(self):
        """Single dot stays in current directory."""
        assert resolve_cd_path(".", "src/foo") == "src/foo"
        assert resolve_cd_path(".", "") == ""
        assert resolve_cd_path(".", "components") == "components"

    def test_dotdot_goes_up_one_level(self):
        """Double dot goes up one level."""
        assert resolve_cd_path("..", "src/foo") == "src"
        assert resolve_cd_path("..", "a/b/c") == "a/b"
        assert resolve_cd_path("..", "single") == ""

    def test_dotdot_at_root_returns_none(self):
        """Cannot go above root."""
        assert resolve_cd_path("..", "") is None

    def test_multiple_dotdot_resolves_correctly(self):
        """Multiple .. segments resolve correctly."""
        assert resolve_cd_path("../..", "a/b/c") == "a"
        assert resolve_cd_path("../../..", "a/b/c") == ""
        assert resolve_cd_path("../../../..", "a/b/c") is None

    def test_absolute_path_starts_from_root(self):
        """Absolute path (starts with /) uses target as base from root."""
        assert resolve_cd_path("/bar", "src/foo") == "bar"
        assert resolve_cd_path("/a/b/c", "x/y/z") == "a/b/c"
        assert resolve_cd_path("/single", "deep/nested") == "single"

    def test_absolute_path_with_dotdot(self):
        """Absolute path can contain .. segments."""
        assert resolve_cd_path("/a/../b", "x/y") == "b"
        assert resolve_cd_path("/a/b/../c", "x") == "a/c"
        assert resolve_cd_path("/..", "x/y") is None

    def test_relative_path_appends(self):
        """Relative path appends to current."""
        assert resolve_cd_path("sub", "src") == "src/sub"
        assert resolve_cd_path("sub/dir", "src") == "src/sub/dir"
        assert resolve_cd_path("foo", "") == "foo"
        assert resolve_cd_path("a/b/c", "") == "a/b/c"

    def test_mixed_relative_path(self):
        """Mixed relative path with .. resolves correctly."""
        assert resolve_cd_path("../baz", "src/foo") == "src/baz"
        assert resolve_cd_path("../bar/baz", "src/foo") == "src/bar/baz"
        assert resolve_cd_path("../../new", "a/b/c") == "a/new"

    def test_trailing_slashes_ignored(self):
        """Trailing slashes in path segments are handled."""
        # Empty components from trailing slashes are skipped
        assert resolve_cd_path("foo/", "src") == "src/foo"
        assert resolve_cd_path("foo//bar", "src") == "src/foo/bar"

    def test_dot_in_middle_ignored(self):
        """Single dot in middle of path is ignored."""
        assert resolve_cd_path("./foo", "src") == "src/foo"
        assert resolve_cd_path("foo/./bar", "src") == "src/foo/bar"
        assert resolve_cd_path("./", "src") == "src"

    def test_complex_navigation(self):
        """Complex navigation patterns work correctly."""
        # Go up, then into sibling
        assert resolve_cd_path("../sibling/child", "parent/current") == "parent/sibling/child"
        # Multiple ups and downs
        assert resolve_cd_path("../../x/y/z", "a/b/c") == "a/x/y/z"
        # Edge case: exactly back to root then into new path
        assert resolve_cd_path("../../../new", "a/b/c") == "new"


@pytest.mark.skipif(not CODE_HELPER_AVAILABLE, reason="code_helper not available")
class TestListDirectories:
    """Tests for the list_directories function."""

    @pytest.fixture
    def temp_directory_tree(self):
        """Create a temporary directory tree for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Create directory structure:
            # root/
            #   dir1/
            #     subdir1/
            #     subdir2/
            #   dir2/
            #   .hidden/
            #   __pycache__/
            (root / "dir1").mkdir()
            (root / "dir1" / "subdir1").mkdir()
            (root / "dir1" / "subdir2").mkdir()
            (root / "dir2").mkdir()
            (root / ".hidden").mkdir()
            (root / "__pycache__").mkdir()
            # Also create some files (should not be included)
            (root / "file.txt").touch()
            (root / "dir1" / "file.py").touch()
            yield root

    def test_list_directories_returns_all_subdirs(self, temp_directory_tree):
        """list_directories returns all subdirectories under root."""
        root = temp_directory_tree
        result = list_directories(root, [])

        # Should include all directories but not root itself
        result_names = {p.name for p in result}
        assert "dir1" in result_names
        assert "dir2" in result_names
        assert "subdir1" in result_names
        assert "subdir2" in result_names
        assert ".hidden" in result_names
        assert "__pycache__" in result_names

    def test_list_directories_excludes_root(self, temp_directory_tree):
        """list_directories does not include root directory."""
        root = temp_directory_tree
        result = list_directories(root, [])

        for path in result:
            assert path.resolve() != root.resolve()

    def test_list_directories_with_ignore_patterns(self, temp_directory_tree):
        """list_directories respects ignore patterns."""
        root = temp_directory_tree
        result = list_directories(root, ["__pycache__", ".*"])

        result_names = {p.name for p in result}
        assert "dir1" in result_names
        assert "dir2" in result_names
        assert "__pycache__" not in result_names
        assert ".hidden" not in result_names

    def test_list_directories_empty_root(self):
        """list_directories on empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = list_directories(root, [])
            assert result == []

    def test_list_directories_returns_paths(self, temp_directory_tree):
        """list_directories returns Path objects."""
        root = temp_directory_tree
        result = list_directories(root, [])

        for item in result:
            assert isinstance(item, Path)
            assert item.is_dir()

    def test_list_directories_find_fallback(self, temp_directory_tree):
        """list_directories falls back to os.walk when find fails."""
        root = temp_directory_tree

        # Create a mock CommandResult that indicates failure
        failed_result = CommandResult(ok=False, stdout="", stderr="find not found", exit_code=127)

        with patch('typeclasses.code.code_helper.run_command', return_value=failed_result):
            result = list_directories(root, [])

        # Should still return directories via fallback
        result_names = {p.name for p in result}
        assert "dir1" in result_names
        assert "dir2" in result_names

    def test_list_directories_nested_ignore(self, temp_directory_tree):
        """list_directories ignores nested directories matching pattern."""
        root = temp_directory_tree
        # Ignore anything with 'sub' in the name
        result = list_directories(root, ["sub*"])

        result_names = {p.name for p in result}
        assert "dir1" in result_names
        assert "dir2" in result_names
        assert "subdir1" not in result_names
        assert "subdir2" not in result_names
