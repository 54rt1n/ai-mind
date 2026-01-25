# tests/unit/repo_watcher/test_watcher.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Unit tests for repo_watcher.watcher.

Tests the RepoWatcher class that performs two-pass code indexing.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from repo_watcher.config import RepoConfig, SourcePath
from repo_watcher.watcher import RepoWatcher, LANGUAGE_EXTENSIONS


class TestLanguageExtensions:
    """Tests for language extension mapping."""

    def test_python_extension(self):
        """Python language maps to .py extension."""
        assert LANGUAGE_EXTENSIONS["python"] == ".py"

    def test_typescript_extension(self):
        """TypeScript language maps to .ts extension."""
        assert LANGUAGE_EXTENSIONS["typescript"] == ".ts"

    def test_bash_extension(self):
        """Bash language maps to .sh extension."""
        assert LANGUAGE_EXTENSIONS["bash"] == ".sh"


class TestRepoWatcherInit:
    """Tests for RepoWatcher initialization."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create a minimal config for testing."""
        return RepoConfig(
            repo_id="test",
            agent_id="blip",
            memory_path=tmp_path / "memory",
            embedding_model="test-model",
            sources=[],
        )

    def test_init_creates_empty_structures(self, config):
        """RepoWatcher initializes with empty data structures."""
        watcher = RepoWatcher(config)

        assert watcher.config == config
        assert watcher.registry is not None
        assert watcher.graph is not None
        assert watcher.symbol_table is not None
        assert watcher.module_registry is not None
        assert watcher.file_cache == {}
        assert watcher.cvm is None  # Not initialized until run()


class TestIterFiles:
    """Tests for file iteration logic."""

    @pytest.fixture
    def watcher(self, tmp_path):
        """Create watcher with temp directory."""
        config = RepoConfig(
            repo_id="test",
            agent_id="blip",
            memory_path=tmp_path / "memory",
            embedding_model="test-model",
            sources=[],
        )
        return RepoWatcher(config)

    def test_iter_files_nonexistent_directory(self, watcher, tmp_path):
        """Should yield nothing for nonexistent directory."""
        source = SourcePath(
            path=tmp_path / "nonexistent",
            language="python",
        )
        files = list(watcher._iter_files(source))
        assert files == []

    def test_iter_files_empty_directory(self, watcher, tmp_path):
        """Should yield nothing for empty directory."""
        source_dir = tmp_path / "empty"
        source_dir.mkdir()
        source = SourcePath(path=source_dir, language="python")
        files = list(watcher._iter_files(source))
        assert files == []

    def test_iter_files_finds_python_files(self, watcher, tmp_path):
        """Should find .py files in directory."""
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        (source_dir / "main.py").write_text("# main")
        (source_dir / "utils.py").write_text("# utils")
        (source_dir / "readme.txt").write_text("# readme")

        source = SourcePath(path=source_dir, language="python")
        files = list(watcher._iter_files(source))

        assert len(files) == 2
        assert all(f.suffix == ".py" for f in files)

    def test_iter_files_recursive(self, watcher, tmp_path):
        """Should recurse into subdirectories when recursive=True."""
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        (source_dir / "main.py").write_text("# main")

        subdir = source_dir / "submodule"
        subdir.mkdir()
        (subdir / "helper.py").write_text("# helper")

        source = SourcePath(path=source_dir, language="python", recursive=True)
        files = list(watcher._iter_files(source))

        assert len(files) == 2
        file_names = {f.name for f in files}
        assert "main.py" in file_names
        assert "helper.py" in file_names

    def test_iter_files_non_recursive(self, watcher, tmp_path):
        """Should not recurse when recursive=False."""
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        (source_dir / "main.py").write_text("# main")

        subdir = source_dir / "submodule"
        subdir.mkdir()
        (subdir / "helper.py").write_text("# helper")

        source = SourcePath(path=source_dir, language="python", recursive=False)
        files = list(watcher._iter_files(source))

        assert len(files) == 1
        assert files[0].name == "main.py"

    def test_iter_files_unknown_language(self, watcher, tmp_path):
        """Should yield nothing for unknown language."""
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        (source_dir / "main.py").write_text("# main")

        source = SourcePath(path=source_dir, language="unknown")
        files = list(watcher._iter_files(source))

        assert files == []


class TestDeriveModulePath:
    """Tests for SPEC.md module path derivation."""

    @pytest.fixture
    def watcher(self, tmp_path):
        """Create watcher with temp directory."""
        config = RepoConfig(
            repo_id="test",
            agent_id="blip",
            memory_path=tmp_path / "memory",
            embedding_model="test-model",
            sources=[],
        )
        return RepoWatcher(config)

    def test_derive_module_path_nested(self, watcher, tmp_path):
        """Should derive dotted module path from nested directory."""
        source_root = tmp_path / "src"
        spec_path = source_root / "aim" / "conversation" / "SPEC.md"

        result = watcher._derive_module_path(spec_path, source_root)
        assert result == "aim.conversation"

    def test_derive_module_path_single_level(self, watcher, tmp_path):
        """Should handle single-level module."""
        source_root = tmp_path / "src"
        spec_path = source_root / "utils" / "SPEC.md"

        result = watcher._derive_module_path(spec_path, source_root)
        assert result == "utils"

    def test_derive_module_path_root(self, watcher, tmp_path):
        """Should return empty string for SPEC.md at root."""
        source_root = tmp_path / "src"
        spec_path = source_root / "SPEC.md"

        result = watcher._derive_module_path(spec_path, source_root)
        assert result == ""


class TestPass1IndexSymbols:
    """Tests for pass 1 symbol indexing."""

    @pytest.fixture
    def watcher_with_mock_cvm(self, tmp_path):
        """Create watcher with mocked CVM."""
        config = RepoConfig(
            repo_id="test",
            agent_id="blip",
            memory_path=tmp_path / "memory",
            embedding_model="test-model",
            sources=[],
        )
        watcher = RepoWatcher(config)
        watcher.cvm = MagicMock()
        return watcher

    def test_pass1_unreadable_file(self, watcher_with_mock_cvm, tmp_path):
        """Should return 0 for unreadable files."""
        count = watcher_with_mock_cvm._pass1_index_symbols(
            str(tmp_path / "nonexistent.py"), "python", str(tmp_path)
        )
        assert count == 0
        watcher_with_mock_cvm.cvm.insert.assert_not_called()

    def test_pass1_registers_module(self, watcher_with_mock_cvm, tmp_path):
        """Should register module in module registry."""
        source_dir = tmp_path / "src" / "aim"
        source_dir.mkdir(parents=True)
        py_file = source_dir / "config.py"
        py_file.write_text("# Empty file")

        watcher_with_mock_cvm._pass1_index_symbols(
            str(py_file), "python", str(tmp_path / "src")
        )

        # Module should be registered
        result = watcher_with_mock_cvm.module_registry.get_file_path("aim.config")
        assert result == str(py_file)


class TestPass2BuildGraph:
    """Tests for pass 2 graph building."""

    @pytest.fixture
    def watcher(self, tmp_path):
        """Create watcher for graph tests."""
        config = RepoConfig(
            repo_id="test",
            agent_id="blip",
            memory_path=tmp_path / "memory",
            embedding_model="test-model",
            sources=[],
        )
        return RepoWatcher(config)

    def test_pass2_empty_symbols(self, watcher):
        """Should return 0 for file with no symbols."""
        from aim_code.graph.models import ParsedFile as GraphParsedFile

        parsed = GraphParsedFile(
            imports={},
            symbols=[],
            attribute_types={},
        )

        edge_count = watcher._pass2_build_graph("/fake/path.py", parsed)
        assert edge_count == 0

    def test_pass2_symbol_without_calls(self, watcher):
        """Should return 0 for symbols with no calls."""
        from aim_code.graph import Symbol
        from aim_code.graph.models import ParsedFile as GraphParsedFile

        parsed = GraphParsedFile(
            imports={},
            symbols=[
                Symbol(
                    name="my_func",
                    symbol_type="function",
                    line_start=1,
                    line_end=5,
                    raw_calls=[],
                )
            ],
            attribute_types={},
        )

        edge_count = watcher._pass2_build_graph("/fake/path.py", parsed)
        assert edge_count == 0


class TestModuleExports:
    """Tests for repo_watcher module exports."""

    def test_can_import_watcher_from_package(self):
        """RepoWatcher should be importable from package."""
        from repo_watcher import RepoWatcher

        assert RepoWatcher is not None
