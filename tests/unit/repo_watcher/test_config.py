# tests/unit/repo_watcher/test_config.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Unit tests for repo_watcher.config.

Tests the configuration models: SourcePath and RepoConfig.
"""

from pathlib import Path

import pytest
import yaml

from repo_watcher.config import RepoConfig, SourcePath


class TestSourcePath:
    """Tests for SourcePath model."""

    def test_minimal_source_path(self):
        """Can create SourcePath with required fields only."""
        source = SourcePath(
            path=Path("packages/aim-core/src"),
            language="python",
        )
        assert source.path == Path("packages/aim-core/src")
        assert source.language == "python"
        assert source.recursive is True  # Default

    def test_source_path_with_recursive_false(self):
        """Can disable recursive scanning."""
        source = SourcePath(
            path=Path("src"),
            language="python",
            recursive=False,
        )
        assert source.recursive is False

    def test_source_path_accepts_string_path(self):
        """SourcePath should accept string paths and convert to Path."""
        source = SourcePath(
            path="packages/aim-core/src",
            language="python",
        )
        assert isinstance(source.path, Path)
        assert source.path == Path("packages/aim-core/src")

    def test_source_path_language_types(self):
        """SourcePath accepts various language types."""
        for lang in ["python", "typescript", "bash"]:
            source = SourcePath(path="src", language=lang)
            assert source.language == lang


class TestRepoConfig:
    """Tests for RepoConfig model."""

    def test_minimal_config(self):
        """Can create RepoConfig with required fields only."""
        config = RepoConfig(
            repo_id="test-repo",
            agent_id="blip",
            sources=[
                SourcePath(path="src", language="python"),
            ],
        )
        assert config.repo_id == "test-repo"
        assert config.agent_id == "blip"
        assert len(config.sources) == 1

    def test_full_config(self):
        """Can create RepoConfig with multiple sources."""
        config = RepoConfig(
            repo_id="ai-mind",
            agent_id="blip",
            sources=[
                SourcePath(
                    path="packages/aim-core/src",
                    recursive=True,
                    language="python",
                ),
                SourcePath(
                    path="webui/src",
                    recursive=True,
                    language="typescript",
                ),
            ],
        )
        assert config.repo_id == "ai-mind"
        assert config.agent_id == "blip"
        assert len(config.sources) == 2

    def test_config_from_dict(self):
        """Can create RepoConfig from a dictionary (like YAML load)."""
        config_dict = {
            "repo_id": "ai-mind",
            "agent_id": "blip",
            "sources": [
                {
                    "path": "packages/aim-core/src",
                    "recursive": True,
                    "language": "python",
                },
            ],
        }
        config = RepoConfig(**config_dict)
        assert config.repo_id == "ai-mind"
        assert config.agent_id == "blip"
        assert len(config.sources) == 1
        assert isinstance(config.sources[0], SourcePath)

    def test_config_roundtrip_yaml(self):
        """RepoConfig can be serialized/deserialized via YAML."""
        yaml_content = """
repo_id: ai-mind
agent_id: blip

sources:
  - path: packages/aim-core/src
    recursive: true
    language: python
  - path: webui/src
    recursive: true
    language: typescript
"""
        config_dict = yaml.safe_load(yaml_content)
        config = RepoConfig(**config_dict)

        assert config.repo_id == "ai-mind"
        assert config.agent_id == "blip"
        assert len(config.sources) == 2
        assert config.sources[0].language == "python"
        assert config.sources[1].language == "typescript"

    def test_empty_sources_allowed(self):
        """RepoConfig can have empty sources list."""
        config = RepoConfig(
            repo_id="empty",
            agent_id="test",
            sources=[],
        )
        assert config.sources == []

    def test_multiple_sources_same_language(self):
        """RepoConfig can have multiple sources with same language."""
        config = RepoConfig(
            repo_id="multi",
            agent_id="test",
            sources=[
                SourcePath(path="src", language="python"),
                SourcePath(path="packages/aim-core/src", language="python"),
                SourcePath(path="packages/aim-mud/src", language="python"),
            ],
        )
        assert len(config.sources) == 3
        assert all(s.language == "python" for s in config.sources)


class TestModuleExports:
    """Tests for repo_watcher.config module exports."""

    def test_can_import_from_package(self):
        """Config models should be importable from package."""
        from repo_watcher import RepoConfig, SourcePath

        assert RepoConfig is not None
        assert SourcePath is not None
