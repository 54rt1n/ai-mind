# repo_watcher/__init__.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Repository watcher for CODE_RAG indexing.

Indexes source code as DOC_SOURCE_CODE documents and builds call graphs
for structural code navigation.

Usage:
    python -m repo_watcher --config config/sources/ai-mind.yaml

Components:
    - RepoConfig: Configuration model for YAML files
    - SourcePath: Source directory configuration
    - RepoWatcher: Main indexer with two-pass processing
    - SourceDoc: DOC_SOURCE_CODE document model
    - SpecDoc: DOC_SPEC document model
"""

from .config import RepoConfig, SourcePath
from .documents import SourceDoc, SourceDocMetadata, SpecDoc
from .watcher import RepoWatcher

__all__ = [
    # Config
    "RepoConfig",
    "SourcePath",
    # Watcher
    "RepoWatcher",
    # Documents
    "SourceDoc",
    "SourceDocMetadata",
    "SpecDoc",
]
