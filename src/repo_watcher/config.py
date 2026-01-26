# repo_watcher/config.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Configuration models for repo-watcher indexing service.

Uses ChatConfig.from_env() for memory_path, embedding_model, etc.
RepoConfig only defines repo-specific settings (sources to index).
"""

from pathlib import Path

from pydantic import BaseModel


class SourcePath(BaseModel):
    """A source path to index.

    Attributes:
        path: Directory path to scan for source files.
        recursive: Whether to recurse into subdirectories (default True).
        language: Language for parsing: "python", "typescript", "bash".
    """

    path: Path
    recursive: bool = True
    language: str  # "python", "typescript", "bash"


class RepoConfig(BaseModel):
    """Configuration for a watched repository.

    Memory path, embedding model, device, and timezone are loaded from
    .env via ChatConfig.from_env(). This config only specifies:
    - repo_id: Identifier for this repository
    - agent_id: Agent to index for (used as persona_id for memory path)
    - sources: List of source paths to index

    Example YAML:
        repo_id: ai-mind
        agent_id: blip
        sources:
          - path: packages/aim-core/src
            recursive: true
            language: python
    """

    repo_id: str
    agent_id: str
    sources: list[SourcePath]
