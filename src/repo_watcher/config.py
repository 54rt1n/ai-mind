# repo_watcher/config.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Configuration models for repo-watcher indexing service.

Defines the structure of YAML configuration files used to specify
which source paths to index and how to configure the CVM.
"""

from pathlib import Path
from typing import Optional

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

    This model maps directly to the YAML configuration file format.

    Attributes:
        repo_id: Unique identifier for this repository.
        agent_id: Agent to index for (e.g., "blip").
        memory_path: Path to store CVM indices and call graph.
        embedding_model: Model name for FAISS embeddings.
        device: Device for embedding generation ("cpu", "cuda:0", etc.).
        user_timezone: Optional timezone for timestamp handling.
        sources: List of source paths to index.

    Example YAML:
        repo_id: ai-mind
        agent_id: blip
        memory_path: memory/blip
        embedding_model: mixedbread-ai/mxbai-embed-large-v1
        device: cuda:0
        user_timezone: America/Los_Angeles
        sources:
          - path: packages/aim-core/src
            recursive: true
            language: python
    """

    repo_id: str
    agent_id: str
    memory_path: Path
    embedding_model: str
    device: str = "cpu"
    user_timezone: Optional[str] = None
    sources: list[SourcePath]
