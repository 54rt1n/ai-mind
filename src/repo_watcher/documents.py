# repo_watcher/documents.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Document models for CODE_RAG indexing.

These Pydantic models define the structure of documents indexed by repo-watcher:

- SourceDoc: Semantic code chunks (DOC_SOURCE_CODE) - file/class/function boundaries
- SpecDoc: Module-level design specifications (DOC_SPEC) - from SPEC.md files

Both models map to ConversationMessage fields for storage in the CVM index.
The doc_id format enables efficient filtering:
- SourceDoc: "file_path::symbol_path" for prefix/exact lookups
- SpecDoc: "module.path" for module-level queries
"""

import json
from typing import Optional

from pydantic import BaseModel

from aim.constants import DOC_SOURCE_CODE, DOC_SPEC


class SourceDocMetadata(BaseModel):
    """Metadata for DOC_SOURCE_CODE documents.

    Stored as JSON in the metadata field of indexed documents.
    Contains structural information for line range filtering and
    graph construction.

    Note: Call graph edges are NOT stored here. They live in
    CodeGraph (edges.json) for efficient graph traversal.
    """

    symbol_name: str
    symbol_type: str  # "file" | "class" | "function" | "method"
    line_start: int
    line_end: int
    parent_symbol: Optional[str] = None
    signature: Optional[str] = None
    imports: list[str] = []


class SourceDoc(BaseModel):
    """Source code document for indexing.

    Maps to ConversationMessage fields for storage in the CVM index.
    The doc_id uses fully qualified symbol path for efficient filtering:
    - "packages/aim-core/src/aim/config.py::ChatConfig.from_env"

    This enables:
    - File prefix filtering via Tantivy query on doc_id prefix
    - Exact symbol lookup via Tantivy exact match on doc_id
    - Semantic search via embedding search on content field
    """

    doc_id: str  # "file_path::symbol_path"
    document_type: str = DOC_SOURCE_CODE
    content: str
    metadata: str  # JSON-encoded SourceDocMetadata
    conversation_id: str = "code"
    user_id: str = "repo-watcher"
    persona_id: str
    role: str = "user"
    timestamp: int
    sequence_no: int = 0
    branch: int = 0

    @classmethod
    def create(
        cls,
        file_path: str,
        symbol_path: str,
        content: str,
        meta: SourceDocMetadata,
        persona_id: str,
        timestamp: int,
    ) -> "SourceDoc":
        """Factory method to create a SourceDoc with proper doc_id.

        Args:
            file_path: Path to source file.
            symbol_path: Symbol path within file (e.g., "ChatConfig.from_env").
                        Empty string for file-level documents.
            content: Source code content.
            meta: SourceDocMetadata with structural information.
            persona_id: Agent ID (e.g., "blip").
            timestamp: File modification time as Unix timestamp.

        Returns:
            SourceDoc ready for indexing.
        """
        doc_id = f"{file_path}::{symbol_path}" if symbol_path else file_path
        return cls(
            doc_id=doc_id,
            content=content,
            metadata=meta.model_dump_json(),
            persona_id=persona_id,
            timestamp=timestamp,
        )


class SpecDoc(BaseModel):
    """Module specification document.

    Stores SPEC.md content as DOC_SPEC for module-level design documentation.
    These are externally generated (by code-systems agent or similar) and
    placed alongside code in the repository.

    The doc_id is the module path (e.g., "aim.conversation") for direct
    module-level queries.
    """

    doc_id: str  # Module path: "aim.conversation"
    document_type: str = DOC_SPEC
    content: str
    metadata: str  # JSON: {"file_path": "...", "module_path": "..."}
    conversation_id: str = "specs"
    user_id: str = "repo-watcher"
    persona_id: str
    role: str = "user"
    timestamp: int
    sequence_no: int = 0
    branch: int = 0

    @classmethod
    def create(
        cls,
        file_path: str,
        module_path: str,
        content: str,
        persona_id: str,
        timestamp: int,
    ) -> "SpecDoc":
        """Factory method to create a SpecDoc.

        Args:
            file_path: Path to SPEC.md file.
            module_path: Derived module path (e.g., "aim.conversation").
            content: Full markdown content of SPEC.md.
            persona_id: Agent ID (e.g., "blip").
            timestamp: File modification time as Unix timestamp.

        Returns:
            SpecDoc ready for indexing.
        """
        return cls(
            doc_id=module_path,
            content=content,
            metadata=json.dumps({"file_path": file_path, "module_path": module_path}),
            persona_id=persona_id,
            timestamp=timestamp,
        )
