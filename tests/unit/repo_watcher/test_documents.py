# tests/unit/repo_watcher/test_documents.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Unit tests for repo_watcher.documents.

Tests the document models used for CODE_RAG indexing: SourceDoc, SourceDocMetadata,
and SpecDoc.
"""

import json

import pytest

from repo_watcher.documents import SourceDoc, SourceDocMetadata, SpecDoc
from aim.constants import DOC_SOURCE_CODE, DOC_SPEC


class TestSourceDocMetadata:
    """Tests for SourceDocMetadata model."""

    def test_minimal_metadata(self):
        """Can create metadata with required fields only."""
        meta = SourceDocMetadata(
            symbol_name="my_function",
            symbol_type="function",
            line_start=10,
            line_end=25,
        )
        assert meta.symbol_name == "my_function"
        assert meta.symbol_type == "function"
        assert meta.line_start == 10
        assert meta.line_end == 25
        assert meta.parent_symbol is None
        assert meta.signature is None
        assert meta.imports == []

    def test_full_metadata(self):
        """Can create metadata with all fields."""
        meta = SourceDocMetadata(
            symbol_name="from_env",
            symbol_type="method",
            line_start=192,
            line_end=196,
            parent_symbol="ChatConfig",
            signature="def from_env(cls) -> ChatConfig",
            imports=["os", "typing.Optional"],
        )
        assert meta.parent_symbol == "ChatConfig"
        assert meta.signature is not None
        assert len(meta.imports) == 2

    def test_metadata_to_json(self):
        """Metadata can be serialized to JSON."""
        meta = SourceDocMetadata(
            symbol_name="test",
            symbol_type="function",
            line_start=1,
            line_end=5,
        )
        json_str = meta.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["symbol_name"] == "test"
        assert parsed["symbol_type"] == "function"
        assert parsed["line_start"] == 1
        assert parsed["line_end"] == 5

    def test_symbol_types(self):
        """Metadata accepts various symbol types."""
        for sym_type in ["file", "class", "function", "method"]:
            meta = SourceDocMetadata(
                symbol_name="test",
                symbol_type=sym_type,
                line_start=1,
                line_end=10,
            )
            assert meta.symbol_type == sym_type


class TestSourceDoc:
    """Tests for SourceDoc model."""

    @pytest.fixture
    def sample_metadata(self) -> SourceDocMetadata:
        """Create sample metadata for tests."""
        return SourceDocMetadata(
            symbol_name="process_data",
            symbol_type="function",
            line_start=15,
            line_end=30,
            signature="def process_data(data: dict) -> str",
            imports=["json"],
        )

    def test_document_type_default(self, sample_metadata):
        """SourceDoc should default to DOC_SOURCE_CODE."""
        doc = SourceDoc(
            doc_id="file.py::process_data",
            content="def process_data(data):\n    pass",
            metadata=sample_metadata.model_dump_json(),
            persona_id="blip",
            timestamp=1234567890,
        )
        assert doc.document_type == DOC_SOURCE_CODE

    def test_default_fields(self, sample_metadata):
        """SourceDoc should have correct defaults."""
        doc = SourceDoc(
            doc_id="file.py::func",
            content="...",
            metadata="{}",
            persona_id="blip",
            timestamp=1234567890,
        )
        assert doc.conversation_id == "code"
        assert doc.user_id == "repo-watcher"
        assert doc.role == "user"
        assert doc.sequence_no == 0
        assert doc.branch == 0

    def test_create_with_symbol_path(self, sample_metadata):
        """SourceDoc.create should build doc_id from file_path::symbol_path."""
        doc = SourceDoc.create(
            file_path="packages/aim-core/src/aim/config.py",
            symbol_path="ChatConfig.from_env",
            content="def from_env(cls): ...",
            meta=sample_metadata,
            persona_id="blip",
            timestamp=1234567890,
        )

        assert doc.doc_id == "packages/aim-core/src/aim/config.py::ChatConfig.from_env"
        assert doc.content == "def from_env(cls): ..."
        assert doc.persona_id == "blip"
        assert doc.timestamp == 1234567890

    def test_create_file_level_doc(self, sample_metadata):
        """SourceDoc.create with empty symbol_path uses file_path as doc_id."""
        doc = SourceDoc.create(
            file_path="packages/aim-core/src/aim/config.py",
            symbol_path="",
            content="# File header...",
            meta=sample_metadata,
            persona_id="blip",
            timestamp=1234567890,
        )

        assert doc.doc_id == "packages/aim-core/src/aim/config.py"

    def test_metadata_is_json_string(self, sample_metadata):
        """Metadata field should be JSON string, not object."""
        doc = SourceDoc.create(
            file_path="file.py",
            symbol_path="func",
            content="...",
            meta=sample_metadata,
            persona_id="blip",
            timestamp=1234567890,
        )

        # metadata should be a string
        assert isinstance(doc.metadata, str)

        # And it should be valid JSON
        parsed = json.loads(doc.metadata)
        assert parsed["symbol_name"] == "process_data"
        assert parsed["symbol_type"] == "function"

    def test_doc_id_format_enables_prefix_search(self):
        """Doc IDs should enable file-level prefix filtering."""
        docs = [
            SourceDoc.create(
                file_path="src/aim/config.py",
                symbol_path="ChatConfig",
                content="...",
                meta=SourceDocMetadata(
                    symbol_name="ChatConfig",
                    symbol_type="class",
                    line_start=10,
                    line_end=50,
                ),
                persona_id="blip",
                timestamp=1234567890,
            ),
            SourceDoc.create(
                file_path="src/aim/config.py",
                symbol_path="ChatConfig.from_env",
                content="...",
                meta=SourceDocMetadata(
                    symbol_name="from_env",
                    symbol_type="method",
                    line_start=20,
                    line_end=30,
                ),
                persona_id="blip",
                timestamp=1234567890,
            ),
        ]

        # All docs from same file share the same prefix
        prefix = "src/aim/config.py::"
        assert all(doc.doc_id.startswith("src/aim/config.py") for doc in docs)


class TestSpecDoc:
    """Tests for SpecDoc model."""

    def test_document_type_default(self):
        """SpecDoc should default to DOC_SPEC."""
        doc = SpecDoc(
            doc_id="aim.conversation",
            content="# Module: aim.conversation\n...",
            metadata="{}",
            persona_id="blip",
            timestamp=1234567890,
        )
        assert doc.document_type == DOC_SPEC

    def test_default_fields(self):
        """SpecDoc should have correct defaults."""
        doc = SpecDoc(
            doc_id="aim.conversation",
            content="...",
            metadata="{}",
            persona_id="blip",
            timestamp=1234567890,
        )
        assert doc.conversation_id == "specs"
        assert doc.user_id == "repo-watcher"
        assert doc.role == "user"
        assert doc.sequence_no == 0
        assert doc.branch == 0

    def test_create_spec_doc(self):
        """SpecDoc.create should build doc with correct metadata."""
        doc = SpecDoc.create(
            file_path="packages/aim-core/src/aim/conversation/SPEC.md",
            module_path="aim.conversation",
            content="# Module: aim.conversation\n\nPurpose...",
            persona_id="blip",
            timestamp=1234567890,
        )

        assert doc.doc_id == "aim.conversation"
        assert "aim.conversation" in doc.content
        assert doc.persona_id == "blip"
        assert doc.timestamp == 1234567890

    def test_metadata_contains_paths(self):
        """SpecDoc metadata should contain both file_path and module_path."""
        doc = SpecDoc.create(
            file_path="packages/aim-core/src/aim/conversation/SPEC.md",
            module_path="aim.conversation",
            content="...",
            persona_id="blip",
            timestamp=1234567890,
        )

        # metadata is JSON string
        assert isinstance(doc.metadata, str)

        parsed = json.loads(doc.metadata)
        assert parsed["file_path"] == "packages/aim-core/src/aim/conversation/SPEC.md"
        assert parsed["module_path"] == "aim.conversation"

    def test_spec_doc_content_preserved(self):
        """SpecDoc should preserve markdown content exactly."""
        markdown = '''# Module: aim.conversation

## Purpose

Provides conversation memory management with vector search.

## Schema

```python
@dataclass
class ConversationMessage:
    doc_id: str
    content: str
```

## Interactions

```mermaid
sequenceDiagram
    participant A as Agent
    participant CVM as ConversationModel
    A->>CVM: query()
```
'''
        doc = SpecDoc.create(
            file_path="SPEC.md",
            module_path="aim.conversation",
            content=markdown,
            persona_id="blip",
            timestamp=1234567890,
        )

        assert doc.content == markdown


class TestModuleExports:
    """Tests for repo_watcher module exports."""

    def test_can_import_from_package(self):
        """All document models should be importable from package."""
        from repo_watcher import SourceDoc, SourceDocMetadata, SpecDoc

        assert SourceDoc is not None
        assert SourceDocMetadata is not None
        assert SpecDoc is not None
