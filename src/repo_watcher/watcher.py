# repo_watcher/watcher.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Repository watcher for CODE_RAG indexing.

Performs two-pass indexing:
- Pass 1: Parse files, extract symbols, index DOC_SOURCE_CODEs, build symbol table + module registry
- Pass 2: Resolve calls, build call graph edges
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Iterator

from aim_code.parsers import ParserRegistry
from aim_code.graph import (
    CodeGraph,
    ModuleRegistry,
    SymbolTable,
    ImportResolver,
    Symbol,
)
from aim_code.graph.models import ParsedFile as GraphParsedFile
from aim.config import ChatConfig
from aim.conversation.model import ConversationModel

from aim_code.documents import SourceDoc, SourceDocMetadata, SpecDoc

from .config import RepoConfig, SourcePath

logger = logging.getLogger(__name__)


# File extension mapping for languages
LANGUAGE_EXTENSIONS = {
    "python": ".py",
    "typescript": ".ts",
    "bash": ".sh",
}


class RepoWatcher:
    """Indexes source code repositories for CODE_RAG.

    The watcher performs two-pass indexing:

    Pass 1 (Index Symbols):
    - Parse all source files using tree-sitter parsers
    - Extract symbols (classes, functions, methods) with their raw call targets
    - Index each symbol as a DOC_SOURCE_CODE document in CVM
    - Build symbol table for qualified name lookups
    - Build module registry for import resolution

    Pass 2 (Build Graph):
    - For each file's symbols, resolve raw calls to SymbolRefs
    - Add edges to CodeGraph for caller -> callee relationships
    - External calls (to packages outside the codebase) get line_start=-1

    The result is a CVM index of code documents and a call graph for
    structural navigation via the focus tool.
    """

    def __init__(self, config: RepoConfig):
        """Initialize watcher with configuration.

        Args:
            config: Repository configuration specifying sources.
                    Memory path, embedding model, etc. loaded from .env via ChatConfig.
        """
        self.config = config
        self.registry = ParserRegistry()
        self.graph = CodeGraph()
        self.symbol_table = SymbolTable()
        self.module_registry = ModuleRegistry()
        self.file_cache: dict[str, GraphParsedFile] = {}
        self.cvm: ConversationModel = None  # type: ignore[assignment]
        self.chat_config: ChatConfig = None  # type: ignore[assignment]

    def run(self) -> None:
        """Run two-pass indexing.

        This is the main entry point. It:
        1. Initializes the CVM for indexing
        2. Pass 1: Indexes symbols from all source files
        3. Indexes SPEC.md files as DOC_SPEC documents
        4. Pass 2: Resolves calls and builds the call graph
        5. Saves the CVM index and call graph
        """
        logger.info(f"Starting indexing for {self.config.repo_id}")

        # Initialize CVM
        self._init_cvm()

        # Pass 1: Index symbols
        file_count = 0
        symbol_count = 0
        skip_count = 0
        for source in self.config.sources:
            for file_path in self._iter_files(source):
                symbols, skipped = self._pass1_index_symbols(
                    str(file_path), source.language, str(source.path)
                )
                file_count += 1
                symbol_count += symbols
                skip_count += skipped
        logger.info(f"Pass 1 complete: {file_count} files, {symbol_count} symbols indexed, {skip_count} unchanged (skipped)")

        # Index SPEC.md files
        spec_count = self._index_spec_files()
        logger.info(f"Indexed {spec_count} SPEC.md files")

        # Pass 2: Resolve calls, build graph
        edge_count = 0
        for file_path, parsed in self.file_cache.items():
            edges = self._pass2_build_graph(file_path, parsed)
            edge_count += edges
        logger.info(f"Pass 2 complete: {edge_count} edges in call graph")

        # Save graph
        graph_path = Path(self.cvm.memory_path) / "graph"
        self.graph.save(graph_path)
        logger.info(f"Graph saved to {graph_path}")

    def _init_cvm(self) -> None:
        """Initialize the ConversationModel for indexing.

        Loads settings from .env via ChatConfig.from_env() to get memory_path,
        embedding_model, etc. Builds the agent's memory path as:
        $MEMORY_PATH/$agent_id
        """
        # Load from .env to get memory_path, embedding_model, etc.
        self.chat_config = ChatConfig.from_env()

        # Build memory path: $MEMORY_PATH/$agent_id
        memory_path = os.path.join(self.chat_config.memory_path, self.config.agent_id)

        ConversationModel.maybe_init_folders(memory_path)
        self.cvm = ConversationModel(
            memory_path=memory_path,
            embedding_model=self.chat_config.embedding_model,
            embedding_device=self.chat_config.embedding_device,
            user_timezone=self.chat_config.user_timezone,
        )

    def _iter_files(self, source: SourcePath) -> Iterator[Path]:
        """Iterate over source files matching the language extension.

        Args:
            source: Source path configuration with language and recursive flag.

        Yields:
            Path objects for matching source files.
        """
        path = Path(source.path)
        if not path.exists():
            logger.warning(f"Source path does not exist: {path}")
            return

        ext = LANGUAGE_EXTENSIONS.get(source.language, "")
        if not ext:
            logger.warning(f"Unknown language: {source.language}")
            return

        if source.recursive:
            yield from path.rglob(f"*{ext}")
        else:
            yield from path.glob(f"*{ext}")

    def _pass1_index_symbols(
        self, file_path: str, language: str, source_root: str
    ) -> tuple[int, int]:
        """Pass 1: Extract symbols and index as DOC_SOURCE_CODE documents.

        Args:
            file_path: Path to the source file.
            language: Programming language ("python", "typescript", "bash").
            source_root: Root directory of the source tree for module derivation.

        Returns:
            Tuple of (symbols_indexed, symbols_skipped).
        """
        try:
            content = Path(file_path).read_text()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return (0, 0)

        timestamp = int(Path(file_path).stat().st_mtime)

        # Register module
        module_name = ModuleRegistry.file_to_module(file_path, source_root)
        self.module_registry.add(module_name, file_path)

        # Parse file
        parser = self.registry.get_parser(language)
        if not parser or not parser.is_available():
            logger.debug(f"No parser available for {language}, skipping {file_path}")
            return (0, 0)

        try:
            parsed = parser.parse_file(content, file_path)
        except Exception as e:
            logger.warning(f"Parse error in {file_path}: {e}")
            return (0, 0)

        # Convert ExtractedSymbol -> Symbol for graph cache (strips content)
        symbols = [
            Symbol(
                name=s.name,
                symbol_type=s.symbol_type,
                line_start=s.line_start,
                line_end=s.line_end,
                parent=s.parent,
                signature=s.signature,
                raw_calls=s.raw_calls,
            )
            for s in parsed.symbols
        ]

        # Cache for pass 2 (use GraphParsedFile with Symbol, not ExtractedSymbol)
        cached = GraphParsedFile(
            imports=parsed.imports,
            symbols=symbols,
            attribute_types=parsed.attribute_types,
        )
        self.file_cache[file_path] = cached

        # Index each symbol as DOC_SOURCE_CODE
        lines = content.split("\n")
        indexed_count = 0
        skipped_count = 0
        for symbol in parsed.symbols:
            symbol_path = (
                f"{symbol.parent}.{symbol.name}" if symbol.parent else symbol.name
            )
            # Get symbol content from file (line numbers are 1-indexed)
            symbol_content = "\n".join(
                lines[symbol.line_start - 1 : symbol.line_end]
            )
            # Hash content for change detection
            content_hash = hashlib.sha256(symbol_content.encode()).hexdigest()
            meta = SourceDocMetadata(
                symbol_name=symbol.name,
                symbol_type=symbol.symbol_type,
                line_start=symbol.line_start,
                line_end=symbol.line_end,
                content_hash=content_hash,
                parent_symbol=symbol.parent,
                signature=symbol.signature,
                imports=list(parsed.imports.keys()),
            )

            doc = SourceDoc.create(
                file_path=file_path,
                symbol_path=symbol_path,
                content=symbol_content,
                meta=meta,
                persona_id=self.config.agent_id,
                timestamp=timestamp,
            )
            if self._insert_doc(doc):
                indexed_count += 1
            else:
                skipped_count += 1

            # Add to symbol table for pass 2 resolution
            self.symbol_table.add(
                file_path, symbol.name, symbol.parent, symbol.line_start
            )

        return (indexed_count, skipped_count)

    def _insert_doc(self, doc: SourceDoc | SpecDoc) -> bool:
        """Insert a document directly into the search index (no JSONL).

        Skips insertion if a document with the same doc_id and content_hash
        already exists (unchanged file).

        Args:
            doc: SourceDoc or SpecDoc to insert.

        Returns:
            True if document was inserted, False if skipped (unchanged).
        """
        import json

        # Check if document already exists with same hash
        existing = self.cvm.index.get_document(doc.doc_id)
        if existing:
            existing_meta = existing.get("metadata", "{}")
            if isinstance(existing_meta, str):
                try:
                    existing_meta = json.loads(existing_meta)
                except json.JSONDecodeError:
                    existing_meta = {}

            # Parse new doc metadata
            new_meta = doc.metadata
            if isinstance(new_meta, str):
                try:
                    new_meta = json.loads(new_meta)
                except json.JSONDecodeError:
                    new_meta = {}

            # Compare content hashes
            existing_hash = existing_meta.get("content_hash", "")
            new_hash = new_meta.get("content_hash", "")
            if existing_hash and new_hash and existing_hash == new_hash:
                logger.debug(f"Skipping unchanged: {doc.doc_id}")
                return False

        doc_dict = doc.model_dump()
        # Add fields required by ConversationMessage
        doc_dict.setdefault("speaker_id", doc.user_id)
        doc_dict.setdefault("listener_id", doc.persona_id)
        doc_dict.setdefault("reference_id", doc.conversation_id)
        # Validate and normalize via ConversationMessage
        from aim.conversation.message import ConversationMessage
        message = ConversationMessage.from_dict(doc_dict)
        # Insert directly into index, bypassing JSONL
        logger.info(f"Inserting {doc.doc_id} into index")
        self.cvm.index.add_document(message.to_dict())
        return True

    def _pass2_build_graph(self, file_path: str, parsed: GraphParsedFile) -> int:
        """Pass 2: Resolve calls and build call graph edges.

        Args:
            file_path: Path to the source file.
            parsed: ParsedFile with imports, symbols, and attribute types.

        Returns:
            Number of edges added for this file.
        """
        resolver = ImportResolver(
            imports=parsed.imports,
            attribute_types=parsed.attribute_types,
            symbol_table=self.symbol_table,
            module_registry=self.module_registry,
            current_file=file_path,
        )

        edge_count = 0
        for symbol in parsed.symbols:
            qualified = (
                f"{symbol.parent}.{symbol.name}" if symbol.parent else symbol.name
            )
            caller_ref = (file_path, qualified, symbol.line_start)

            for raw_call in symbol.raw_calls:
                callee_ref = resolver.resolve(raw_call, parent_class=symbol.parent)
                if callee_ref:
                    self.graph.add_edge(caller_ref, callee_ref)
                    edge_count += 1

        return edge_count

    def _index_spec_files(self) -> int:
        """Find and index SPEC.md files as DOC_SPEC documents.

        Returns:
            Number of SPEC.md files indexed.
        """
        count = 0
        for source in self.config.sources:
            source_path = Path(source.path)
            if not source_path.exists():
                continue

            for spec_path in source_path.rglob("SPEC.md"):
                try:
                    content = spec_path.read_text()
                    module_path = self._derive_module_path(spec_path, source_path)
                    timestamp = int(spec_path.stat().st_mtime)

                    doc = SpecDoc.create(
                        file_path=str(spec_path),
                        module_path=module_path,
                        content=content,
                        persona_id=self.config.agent_id,
                        timestamp=timestamp,
                    )
                    self._insert_doc(doc)
                    count += 1
                except Exception as e:
                    logger.warning(f"Error indexing {spec_path}: {e}")

        return count

    def _derive_module_path(self, spec_path: Path, source_root: Path) -> str:
        """Derive module path from SPEC.md location.

        The SPEC.md file's parent directory is converted to a dotted module path
        relative to the source root.

        Args:
            spec_path: Path to the SPEC.md file.
            source_root: Root of the source tree.

        Returns:
            Module path string (e.g., "aim.conversation").
        """
        try:
            rel = spec_path.parent.relative_to(source_root)
            return str(rel).replace("/", ".") if str(rel) != "." else ""
        except ValueError:
            # spec_path is not under source_root
            return spec_path.parent.name
