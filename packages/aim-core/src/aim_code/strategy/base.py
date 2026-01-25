# aim_code/strategy/base.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
XMLCodeTurnStrategy: Code-focused turn strategy with structural context.

Provides consciousness building for code agents with:
- Focused code retrieval (explicit file/line range selection)
- Call graph traversal (height/depth neighborhood)
- Semantic code search (implicit query-based retrieval)
- Module spec integration (DOC_SPEC documents)
"""

from typing import Optional, List, Dict
import json
import pandas as pd

from aim.chat.strategy.base import ChatTurnStrategy, DEFAULT_MAX_CONTEXT, DEFAULT_MAX_OUTPUT
from aim.chat.manager import ChatManager
from aim.agents.persona import Persona
from aim.utils.xml import XmlFormatter
from aim.constants import DOC_SOURCE_CODE, DOC_SPEC
from aim_code.graph import CodeGraph, SymbolRef, generate_mermaid


class FocusRequest:
    """Request to focus on specific code.

    Represents an explicit focus on files and optionally line ranges,
    with parameters for call graph traversal depth and height.

    Attributes:
        files: File paths to focus on
        start_line: Optional start line to narrow focus
        end_line: Optional end line to narrow focus
        height: Levels UP the call graph (who calls these symbols)
        depth: Levels DOWN the call graph (what these symbols call)
    """

    def __init__(
        self,
        files: list[str],
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        height: int = 1,
        depth: int = 1,
    ):
        self.files = files
        self.start_line = start_line
        self.end_line = end_line
        self.height = height
        self.depth = depth


class XMLCodeTurnStrategy(ChatTurnStrategy):
    """Code-focused strategy with structural context.

    Extends ChatTurnStrategy to provide code-aware consciousness building.
    Works with CVM containing DOC_SOURCE_CODE and DOC_SPEC documents indexed
    by repo-watcher.

    The strategy supports:
    1. Explicit focus via FocusRequest (files, line ranges, graph traversal)
    2. Implicit semantic search based on query text
    3. Call graph visualization via Mermaid diagrams
    4. Module spec retrieval for design context

    Attributes:
        focus: Current FocusRequest or None
        code_graph: CodeGraph for call relationship traversal
    """

    def __init__(self, chat: ChatManager):
        """Initialize code strategy.

        Args:
            chat: ChatManager with CVM pointing to code index
        """
        super().__init__(chat)
        self.focus: Optional[FocusRequest] = None
        self.code_graph: Optional[CodeGraph] = None

    def set_code_graph(self, graph: CodeGraph) -> None:
        """Set the code graph for call graph traversal.

        Args:
            graph: CodeGraph loaded from edges.json
        """
        self.code_graph = graph

    def set_focus(self, focus: Optional[FocusRequest]) -> None:
        """Set current focus for consciousness building.

        Args:
            focus: FocusRequest to set, or None to clear
        """
        self.focus = focus

    def clear_focus(self) -> None:
        """Clear current focus."""
        self.focus = None

    def user_turn_for(
        self,
        persona: Persona,
        user_input: str,
        history: List[Dict[str, str]] = [],
    ) -> Dict[str, str]:
        """Generate a user turn for storage in history.

        Args:
            persona: The persona configuration
            user_input: The raw user input
            history: Prior conversation history

        Returns:
            User turn dict with role and content
        """
        return {"role": "user", "content": user_input}

    def chat_turns_for(
        self,
        persona: Persona,
        user_input: str,
        history: List[Dict[str, str]] = [],
        content_len: Optional[int] = None,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT,
    ) -> List[Dict[str, str]]:
        """Generate chat turns with code consciousness.

        Builds consciousness block with focused code, call graph, and
        semantic search results, then prepends to conversation.

        Args:
            persona: The persona configuration
            user_input: Current user input
            history: Prior conversation history
            content_len: Optional pre-calculated content length
            max_context_tokens: Maximum context window size
            max_output_tokens: Maximum output tokens

        Returns:
            List of chat turns including consciousness and history
        """
        # Build consciousness with code context
        consciousness, _ = self.get_code_consciousness(
            persona,
            user_input,
            max_context_tokens=max_context_tokens,
            max_output_tokens=max_output_tokens,
        )

        # Build turns
        turns = []
        if consciousness:
            turns.append(
                {"role": "user", "content": f"<consciousness>\n{consciousness}\n</consciousness>"}
            )
            turns.append({"role": "assistant", "content": "I understand the code context."})

        turns.extend(history)
        turns.append({"role": "user", "content": user_input})

        return turns

    def get_code_consciousness(
        self,
        persona: Persona,
        query: str,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT,
    ) -> tuple[str, int]:
        """Build code-aware consciousness block.

        Assembles consciousness from four sources:
        1. Focused Code (explicit) - Full source for files/line ranges
        2. Call Graph (explicit) - Mermaid diagram of relationships
        3. Relevant Code (implicit) - Semantic search results
        4. Module Specs - Design documentation for focused modules

        Args:
            persona: The persona configuration
            query: Query text for semantic search
            max_context_tokens: Maximum context window size
            max_output_tokens: Maximum output tokens

        Returns:
            Tuple of (consciousness_text, memory_count)
        """
        formatter = XmlFormatter()
        memory_count = 0

        # 1. Focused code (explicit)
        if self.focus:
            focused_source, focus_count = self._get_focused_source()
            if focused_source:
                formatter.add_element(
                    "code", "focused",
                    content=focused_source,
                    noindent=True,
                )
                memory_count += focus_count

            # 2. Call graph for focused symbols
            if self.code_graph:
                focused_symbols = self._get_focused_symbols()
                if focused_symbols:
                    edges = self.code_graph.get_neighborhood(
                        symbols=focused_symbols,
                        height=self.focus.height,
                        depth=self.focus.depth,
                    )
                    if edges:
                        mermaid = generate_mermaid(edges)
                        formatter.add_element(
                            "code", "call_graph",
                            content=f"```mermaid\n{mermaid}\n```",
                            noindent=True,
                        )

        # 3. Semantic code search (implicit)
        if query and self.chat.cvm:
            results = self.chat.cvm.query(
                query_texts=[query],
                query_document_type=DOC_SOURCE_CODE,
                top_n=5,
            )
            if not results.empty:
                snippets = self._format_search_results(results)
                formatter.add_element(
                    "code", "relevant",
                    content=snippets,
                    noindent=True,
                )
                memory_count += len(results)

        # 4. Module specs for focused files
        if self.focus and self.chat.cvm:
            module_paths = self._get_module_paths_for_files(self.focus.files)
            if module_paths:
                specs = self.chat.cvm.query(
                    query_texts=module_paths,
                    query_document_type=DOC_SPEC,
                    top_n=3,
                )
                if not specs.empty:
                    formatter.add_element(
                        "code", "specs",
                        content=self._format_specs(specs),
                        noindent=True,
                    )

        return formatter.render(), memory_count

    def _get_focused_source(self) -> tuple[str, int]:
        """Retrieve DOC_SOURCE_CODE for focused files/lines.

        Returns:
            Tuple of (formatted_source, doc_count)
        """
        if not self.focus or not self.chat.cvm:
            return "", 0

        results = []
        for file_path in self.focus.files:
            docs = self.chat.cvm.query(
                query_texts=[file_path],
                query_document_type=DOC_SOURCE_CODE,
                top_n=50,  # Get many docs for a file
            )
            if not docs.empty:
                # Filter by line range if specified
                if self.focus.start_line is not None or self.focus.end_line is not None:
                    docs = self._filter_by_line_range(docs)
                results.append(docs)

        if not results:
            return "", 0

        combined = pd.concat(results, ignore_index=True)
        return self._format_source_docs(combined), len(combined)

    def _filter_by_line_range(self, docs: pd.DataFrame) -> pd.DataFrame:
        """Filter docs by line range using metadata.

        Checks for overlap between document line range and focus line range.

        Args:
            docs: DataFrame with metadata column containing JSON

        Returns:
            Filtered DataFrame with only overlapping documents
        """
        filtered = []
        for _, row in docs.iterrows():
            meta = json.loads(row.get("metadata", "{}"))
            line_start = meta.get("line_start", 0)
            line_end = meta.get("line_end", float("inf"))

            # Check overlap with focus range
            if self.focus.end_line is not None and line_start > self.focus.end_line:
                continue
            if self.focus.start_line is not None and line_end < self.focus.start_line:
                continue
            filtered.append(row)

        return pd.DataFrame(filtered) if filtered else pd.DataFrame()

    def _get_focused_symbols(self) -> list[SymbolRef]:
        """Get symbol refs from focused docs for graph lookup.

        Returns:
            List of SymbolRef tuples (file_path, symbol, line_start)
        """
        if not self.focus or not self.chat.cvm:
            return []

        symbols = []
        for file_path in self.focus.files:
            docs = self.chat.cvm.query(
                query_texts=[file_path],
                query_document_type=DOC_SOURCE_CODE,
                top_n=50,
            )
            if self.focus.start_line is not None or self.focus.end_line is not None:
                docs = self._filter_by_line_range(docs)

            for _, row in docs.iterrows():
                meta = json.loads(row.get("metadata", "{}"))
                doc_id = row.get("doc_id", "")
                fp = doc_id.split("::")[0] if "::" in doc_id else file_path
                symbol = meta.get("symbol_name", "")
                parent = meta.get("parent_symbol")
                if parent:
                    symbol = f"{parent}.{symbol}"
                line = meta.get("line_start", 0)
                symbols.append((fp, symbol, line))

        return symbols

    def _format_source_docs(self, docs: pd.DataFrame) -> str:
        """Format source docs for consciousness.

        Args:
            docs: DataFrame with doc_id, content, metadata columns

        Returns:
            Formatted string with file::symbol headers and code blocks
        """
        parts = []
        for _, row in docs.iterrows():
            doc_id = row.get("doc_id", "")
            content = row.get("content", "")
            meta = json.loads(row.get("metadata", "{}"))
            line_start = meta.get("line_start", 0)
            parts.append(f"## {doc_id}:{line_start}\n```\n{content}\n```")
        return "\n\n".join(parts)

    def _format_search_results(self, docs: pd.DataFrame) -> str:
        """Format semantic search results.

        Args:
            docs: DataFrame with doc_id and content columns

        Returns:
            Formatted string with truncated code snippets
        """
        parts = []
        for _, row in docs.iterrows():
            doc_id = row.get("doc_id", "")
            content = row.get("content", "")[:500]  # Truncate for consciousness
            parts.append(f"### {doc_id}\n```\n{content}\n```")
        return "\n\n".join(parts)

    def _format_specs(self, specs: pd.DataFrame) -> str:
        """Format SPEC.md content.

        Args:
            specs: DataFrame with doc_id and content columns

        Returns:
            Formatted string with module headers
        """
        parts = []
        for _, row in specs.iterrows():
            doc_id = row.get("doc_id", "")
            content = row.get("content", "")[:1000]
            parts.append(f"### Module: {doc_id}\n{content}")
        return "\n\n".join(parts)

    def _get_module_paths_for_files(self, files: list[str]) -> list[str]:
        """Get module paths for focused files.

        Derives Python module paths from file paths for DOC_SPEC queries.

        Args:
            files: List of file paths

        Returns:
            List of derived module paths
        """
        paths = []
        for f in files:
            # Convert file path to module path guess
            parts = f.replace("/", ".").replace(".py", "").split(".")
            if "src" in parts:
                idx = parts.index("src")
                paths.append(".".join(parts[idx + 1 :]))
        return paths
