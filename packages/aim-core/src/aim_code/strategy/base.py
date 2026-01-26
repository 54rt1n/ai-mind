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
from pathlib import Path
import json
import logging
import pandas as pd

from aim.chat.strategy.base import ChatTurnStrategy, DEFAULT_MAX_CONTEXT, DEFAULT_MAX_OUTPUT
from aim.chat.manager import ChatManager
from aim.agents.persona import Persona
from aim.utils.xml import XmlFormatter
from aim.utils.tokens import count_tokens as _count_tokens
from aim.constants import DOC_SOURCE_CODE, DOC_SPEC
from aim_code.graph import CodeGraph, SymbolRef, generate_mermaid

logger = logging.getLogger(__name__)

# Default token budget for consciousness (40% of default context)
DEFAULT_CONSCIOUSNESS_BUDGET = int(DEFAULT_MAX_CONTEXT * 0.4)


class FocusRequest:
    """Request to focus on specific code.

    Represents an explicit focus on files with per-file line ranges,
    with parameters for call graph traversal depth and height.

    Attributes:
        files: List of file specs, each a dict with:
            - path: File path to focus on
            - start: Optional start line (1-indexed, inclusive)
            - end: Optional end line (1-indexed, inclusive)
        height: Levels UP the call graph (who calls these symbols)
        depth: Levels DOWN the call graph (what these symbols call)

    Example:
        FocusRequest(
            files=[
                {"path": "model.py", "start": 10, "end": 50},
                {"path": "utils.py", "start": 100, "end": 200},
                {"path": "helpers.py"},  # Full file, no range
            ],
            height=2,
            depth=1,
        )
    """

    def __init__(
        self,
        files: list[dict],
        height: int = 1,
        depth: int = 1,
    ):
        self.files = files
        self.height = height
        self.depth = depth

    def get_file_paths(self) -> list[str]:
        """Return list of file paths from all file specs.

        Returns:
            List of file path strings.
        """
        return [f["path"] for f in self.files]

    def get_line_range(self, file_path: str) -> tuple[Optional[int], Optional[int]]:
        """Get the line range for a specific file.

        Args:
            file_path: The file path to look up.

        Returns:
            Tuple of (start_line, end_line), either may be None.
        """
        for f in self.files:
            if f["path"] == file_path:
                return f.get("start"), f.get("end")
        return None, None


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
        self.hud_name = "HUD Display Output"  # Same as XMLMemoryTurnStrategy
        self.focus: Optional[FocusRequest] = None
        self.code_graph: Optional[CodeGraph] = None
        self._graph_path: Optional[Path] = None

        # Interface compatibility with ProfileMixin thought injection
        self.thought_content: str = ""

    def count_tokens(self, text: str) -> int:
        """Count tokens using shared utility."""
        return _count_tokens(text)

    def _calc_max_context_tokens(self, max_context_tokens: int, max_output_tokens: int) -> int:
        """Calculate usable context tokens (reserve output + system prompt + safety margin).

        The system prompt is prepended to messages in the LLM provider's stream_turns(),
        so we must account for it here to avoid exceeding context limits.

        Args:
            max_context_tokens: Maximum context window size for the model.
            max_output_tokens: Maximum output tokens for the model.

        Returns:
            Usable context tokens after reservations.
        """
        system_tokens = 0
        if hasattr(self.chat, 'config') and self.chat.config:
            system_message = getattr(self.chat.config, 'system_message', None)
            if system_message and isinstance(system_message, str):
                system_tokens = self.count_tokens(system_message)
        return max_context_tokens - max_output_tokens - system_tokens - 1024

    def _trim_history(self, history: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
        """Trim history from oldest, keeping newest within budget.

        Iterates from newest to oldest, adding turns until budget is exhausted.
        This preserves the most recent context which is typically most relevant.

        Args:
            history: List of conversation turns.
            max_tokens: Maximum tokens allowed for history.

        Returns:
            Trimmed history list preserving newest turns.
        """
        if not history:
            return []

        if max_tokens <= 0:
            return []

        total = 0
        trimmed: List[Dict[str, str]] = []

        # Iterate from newest to oldest
        for turn in reversed(history):
            turn_tokens = self.count_tokens(turn.get("content", ""))
            if total + turn_tokens > max_tokens:
                break
            trimmed.insert(0, turn)
            total += turn_tokens

        return trimmed

    def set_code_graph(self, graph: CodeGraph, graph_path: Optional[Path] = None) -> None:
        """Set the code graph for call graph traversal.

        Args:
            graph: CodeGraph loaded from edges.json
            graph_path: Path to graph directory for hot reloading (optional)
        """
        self.code_graph = graph
        self._graph_path = graph_path

    def reload_code_graph(self) -> bool:
        """Reload the code graph from disk.

        Called at the start of each turn to pick up any code changes.
        This is cheap (just JSON file load) so safe to call frequently.

        Returns:
            True if reload succeeded, False if no path set or load failed.
        """
        if not self._graph_path:
            return False
        try:
            self.code_graph = CodeGraph.load(self._graph_path)
            logger.debug(f"Reloaded code graph from {self._graph_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to reload code graph: {e}")
            return False

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

        Budget allocation:
        - 40% for consciousness (focused code, call graph, semantic search)
        - 50% for history (trimmed from oldest if over budget)
        - 10% safety margin for user input and structural tokens

        Args:
            persona: The persona configuration
            user_input: Current user input
            history: Prior conversation history
            content_len: Optional pre-calculated content length (unused, kept for API compatibility)
            max_context_tokens: Maximum context window size
            max_output_tokens: Maximum output tokens

        Returns:
            List of chat turns including consciousness and history
        """
        # Calculate usable context budget
        usable_tokens = self._calc_max_context_tokens(max_context_tokens, max_output_tokens)

        # Budget allocation: 40% consciousness, 50% history, 10% safety
        consciousness_budget = int(usable_tokens * 0.4)
        history_budget = int(usable_tokens * 0.5)

        logger.debug(
            f"Token budget: usable={usable_tokens}, "
            f"consciousness={consciousness_budget}, history={history_budget}"
        )

        # Trim history to fit budget (keeps newest)
        trimmed_history = self._trim_history(history, history_budget)
        if len(trimmed_history) < len(history):
            logger.info(
                f"Trimmed history from {len(history)} to {len(trimmed_history)} turns "
                f"to fit {history_budget} token budget"
            )

        # Extract meaningful content from conversation history for semantic search
        # This captures actual user questions like "help me fix the parser"
        user_messages = [h["content"] for h in trimmed_history if h.get("role") == "user"]
        assistant_messages = [h["content"] for h in trimmed_history if h.get("role") == "assistant"]

        # Build query prioritizing thought content over generic templates
        query_texts = []

        # 1. Thought content (highest priority - semantic reasoning)
        if self.thought_content and self.thought_content.strip():
            query_texts.append(self.thought_content.strip())
            logger.debug("Added thought_content to query (%d chars)", len(self.thought_content))

        # 2. User input (unless it's the REASONING_PROMPT template)
        if user_input and not user_input.strip().startswith("[~~ Thought Turn ~~]"):
            query_texts.append(user_input)
        elif user_input and user_input.strip().startswith("[~~ Thought Turn ~~]"):
            logger.debug("Excluded REASONING_PROMPT from semantic query")

        # 3. Recent history
        query_texts.extend(user_messages[-3:])
        query_texts.extend(assistant_messages[-2:])

        query = " ".join(query_texts)

        # Build consciousness with code context, respecting token budget
        consciousness, memory_count = self.get_code_consciousness(
            persona,
            query,
            max_context_tokens=max_context_tokens,
            max_output_tokens=max_output_tokens,
            token_budget=consciousness_budget,
        )

        # Build turns
        turns: List[Dict[str, str]] = []
        # Only add consciousness if it has actual content (not just empty wrapper)
        if consciousness and memory_count > 0:
            turns.append(
                {"role": "user", "content": f"<consciousness>\n{consciousness}\n</consciousness>"}
            )
            turns.append({"role": "assistant", "content": "I understand the code context."})

        turns.extend(trimmed_history)
        turns.append({"role": "user", "content": user_input})

        return turns

    def get_code_consciousness(
        self,
        persona: Persona,
        query: str,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT,
        token_budget: int = DEFAULT_CONSCIOUSNESS_BUDGET,
    ) -> tuple[str, int]:
        """Build code-aware consciousness block.

        Assembles consciousness from four sources with token budgeting:
        1. Focused Code (explicit) - Full source for files/line ranges (60% of budget)
        2. Call Graph (explicit) - Mermaid diagram of relationships (small fixed allocation)
        3. Relevant Code (implicit) - Semantic search results (remaining budget)
        4. Module Specs - Design documentation for focused modules (small fixed allocation)

        Priority order ensures focused code gets most space, with semantic search
        filling remaining budget.

        Args:
            persona: The persona configuration
            query: Query text for semantic search
            max_context_tokens: Maximum context window size (unused, kept for API compatibility)
            max_output_tokens: Maximum output tokens (unused, kept for API compatibility)
            token_budget: Maximum tokens for consciousness content

        Returns:
            Tuple of (consciousness_text, memory_count)
        """
        formatter = XmlFormatter()
        memory_count = 0
        tokens_used = 0

        # Budget allocation for consciousness sections
        # Focused code gets 60%, call graph 10%, specs 10%, semantic search 20%
        focused_budget = int(token_budget * 0.6)
        call_graph_budget = int(token_budget * 0.1)
        specs_budget = int(token_budget * 0.1)

        # 1. Focused code (explicit) - highest priority, gets most budget
        if self.focus:
            focused_source, focus_count = self._get_focused_source(max_tokens=focused_budget)
            if focused_source:
                focused_tokens = self.count_tokens(focused_source)
                formatter.add_element(
                    "code", "focused",
                    content=focused_source,
                    noindent=True,
                )
                memory_count += focus_count
                tokens_used += focused_tokens
                logger.debug(f"Focused code: {focused_tokens} tokens, {focus_count} docs")

            # 2. Call graph for focused symbols (small allocation)
            if self.code_graph and tokens_used < token_budget:
                focused_symbols = self._get_focused_symbols()
                if focused_symbols:
                    edges = self.code_graph.get_neighborhood(
                        symbols=focused_symbols,
                        height=self.focus.height,
                        depth=self.focus.depth,
                    )
                    if edges:
                        mermaid = generate_mermaid(edges)
                        mermaid_content = f"```mermaid\n{mermaid}\n```"
                        mermaid_tokens = self.count_tokens(mermaid_content)

                        # Only include if within call graph budget
                        if mermaid_tokens <= call_graph_budget:
                            formatter.add_element(
                                "code", "call_graph",
                                content=mermaid_content,
                                noindent=True,
                            )
                            tokens_used += mermaid_tokens
                            logger.debug(f"Call graph: {mermaid_tokens} tokens")

        # 3. Module specs for focused files (small allocation)
        if self.focus and self.chat.cvm and tokens_used < token_budget:
            module_paths = self._get_module_paths_for_files(self.focus.get_file_paths())
            if module_paths:
                specs = self.chat.cvm.query(
                    query_texts=module_paths,
                    query_document_type=DOC_SPEC,
                    top_n=3,
                )
                if not specs.empty:
                    specs_content = self._format_specs(specs, max_tokens=specs_budget)
                    specs_tokens = self.count_tokens(specs_content)
                    if specs_tokens > 0:
                        formatter.add_element(
                            "code", "specs",
                            content=specs_content,
                            noindent=True,
                        )
                        tokens_used += specs_tokens
                        logger.debug(f"Module specs: {specs_tokens} tokens")

        # 4. Semantic code search (implicit) - uses remaining budget
        remaining_budget = token_budget - tokens_used
        if query and self.chat.cvm and remaining_budget > 500:
            # Limit results based on remaining budget (~500 tokens per result estimate)
            max_results = max(1, min(5, remaining_budget // 500))
            results = self.chat.cvm.query(
                query_texts=[query],
                query_document_type=DOC_SOURCE_CODE,
                top_n=max_results,
            )
            if not results.empty:
                snippets = self._format_search_results(results, max_tokens=remaining_budget)
                snippets_tokens = self.count_tokens(snippets)
                if snippets_tokens > 0:
                    formatter.add_element(
                        "code", "relevant",
                        content=snippets,
                        noindent=True,
                    )
                    memory_count += len(results)
                    tokens_used += snippets_tokens
                    logger.debug(f"Semantic search: {snippets_tokens} tokens, {len(results)} results")

        logger.debug(f"Total consciousness: {tokens_used}/{token_budget} tokens, {memory_count} memories")
        return formatter.render(), memory_count

    def _get_focused_source(self, max_tokens: int = 8000) -> tuple[str, int]:
        """Retrieve DOC_SOURCE_CODE for focused files/lines within token budget.

        Fetches documents for focused files, filters by per-file line range
        if specified, then truncates to fit within the token budget.

        Args:
            max_tokens: Maximum tokens for the returned content.

        Returns:
            Tuple of (formatted_source, doc_count)
        """
        if not self.focus or not self.chat.cvm:
            return "", 0

        if max_tokens <= 0:
            return "", 0

        results = []
        for file_spec in self.focus.files:
            file_path = file_spec["path"]
            start_line = file_spec.get("start")
            end_line = file_spec.get("end")

            docs = self.chat.cvm.query(
                query_texts=[file_path],
                query_document_type=DOC_SOURCE_CODE,
                top_n=50,  # Get many docs for a file
            )
            if not docs.empty:
                # Filter by line range if specified for this file
                if start_line is not None or end_line is not None:
                    docs = self._filter_by_line_range(docs, start_line, end_line)
                results.append(docs)

        if not results:
            return "", 0

        combined = pd.concat(results, ignore_index=True)
        return self._format_source_docs(combined, max_tokens=max_tokens)

    def _filter_by_line_range(
        self,
        docs: pd.DataFrame,
        focus_start: Optional[int],
        focus_end: Optional[int],
    ) -> pd.DataFrame:
        """Filter docs by line range using metadata.

        Checks for overlap between document line range and focus line range.

        Args:
            docs: DataFrame with metadata column containing JSON
            focus_start: Start line to filter by (inclusive), or None for no lower bound
            focus_end: End line to filter by (inclusive), or None for no upper bound

        Returns:
            Filtered DataFrame with only overlapping documents
        """
        filtered = []
        for _, row in docs.iterrows():
            meta = json.loads(row.get("metadata", "{}"))
            line_start = meta.get("line_start", 0)
            line_end = meta.get("line_end", float("inf"))

            # Check overlap with focus range
            if focus_end is not None and line_start > focus_end:
                continue
            if focus_start is not None and line_end < focus_start:
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
        for file_spec in self.focus.files:
            file_path = file_spec["path"]
            start_line = file_spec.get("start")
            end_line = file_spec.get("end")

            docs = self.chat.cvm.query(
                query_texts=[file_path],
                query_document_type=DOC_SOURCE_CODE,
                top_n=50,
            )
            if start_line is not None or end_line is not None:
                docs = self._filter_by_line_range(docs, start_line, end_line)

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

    def _format_source_docs(self, docs: pd.DataFrame, max_tokens: int = 8000) -> tuple[str, int]:
        """Format source docs for consciousness within token budget.

        Iterates through documents, adding each until the token budget is exhausted.
        Documents are added in order (as returned from query), prioritizing
        earlier documents.

        Args:
            docs: DataFrame with doc_id, content, metadata columns
            max_tokens: Maximum tokens for the output.

        Returns:
            Tuple of (formatted string with file::symbol headers and code blocks, doc_count)
        """
        parts = []
        tokens_used = 0
        doc_count = 0

        for _, row in docs.iterrows():
            doc_id = row.get("doc_id", "")
            content = row.get("content", "")
            meta = json.loads(row.get("metadata", "{}"))
            line_start = meta.get("line_start", 0)

            part = f"## {doc_id}:{line_start}\n```\n{content}\n```"
            part_tokens = self.count_tokens(part)

            # Check if adding this part would exceed budget
            if tokens_used + part_tokens > max_tokens:
                # If we have no parts yet, truncate this one to fit
                if not parts:
                    # Estimate chars from remaining tokens (rough: 4 chars per token)
                    remaining_tokens = max_tokens - tokens_used
                    max_chars = remaining_tokens * 4
                    truncated_content = content[:max_chars] if len(content) > max_chars else content
                    if truncated_content:
                        part = f"## {doc_id}:{line_start}\n```\n{truncated_content}...\n```"
                        parts.append(part)
                        doc_count += 1
                break

            parts.append(part)
            tokens_used += part_tokens
            doc_count += 1

        return "\n\n".join(parts), doc_count

    def _format_search_results(self, docs: pd.DataFrame, max_tokens: int = 2000) -> str:
        """Format semantic search results within token budget.

        Iterates through search results, truncating each snippet and stopping
        when the token budget is exhausted.

        Args:
            docs: DataFrame with doc_id and content columns
            max_tokens: Maximum tokens for the output.

        Returns:
            Formatted string with truncated code snippets
        """
        parts = []
        tokens_used = 0
        # Per-snippet char limit (roughly 500 chars = ~125 tokens)
        snippet_char_limit = 500

        for _, row in docs.iterrows():
            doc_id = row.get("doc_id", "")
            content = row.get("content", "")[:snippet_char_limit]
            part = f"### {doc_id}\n```\n{content}\n```"
            part_tokens = self.count_tokens(part)

            if tokens_used + part_tokens > max_tokens:
                break

            parts.append(part)
            tokens_used += part_tokens

        return "\n\n".join(parts)

    def _format_specs(self, specs: pd.DataFrame, max_tokens: int = 1000) -> str:
        """Format SPEC.md content within token budget.

        Iterates through specs, truncating each and stopping when the
        token budget is exhausted.

        Args:
            specs: DataFrame with doc_id and content columns
            max_tokens: Maximum tokens for the output.

        Returns:
            Formatted string with module headers
        """
        parts = []
        tokens_used = 0
        # Per-spec char limit (roughly 1000 chars = ~250 tokens)
        spec_char_limit = 1000

        for _, row in specs.iterrows():
            doc_id = row.get("doc_id", "")
            content = row.get("content", "")[:spec_char_limit]
            part = f"### Module: {doc_id}\n{content}"
            part_tokens = self.count_tokens(part)

            if tokens_used + part_tokens > max_tokens:
                break

            parts.append(part)
            tokens_used += part_tokens

        return "\n\n".join(parts)

    def _get_module_paths_for_files(self, file_paths: list[str]) -> list[str]:
        """Get module paths for focused files.

        Derives Python module paths from file paths for DOC_SPEC queries.

        Args:
            file_paths: List of file paths

        Returns:
            List of derived module paths
        """
        paths = []
        for f in file_paths:
            # Convert file path to module path guess
            parts = f.replace("/", ".").replace(".py", "").split(".")
            if "src" in parts:
                idx = parts.index("src")
                paths.append(".".join(parts[idx + 1 :]))
        return paths
