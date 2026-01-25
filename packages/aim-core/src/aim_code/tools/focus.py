# aim_code/tools/focus.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
FocusTool: Tool for setting code focus context.

Provides explicit focus on files and line ranges with call graph traversal
parameters. Focus persists across turns until explicitly cleared.
"""

from typing import Optional, Dict, Any, TYPE_CHECKING
import json

from aim.constants import DOC_SOURCE_CODE

if TYPE_CHECKING:
    from aim_code.strategy.base import XMLCodeTurnStrategy

from aim_code.strategy.base import FocusRequest


class FocusTool:
    """Tool for setting code focus context.

    Used by code agents to explicitly focus on specific files or line ranges.
    When focus is set, the XMLCodeTurnStrategy includes full source code
    and call graph context in consciousness.

    Attributes:
        strategy: XMLCodeTurnStrategy to set focus on
    """

    def __init__(self, strategy: "XMLCodeTurnStrategy"):
        """Initialize focus tool with strategy reference.

        Args:
            strategy: XMLCodeTurnStrategy instance to control
        """
        self.strategy = strategy

    def focus(
        self,
        files: list[str],
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        height: int = 1,
        depth: int = 1,
    ) -> Dict[str, Any]:
        """Set focus on specific files or line ranges.

        Creates a FocusRequest and sets it on the strategy. Focus persists
        across turns until clear_focus() is called or new focus is set.

        Args:
            files: File paths to focus on
            start_line: Optional start line to narrow focus
            end_line: Optional end line to narrow focus
            height: Levels UP the call graph (who calls these symbols)
            depth: Levels DOWN the call graph (what these symbols call)

        Returns:
            Dict with success status and focus preview information
        """
        request = FocusRequest(
            files=files,
            start_line=start_line,
            end_line=end_line,
            height=height,
            depth=depth,
        )

        self.strategy.set_focus(request)

        # Build preview
        symbols = self._get_focused_symbols(request)

        return {
            "success": True,
            "focused_files": files,
            "line_range": f"{start_line or 'start'}-{end_line or 'end'}",
            "symbols_in_focus": len(symbols),
            "graph_height": height,
            "graph_depth": depth,
        }

    def _get_focused_symbols(self, request: FocusRequest) -> list[str]:
        """Get symbol names in focused range.

        Queries CVM for DOC_SOURCE_CODE documents matching the focused files
        and filters by line range if specified.

        Args:
            request: FocusRequest with files and optional line range

        Returns:
            List of symbol names found in focused range
        """
        if not self.strategy.chat.cvm:
            return []

        symbols = []
        for file_path in request.files:
            docs = self.strategy.chat.cvm.query(
                query_texts=[file_path],
                query_document_type=DOC_SOURCE_CODE,
                top_n=50,
            )
            for _, row in docs.iterrows():
                meta = json.loads(row.get("metadata", "{}"))
                line_start = meta.get("line_start", 0)
                line_end = meta.get("line_end", float("inf"))

                # Filter by line range
                if request.start_line is not None and line_end < request.start_line:
                    continue
                if request.end_line is not None and line_start > request.end_line:
                    continue
                symbols.append(meta.get("symbol_name", ""))
        return symbols

    def clear_focus(self) -> Dict[str, Any]:
        """Clear current focus.

        Returns:
            Dict with success status and confirmation message
        """
        self.strategy.clear_focus()
        return {"success": True, "message": "Focus cleared"}
