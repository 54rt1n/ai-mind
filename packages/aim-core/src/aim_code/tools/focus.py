# aim_code/tools/focus.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
FocusTool: Tool for setting code focus context.

Provides explicit focus on files and line ranges with call graph traversal
parameters. Focus persists across turns until explicitly cleared.

Supports resolution of entity names to file paths via metadata, allowing
agents to focus on files by their entity name (e.g., "model.py") rather
than full path (e.g., "/repo/src/model.py").
"""

from typing import Optional, Dict, Any, List, TYPE_CHECKING
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

    Supports entity name resolution: when entities are provided, file names
    without "/" are resolved to full paths via entity metadata.

    Attributes:
        strategy: XMLCodeTurnStrategy to set focus on
    """

    def __init__(self, strategy: "XMLCodeTurnStrategy"):
        """Initialize focus tool with strategy reference.

        Args:
            strategy: XMLCodeTurnStrategy instance to control
        """
        self.strategy = strategy

    def _resolve_file_name(self, name: str, entities: List[Any]) -> str:
        """Resolve an entity name to a file path via metadata.

        Searches entities for one with a matching name and returns the
        file_path from its metadata if available.

        Args:
            name: Entity name to resolve (e.g., "model.py")
            entities: List of EntityState objects with metadata

        Returns:
            Resolved file path, or original name if not found
        """
        for entity in entities:
            # Handle both dict and object forms
            if isinstance(entity, dict):
                entity_name = entity.get("name", "")
                metadata = entity.get("metadata", {})
            else:
                entity_name = getattr(entity, "name", "")
                metadata = getattr(entity, "metadata", {})

            if entity_name == name:
                file_path = metadata.get("file_path")
                if file_path:
                    return file_path
        return name

    def _resolve_files(self, files: List[str], entities: List[Any]) -> List[str]:
        """Resolve file names to paths using entity metadata.

        Files containing "/" are treated as paths and passed through unchanged.
        Files without "/" are treated as entity names and resolved via metadata.

        Args:
            files: List of file paths or entity names
            entities: List of EntityState objects with metadata

        Returns:
            List of resolved file paths
        """
        if not entities:
            return files

        resolved = []
        for f in files:
            if "/" in f:
                # Already a path
                resolved.append(f)
            else:
                # Try to resolve from entities
                resolved.append(self._resolve_file_name(f, entities))
        return resolved

    def focus(
        self,
        files: list[str],
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        height: int = 1,
        depth: int = 1,
        entities: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Set focus on specific files or line ranges.

        Creates a FocusRequest and sets it on the strategy. Focus persists
        across turns until clear_focus() is called or new focus is set.

        File names without "/" are resolved to full paths via entity metadata
        when entities are provided. This allows focusing on files by their
        entity name (e.g., "model.py") rather than full path.

        Note: This API accepts a global start_line/end_line for backward
        compatibility with tool definitions. These are applied to ALL files
        in the focus request. For per-file ranges, use focus_files() instead.

        Args:
            files: File paths or entity names to focus on
            start_line: Optional start line to apply to all files
            end_line: Optional end line to apply to all files
            height: Levels UP the call graph (who calls these symbols)
            depth: Levels DOWN the call graph (what these symbols call)
            entities: Optional list of EntityState objects for name resolution

        Returns:
            Dict with success status and focus preview information
        """
        # Resolve entity names to file paths
        resolved_paths = self._resolve_files(files, entities or [])

        # Build file specs - apply global range to all files
        file_specs = []
        for path in resolved_paths:
            spec: Dict[str, Any] = {"path": path}
            if start_line is not None:
                spec["start"] = start_line
            if end_line is not None:
                spec["end"] = end_line
            file_specs.append(spec)

        request = FocusRequest(
            files=file_specs,
            height=height,
            depth=depth,
        )

        self.strategy.set_focus(request)

        # Build preview
        symbols = self._get_focused_symbols(request)

        return {
            "success": True,
            "focused_files": resolved_paths,
            "line_range": f"{start_line or 'start'}-{end_line or 'end'}",
            "symbols_in_focus": len(symbols),
            "graph_height": height,
            "graph_depth": depth,
        }

    def focus_files(
        self,
        file_specs: list[dict],
        height: int = 1,
        depth: int = 1,
        entities: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Set focus on specific files with per-file line ranges.

        Creates a FocusRequest with per-file line ranges. Focus persists
        across turns until clear_focus() is called or new focus is set.

        Args:
            file_specs: List of file specs, each a dict with:
                - path: File path or entity name
                - start: Optional start line (1-indexed, inclusive)
                - end: Optional end line (1-indexed, inclusive)
            height: Levels UP the call graph (who calls these symbols)
            depth: Levels DOWN the call graph (what these symbols call)
            entities: Optional list of EntityState objects for name resolution

        Returns:
            Dict with success status and focus preview information
        """
        # Resolve entity names in file specs
        resolved_specs = []
        for spec in file_specs:
            path = spec.get("path", "")
            if "/" not in path:
                path = self._resolve_file_name(path, entities or [])
            resolved_spec: Dict[str, Any] = {"path": path}
            if "start" in spec:
                resolved_spec["start"] = spec["start"]
            if "end" in spec:
                resolved_spec["end"] = spec["end"]
            resolved_specs.append(resolved_spec)

        request = FocusRequest(
            files=resolved_specs,
            height=height,
            depth=depth,
        )

        self.strategy.set_focus(request)

        # Build preview
        symbols = self._get_focused_symbols(request)
        file_paths = [s["path"] for s in resolved_specs]

        # Format line ranges for display
        range_displays = []
        for spec in resolved_specs:
            path = spec["path"]
            start = spec.get("start")
            end = spec.get("end")
            if start is not None and end is not None:
                range_displays.append(f"{path}:{start}-{end}")
            elif start is not None:
                range_displays.append(f"{path}:{start}-end")
            elif end is not None:
                range_displays.append(f"{path}:start-{end}")
            else:
                range_displays.append(path)

        return {
            "success": True,
            "focused_files": file_paths,
            "file_ranges": range_displays,
            "symbols_in_focus": len(symbols),
            "graph_height": height,
            "graph_depth": depth,
        }

    def _get_focused_symbols(self, request: FocusRequest) -> list[str]:
        """Get symbol names in focused range.

        Queries CVM for DOC_SOURCE_CODE documents matching the focused files
        and filters by per-file line range if specified.

        Args:
            request: FocusRequest with files and optional line ranges

        Returns:
            List of symbol names found in focused range
        """
        if not self.strategy.chat.cvm:
            return []

        symbols = []
        for file_spec in request.files:
            file_path = file_spec["path"]
            focus_start = file_spec.get("start")
            focus_end = file_spec.get("end")

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
                if focus_start is not None and line_end < focus_start:
                    continue
                if focus_end is not None and line_start > focus_end:
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
