# aim_code/graph/module_registry.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Module name to file path mapping.

Maps Python module names (e.g., "aim.config") to their file paths
(e.g., "packages/aim-core/src/aim/config.py") for import resolution.
"""

from typing import Optional


class ModuleRegistry:
    """Maps Python module names to file paths.

    Built during Pass 1 by deriving module name from file path.
    Used during Pass 2 to resolve imports to file paths.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._module_to_file: dict[str, str] = {}

    def add(self, module_name: str, file_path: str) -> None:
        """Register a module -> file_path mapping.

        Args:
            module_name: Python module name (e.g., "aim.config")
            file_path: Path to the Python file
        """
        self._module_to_file[module_name] = file_path

    def get_file_path(self, module_name: str) -> Optional[str]:
        """Look up file path for a module name.

        Args:
            module_name: Python module name to look up

        Returns:
            File path if registered, None otherwise.
        """
        return self._module_to_file.get(module_name)

    @staticmethod
    def file_to_module(file_path: str, source_root: str) -> str:
        """Derive module name from file path.

        Args:
            file_path: Full path to Python file
            source_root: Root of source tree (stripped from path)

        Returns:
            Python module name derived from path.

        Example:
            file_path: "packages/aim-core/src/aim/config.py"
            source_root: "packages/aim-core/src"
            returns: "aim.config"
        """
        # Strip source root prefix
        rel_path = file_path
        prefix = source_root + "/"
        if file_path.startswith(prefix):
            rel_path = file_path[len(prefix) :]

        # Remove .py extension
        if rel_path.endswith(".py"):
            rel_path = rel_path[:-3]

        # Convert path separators to dots
        module = rel_path.replace("/", ".")

        # Handle __init__.py -> parent module
        if module.endswith(".__init__"):
            module = module[:-9]

        return module
