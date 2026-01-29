# tests/core_tests/unit/aim_code/graph/test_module_registry.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for ModuleRegistry - module name to file path mapping."""


class TestModuleRegistryBasics:
    """Tests for basic add/get operations."""

    def test_empty_registry_returns_none(self):
        """get_file_path should return None for unknown modules."""
        from aim_code.graph import ModuleRegistry

        registry = ModuleRegistry()

        assert registry.get_file_path("unknown.module") is None

    def test_add_and_get_module(self):
        """add should register mapping, get_file_path should retrieve it."""
        from aim_code.graph import ModuleRegistry

        registry = ModuleRegistry()
        registry.add("aim.config", "packages/aim-core/src/aim/config.py")

        assert registry.get_file_path("aim.config") == "packages/aim-core/src/aim/config.py"

    def test_add_multiple_modules(self):
        """Registry should store multiple module mappings."""
        from aim_code.graph import ModuleRegistry

        registry = ModuleRegistry()
        registry.add("aim.config", "packages/aim-core/src/aim/config.py")
        registry.add("aim.utils", "packages/aim-core/src/aim/utils.py")

        assert registry.get_file_path("aim.config") == "packages/aim-core/src/aim/config.py"
        assert registry.get_file_path("aim.utils") == "packages/aim-core/src/aim/utils.py"

    def test_add_overwrites_existing(self):
        """Adding same module again should overwrite the path."""
        from aim_code.graph import ModuleRegistry

        registry = ModuleRegistry()
        registry.add("aim.config", "old/path.py")
        registry.add("aim.config", "new/path.py")

        assert registry.get_file_path("aim.config") == "new/path.py"


class TestModuleRegistryFileToModule:
    """Tests for the file_to_module static method."""

    def test_simple_file_to_module(self):
        """Should convert simple file path to module name."""
        from aim_code.graph import ModuleRegistry

        result = ModuleRegistry.file_to_module(
            "packages/aim-core/src/aim/config.py",
            "packages/aim-core/src"
        )

        assert result == "aim.config"

    def test_nested_module_path(self):
        """Should handle deeply nested paths."""
        from aim_code.graph import ModuleRegistry

        result = ModuleRegistry.file_to_module(
            "packages/aim-core/src/aim/chat/strategy/xmlmemory.py",
            "packages/aim-core/src"
        )

        assert result == "aim.chat.strategy.xmlmemory"

    def test_init_file_becomes_package(self):
        """__init__.py should resolve to parent module name."""
        from aim_code.graph import ModuleRegistry

        result = ModuleRegistry.file_to_module(
            "packages/aim-core/src/aim/chat/__init__.py",
            "packages/aim-core/src"
        )

        assert result == "aim.chat"

    def test_root_init_file(self):
        """__init__.py at root should resolve to package name."""
        from aim_code.graph import ModuleRegistry

        result = ModuleRegistry.file_to_module(
            "packages/aim-core/src/aim/__init__.py",
            "packages/aim-core/src"
        )

        assert result == "aim"

    def test_handles_missing_prefix(self):
        """Should work even if file_path doesn't start with source_root."""
        from aim_code.graph import ModuleRegistry

        result = ModuleRegistry.file_to_module(
            "aim/config.py",
            "packages/aim-core/src"
        )

        # Falls back to using the full path
        assert result == "aim.config"

    def test_single_file_module(self):
        """Should handle file directly in source root."""
        from aim_code.graph import ModuleRegistry

        result = ModuleRegistry.file_to_module(
            "packages/aim-core/src/constants.py",
            "packages/aim-core/src"
        )

        assert result == "constants"
