# aim_code/parsers/registry.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Parser registry for managing language-specific parsers in CODE_RAG."""

from typing import Optional
import warnings

from .base import BaseParser

# Import parsers with graceful fallback for missing implementations
_parser_imports: dict[str, type[BaseParser] | None] = {}

try:
    from .python_parser import PythonParser

    _parser_imports["python"] = PythonParser
except ImportError as e:
    warnings.warn(f"PythonParser not available: {e}")
    _parser_imports["python"] = None

try:
    from .typescript_parser import TypeScriptParser

    _parser_imports["typescript"] = TypeScriptParser
except ImportError as e:
    warnings.warn(f"TypeScriptParser not available: {e}")
    _parser_imports["typescript"] = None

try:
    from .bash_parser import BashParser

    _parser_imports["bash"] = BashParser
except ImportError as e:
    warnings.warn(f"BashParser not available: {e}")
    _parser_imports["bash"] = None


class ParserRegistry:
    """Registry for managing CODE_RAG language parsers.

    Initializes and caches parser instances, providing access by language name.
    Handles parser loading failures gracefully with warnings.
    """

    def __init__(self):
        """Initialize parser registry and load all available parsers."""
        self._parsers: dict[str, BaseParser] = {}
        self._load_parsers()

    def _load_parsers(self) -> None:
        """Load all available language parsers.

        Iterates through imported parser classes, instantiates each,
        and registers those that are available (tree-sitter loaded successfully).
        """
        for lang_hint, parser_class in _parser_imports.items():
            if parser_class is None:
                continue

            try:
                parser = parser_class()
                if parser.is_available():
                    lang_name = parser.get_language_name()
                    self._parsers[lang_name] = parser
                else:
                    warnings.warn(
                        f"{parser_class.__name__} loaded but tree-sitter not available"
                    )
            except Exception as e:
                warnings.warn(f"Failed to load {parser_class.__name__}: {e}")

    def get_parser(self, language: str) -> Optional[BaseParser]:
        """Get parser for a specific language.

        Args:
            language: Language name (e.g., 'python', 'typescript').
                      Case-insensitive.

        Returns:
            BaseParser instance if available, None otherwise.
        """
        return self._parsers.get(language.lower())

    def has_parser(self, language: str) -> bool:
        """Check if a parser exists for the specified language.

        Args:
            language: Language name. Case-insensitive.

        Returns:
            True if parser is available, False otherwise.
        """
        return language.lower() in self._parsers

    def list_available_languages(self) -> list[str]:
        """Get list of languages with available parsers.

        Returns:
            Sorted list of language names.
        """
        return sorted(self._parsers.keys())

    def __repr__(self) -> str:
        """String representation showing available parsers."""
        langs = ", ".join(self.list_available_languages())
        return f"ParserRegistry({len(self._parsers)} parsers: {langs})"
