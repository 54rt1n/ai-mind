# aim/refiner/paradigm_config.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Paradigm configuration loader.

Loads paradigm configs from config/paradigm/*.yaml files.
Each paradigm defines its own document types, queries, and validation logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Default config directory
CONFIG_DIR = Path(__file__).parent.parent.parent / "config" / "paradigm"


@dataclass
class ParadigmConfig:
    """Configuration for a single paradigm."""

    name: str
    aspect: str
    doc_types: List[str]
    queries: List[dict]
    prior_work_doc_types: List[str]
    think: str
    instructions: str
    approach_doc_types: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "ParadigmConfig":
        """Load a paradigm config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Handle approach_doc_types - can be a list or a dict
        approach_doc_types = data.get("approach_doc_types", {})
        if isinstance(approach_doc_types, list):
            # Single list means same doc types for all approaches
            approach_doc_types = {"default": approach_doc_types}

        return cls(
            name=data["name"],
            aspect=data["aspect"],
            doc_types=data.get("doc_types", []),
            queries=data.get("queries", []),
            prior_work_doc_types=data.get("prior_work_doc_types", []),
            think=data.get("think", ""),
            instructions=data.get("instructions", ""),
            approach_doc_types=approach_doc_types,
        )

    def get_approach_doc_types(self, approach: str) -> List[str]:
        """Get document types for a specific approach."""
        if approach in self.approach_doc_types:
            return self.approach_doc_types[approach]
        if "default" in self.approach_doc_types:
            return self.approach_doc_types["default"]
        return self.doc_types


class ParadigmConfigLoader:
    """Loads and caches paradigm configurations."""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or CONFIG_DIR
        self._cache: Dict[str, ParadigmConfig] = {}

    def load(self, paradigm: str) -> Optional[ParadigmConfig]:
        """Load a paradigm config by name."""
        if paradigm in self._cache:
            return self._cache[paradigm]

        path = self.config_dir / f"{paradigm}.yaml"
        if not path.exists():
            logger.warning(f"No config found for paradigm '{paradigm}' at {path}")
            return None

        try:
            config = ParadigmConfig.from_yaml(path)
            self._cache[paradigm] = config
            return config
        except Exception as e:
            logger.error(f"Error loading paradigm config '{paradigm}': {e}")
            return None

    def load_all(self) -> Dict[str, ParadigmConfig]:
        """Load all paradigm configs from the config directory."""
        configs = {}
        if not self.config_dir.exists():
            logger.warning(f"Paradigm config directory does not exist: {self.config_dir}")
            return configs

        for path in self.config_dir.glob("*.yaml"):
            paradigm = path.stem
            config = self.load(paradigm)
            if config:
                configs[paradigm] = config

        return configs


# Module-level loader instance
_loader: Optional[ParadigmConfigLoader] = None


def get_loader() -> ParadigmConfigLoader:
    """Get or create the module-level loader."""
    global _loader
    if _loader is None:
        _loader = ParadigmConfigLoader()
    return _loader


def get_paradigm_config(paradigm: str) -> Optional[ParadigmConfig]:
    """Get a paradigm config by name."""
    return get_loader().load(paradigm)
