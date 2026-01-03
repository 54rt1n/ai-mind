# tests/unit/refiner/conftest.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Fixtures for refiner tests."""

import pytest
from pathlib import Path
from unittest.mock import patch


@pytest.fixture(autouse=True)
def patch_config_dir(monkeypatch):
    """
    Patch CONFIG_DIR in paradigm module to point to project root config.

    The aim-core package is installed in packages/aim-core/, but config files
    live at the project root in config/paradigm/. This fixture ensures tests
    can find the config files.
    """
    # Find project root (4 levels up from test file)
    test_file = Path(__file__)
    project_root = test_file.parent.parent.parent.parent.parent.parent
    config_dir = project_root / "config" / "paradigm"

    # Patch the CONFIG_DIR in paradigm module
    import aim.refiner.paradigm as paradigm_module
    monkeypatch.setattr(paradigm_module, "CONFIG_DIR", config_dir)
