# tests/conftest.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Root conftest for aim-core tests.

Ensures the src directory is in the Python path for all tests.
Provides fixtures for tests that need access to repository config files.
"""

import os
import sys
import pytest
from pathlib import Path

# Add the src directory to sys.path to ensure aim_code is importable
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture(scope="function", autouse=False)
def repo_root_cwd(monkeypatch):
    """Change working directory to repository root for tests that need config files.

    This fixture is NOT autouse - tests must explicitly request it if they need
    access to config/ directory files (like paradigm configs, tool configs, etc.).
    """
    repo_root = Path(__file__).parent.parent.parent.parent
    monkeypatch.chdir(repo_root)
