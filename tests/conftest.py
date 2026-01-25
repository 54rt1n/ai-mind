# tests/conftest.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Pytest configuration for ai-mind root tests.

Ensures src packages (aim_cli, aim_server, repo_watcher, etc.) are importable.
"""

import sys
from pathlib import Path

# Add src directory to path for local packages
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
