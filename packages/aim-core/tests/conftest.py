# tests/conftest.py
# AI-Mind (c) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Root conftest for aim-core tests.

Ensures the src directory is in the Python path for all tests.
"""

import sys
from pathlib import Path

# Add the src directory to sys.path to ensure aim_code is importable
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
