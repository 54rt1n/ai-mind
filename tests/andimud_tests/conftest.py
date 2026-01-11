# tests/andimud_tests/conftest.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Pytest configuration for ANDIMUD tests - sets up Django/Evennia environment."""

import os
import sys
from pathlib import Path


_django_setup_done = False


def pytest_configure(config):
    """Configure Django before running tests - this runs globally but only once."""
    global _django_setup_done

    if _django_setup_done:
        return

    # Preserve working directory (Django changes it during setup)
    original_cwd = os.getcwd()

    # Add andimud to Python path
    andimud_path = Path(__file__).parent.parent.parent / "andimud"
    if str(andimud_path) not in sys.path:
        sys.path.insert(0, str(andimud_path))

    # Set Django settings module
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.conf.settings')

    # Import and setup Django
    import django
    django.setup()

    # Restore working directory for other tests
    os.chdir(original_cwd)

    _django_setup_done = True
