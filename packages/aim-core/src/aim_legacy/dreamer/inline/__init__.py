# aim/dreamer/inline/__init__.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Inline pipeline execution without distributed infrastructure.

This module provides a clean, non-distributed pipeline execution interface
that runs scenarios synchronously in-process without Redis queues or state stores.

Use this for:
- CLI tools that need immediate results
- Testing and debugging scenarios
- Single-user applications
- Local development

For distributed, multi-worker deployments, use the dreamer.server module instead.
"""

from .scheduler import execute_pipeline_inline

__all__ = ['execute_pipeline_inline']
