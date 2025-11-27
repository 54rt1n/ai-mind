# aim/watcher/__init__.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Watcher module for monitoring conversations and triggering pipelines.

Provides rule-based watching of the conversation index to detect
conditions that should trigger dreamer pipelines.
"""

from aim.watcher.rules import Rule, UnanalyzedConversationRule
from aim.watcher.watcher import Watcher

__all__ = [
    "Rule",
    "UnanalyzedConversationRule",
    "Watcher",
]
