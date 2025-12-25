# aim/watcher/__init__.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Watcher module for monitoring conversations and triggering pipelines.

Provides rule-based watching of the conversation index to detect
conditions that should trigger dreamer pipelines.
"""

from aim.watcher.rules import (
    Rule,
    RuleMatch,
    UnanalyzedConversationRule,
    AnalysisWithSummaryRule,
    PostSummaryAnalysisRule,
    StaleConversationRule,
)
from aim.watcher.stability import StabilityTracker
from aim.watcher.watcher import Watcher

__all__ = [
    "Rule",
    "RuleMatch",
    "UnanalyzedConversationRule",
    "AnalysisWithSummaryRule",
    "PostSummaryAnalysisRule",
    "StaleConversationRule",
    "StabilityTracker",
    "Watcher",
]
