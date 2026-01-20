# aim_legacy/watcher/__init__.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Watcher module for monitoring conversations and triggering pipelines.

Provides rule-based watching of the conversation index to detect
conditions that should trigger dreamer pipelines.

** LEGACY MODULE ** - Preserved from pre-migration codebase
This module was part of the legacy dreamer infrastructure and has been
moved to aim_legacy for backward compatibility.
"""

from aim_legacy.watcher.rules import (
    Rule,
    RuleMatch,
    UnanalyzedConversationRule,
    AnalysisWithSummaryRule,
    PostSummaryAnalysisRule,
    StaleConversationRule,
)
from aim_legacy.watcher.stability import StabilityTracker
from aim_legacy.watcher.watcher import Watcher

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
