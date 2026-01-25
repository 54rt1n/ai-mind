# andimud_worker/conversation/code/__init__.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Code agent strategies for CODE_RAG integration.

This module provides turn building strategies for code-focused agents (like blip)
that use the CODE_RAG system. Key differences from memory-based strategies:

- No conversation memory in CVM (only DOC_SOURCE_CODE, DOC_SPEC documents)
- Session history lives in Redis via MUDConversationManager (ephemeral)
- Consciousness includes focused code, call graph, and semantic search
- FocusTool allows explicit file/line range targeting

Strategies:
- CodeDecisionStrategy: Phase 1 tool selection with lightweight consciousness
- CodeResponseStrategy: Phase 2 narrative generation with code consciousness
"""

from .decision import CodeDecisionStrategy
from .response import CodeResponseStrategy

__all__ = ["CodeDecisionStrategy", "CodeResponseStrategy"]
