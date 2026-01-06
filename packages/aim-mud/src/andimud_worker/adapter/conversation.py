# andimud_worker/adapter/conversation.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Conversation entry to chat turn adapter for MUD agent integration.

This module translates MUD conversation entries into the chat API format
expected by LLM providers.
"""

from aim_mud_types import MUDConversationEntry


def entries_to_chat_turns(entries: list["MUDConversationEntry"], inject_think: bool = False) -> list[dict[str, str]]:
    """Convert conversation entries to chat turn format.

    Takes MUDConversationEntry objects from the Redis conversation list
    and converts them to the chat turn format expected by LLM providers.

    For assistant turns with think content, the think block is prepended
    wrapped in <think> tags so the LLM can see its prior reasoning.

    Args:
        entries: List of MUDConversationEntry objects in chronological order.

    Returns:
        List of chat turns with 'role' and 'content' keys.
    """
    turns: list[dict[str, str]] = []
    for entry in entries:
        content = entry.content
        # For assistant turns, prepend think block if present
        if entry.role == "assistant" and entry.think and inject_think:
            content = f"<think>{entry.think}</think>\n{content}"
        turns.append({"role": entry.role, "content": content})
    return turns
