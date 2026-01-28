# andimud_worker/adapter/event.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""MUDEvent to chat turn adapter for MUD agent integration.

This module translates MUDEvent into a human-readable string representation
appropriate for including in chat context. Actions (emotes, object interactions,
movement) are wrapped in *asterisks*. Emotes containing quoted speech have the
action and speech separated.

Note: The formatting functions have been moved to aim_mud_types.formatters.
This module re-exports them for backwards compatibility.
"""

# Re-export all formatting functions from the shared module
from aim_mud_types.formatters import (
    format_event,
    format_self_event,
    format_self_action_guidance,
    format_you_see_guidance,
    _format_emote_with_quotes,
    _format_code_event,
    _extract_object_from_content,
    _get_ground_items,
    _get_container_items,
    _get_current_inventory_summary,
)

__all__ = [
    "format_event",
    "format_self_event",
    "format_self_action_guidance",
    "format_you_see_guidance",
    "_format_emote_with_quotes",
    "_format_code_event",
    "_extract_object_from_content",
    "_get_ground_items",
    "_get_container_items",
    "_get_current_inventory_summary",
]
