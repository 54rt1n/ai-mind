# andimud_worker/adapter/turn.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""MUD turn to chat turn adapter for MUD agent integration.

This module translates MUD turn state into the chat API format
expected by LLM providers. It bridges the MUD turn state
(events, actions, thinking) with the conversational turn structure
(system/user/assistant messages).
"""

from aim_mud_types import MUDTurn

from .event import format_event


def format_turn_events(turn: MUDTurn) -> str:
    """Format events from a past turn.

    Converts all events from a historical turn into a single string
    for inclusion as a user message in the chat history.

    Args:
        turn: The historical turn containing events.

    Returns:
        Formatted string of all events, or a message indicating
        spontaneous action if no events were present.
    """
    if not turn.events_received:
        return "[No events - spontaneous action]"

    return "\n".join(format_event(e) for e in turn.events_received)


def format_turn_response(turn: MUDTurn) -> str:
    """Format the agent's response from a past turn.

    Converts the agent's thinking and actions from a historical turn
    into a single string for inclusion as an assistant message.

    Args:
        turn: The historical turn containing the agent's response.

    Returns:
        Formatted string containing thinking and actions, or
        "[No response]" if the turn had no content.
    """
    parts: list[str] = []

    # Include thinking if present
    if turn.thinking:
        parts.append(turn.thinking)

    # Include actions
    if turn.actions_taken:
        for action in turn.actions_taken:
            if action.tool == "speak":
                text = action.args.get("text", "")
                if text:
                    parts.append(text)
                continue
            parts.append(f"[Action: {action.tool}] {action.to_command()}")

    return "\n".join(parts) if parts else "[No response]"
