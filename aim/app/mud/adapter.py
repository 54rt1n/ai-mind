# aim/app/mud/adapter.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Event-to-chat-turn adapter for MUD agent integration.

This module translates MUD session state into the chat API format
expected by LLM providers. It bridges the MUD world representation
(rooms, entities, events) with the conversational turn structure
(system/user/assistant messages).

The adapter follows the design in ANDIMUD.md Section 4 (Session Model).
"""

from typing import Optional

from aim_mud_types import EventType, MUDEvent, MUDAction

from .session import MUDSession, MUDTurn
from ...agents.persona import Persona


# Maximum number of recent turns to include in history
MAX_RECENT_TURNS = 10


def build_chat_turns(session: MUDSession, persona: Persona) -> list[dict[str, str]]:
    """Convert MUD session state into chat API format.

    Assembles the complete context for LLM inference:
    1. System prompt with persona and MUD context
    2. Retrieved memories (placeholder for future integration)
    3. Recent turn history (events and responses)
    4. Current context as the final user turn

    Args:
        session: Current MUD session with room, entities, events, history.
        persona: The agent's persona for system prompt generation.

    Returns:
        List of chat turns ready for LLM inference. Each turn is a dict
        with 'role' (system/user/assistant) and 'content' keys.
    """
    turns: list[dict[str, str]] = []

    # 1. System prompt
    system = build_system_prompt(session, persona)
    turns.append({"role": "system", "content": system})

    # 2. Retrieved memories (placeholder - will integrate with memory search later)
    # For now, skip. When implemented, format as XML block and add as user turn.

    # 3. Recent turn history
    history_turns = session.recent_turns[-MAX_RECENT_TURNS:]
    for turn in history_turns:
        # Each historical turn becomes a user message (events) and assistant message (response)
        events_content = format_turn_events(turn)
        turns.append({"role": "user", "content": events_content})

        response_content = format_turn_response(turn)
        turns.append({"role": "assistant", "content": response_content})

    # 4. Current context as final user turn
    current = build_current_context(session)
    turns.append({"role": "user", "content": current})

    return turns


def build_system_prompt(session: MUDSession, persona: Persona) -> str:
    """Build system prompt with MUD context.

    Combines the persona's base system prompt with MUD-specific context
    including current location and who is present in the room.

    Args:
        session: Current MUD session with room and entity information.
        persona: The agent's persona configuration.

    Returns:
        Complete system prompt string for LLM inference.
    """
    # Build location description
    location_desc = ""
    if session.current_room:
        location_desc = (
            f"You are in {session.current_room.name}. "
            f"{session.current_room.description}"
        )

    # Build list of present entities (excluding self)
    present = [
        entity.name
        for entity in session.entities_present
        if not entity.is_self
    ]
    if present:
        present_desc = f"Present: {', '.join(present)}"
    else:
        present_desc = "You are alone."

    # Get base persona prompt
    # Persona.system_prompt() accepts optional location parameter
    base_prompt = persona.system_prompt(location=location_desc if location_desc else None)

    # Append MUD context
    mud_context_parts = []
    if location_desc:
        mud_context_parts.append(f"[Current Location]\n{location_desc}")
    mud_context_parts.append(present_desc)

    mud_context = "\n\n".join(mud_context_parts)

    return f"{base_prompt}\n\n{mud_context}"


def build_current_context(session: MUDSession) -> str:
    """Format current world state and pending events as user turn.

    Creates an XML-structured representation of the current game state
    that serves as the final user turn before the agent responds.

    Args:
        session: Current MUD session with room, entities, and pending events.

    Returns:
        Formatted string containing room context, present entities,
        pending events, and the "What do you do?" prompt.
    """
    parts: list[str] = []

    # Room context
    if session.current_room:
        parts.append(f'<location name="{session.current_room.name}">')
        parts.append(f"  {session.current_room.description}")
        if session.current_room.exits:
            exits = ", ".join(session.current_room.exits.keys())
            parts.append(f"  Exits: {exits}")
        parts.append("</location>")

    # Entities present (excluding self)
    non_self_entities = [e for e in session.entities_present if not e.is_self]
    if non_self_entities:
        parts.append("<present>")
        for entity in non_self_entities:
            parts.append(
                f'  <entity name="{entity.name}" type="{entity.entity_type}"/>'
            )
        parts.append("</present>")

    # Pending events
    if session.pending_events:
        parts.append(f'<events count="{len(session.pending_events)}">')
        for event in session.pending_events:
            parts.append(format_event(event))
        parts.append("</events>")

    # Action prompt
    parts.append("\nWhat do you do?")

    return "\n".join(parts)


def format_event(event: MUDEvent) -> str:
    """Format a single event for display.

    Converts a MUDEvent into a human-readable string representation
    appropriate for including in chat context.

    Args:
        event: The MUD event to format.

    Returns:
        Formatted event string with appropriate prefix and structure.
    """
    if event.event_type == EventType.SPEECH:
        return f'  {event.actor} says: "{event.content}"'

    elif event.event_type == EventType.EMOTE:
        return f"  {event.actor} {event.content}"

    elif event.event_type == EventType.MOVEMENT:
        # Determine if this is an arrival or departure based on content
        content_lower = event.content.lower()
        if "enter" in content_lower or "arrive" in content_lower:
            return f"  {event.actor} has arrived."
        else:
            return f"  {event.actor} has left."

    elif event.event_type == EventType.OBJECT:
        return f"  {event.actor} {event.content}"

    elif event.event_type == EventType.AMBIENT:
        return f"  {event.content}"

    elif event.event_type == EventType.SYSTEM:
        return f"  [System] {event.content}"

    else:
        # Fallback for unknown event types
        return f"  [{event.event_type.value}] {event.actor}: {event.content}"


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

    if turn.thinking:
        parts.append(turn.thinking)

    if turn.actions_taken:
        for action in turn.actions_taken:
            parts.append(f"[Action: {action.tool}] {action.to_command()}")

    return "\n".join(parts) if parts else "[No response]"
