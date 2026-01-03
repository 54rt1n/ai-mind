# aim/app/mud/adapter.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Event-to-chat-turn adapter for MUD agent integration.

This module translates MUD session state into the chat API format
expected by LLM providers. It bridges the MUD world representation
(rooms, entities, events) with the conversational turn structure
(system/user/assistant messages).

The adapter follows the design in ANDIMUD.md Section 4 (Session Model).
"""

from typing import Optional, TYPE_CHECKING

from aim_mud_types import EventType, MUDEvent, MUDAction

from .session import MUDSession, MUDTurn
from aim.agents.persona import Persona

if TYPE_CHECKING:
    from .conversation import MUDConversationEntry


# Maximum number of recent turns to include in history
MAX_RECENT_TURNS = 10


def build_system_prompt(session: MUDSession, persona: Persona) -> str:
    """Build system prompt with persona only.

    World state is provided separately in the memory block to avoid
    persisting it into saved documents.

    Args:
        session: Current MUD session (unused, kept for API compatibility).
        persona: The agent's persona configuration.

    Returns:
        Base persona system prompt string for LLM inference.
    """
    return persona.system_prompt()


def build_current_context(
    session: MUDSession,
    idle_mode: bool = False,
    guidance: Optional[str] = None,
    coming_online: bool = False,
    include_events: bool = True,
) -> str:
    """Format current world state and pending events as user turn.

    Creates an XML-structured representation of the current game state
    that serves as the final user turn before the agent responds.

    Args:
        session: Current MUD session with room, entities, and pending events.
        idle_mode: Whether this is a spontaneous (idle) turn.
        guidance: Optional user-provided guidance.
        coming_online: Whether this is the first turn (agent coming online).
        include_events: Whether to include pending events. Set to False for
            Phase 2 turns when events are already in conversation history.

    Returns:
        Formatted string containing room context, present entities,
        pending events (if include_events=True), and optional idle guidance.
    """
    parts: list[str] = []

    # Coming online guidance for fresh sessions
    if coming_online:
        parts.append("[~~ Guidance: Coming online ~~]")
        parts.append("Orient yourself in your apartment, feel your surroundings, greet whoever is present.")
        parts.append("[/~~Guidance~~/]")

    # Pending events - only include if requested (not when already in history)
    if include_events:
        if session.pending_events:
            parts.append(f'<events count="{len(session.pending_events)}">')
            for event in session.pending_events:
                parts.append(format_event(event))
            parts.append("</events>")
        elif idle_mode:
            parts.append('<events count="0">')
            parts.append("  No new events have occurred.")
            parts.append("</events>")
            parts.append("<idle>")
            parts.append(
                "  This is a spontaneous turn. Act only if you have a meaningful, "
                "persona-grounded reason based on the current world state."
            )
            parts.append("</idle>")

    if guidance:
        parts.append(f"\n[Guidance: {guidance}]")

    # Formatting reminder for the agent
    parts.append("\n[~~ FORMAT: <think>...</think> then [== Andi's Emotional State: +Emotion+ ==] then *prose narrative* ~~]")

    return "\n".join(parts)


def build_history_turns(session: MUDSession) -> list[dict[str, str]]:
    """Build recent turn history into chat format."""
    turns: list[dict[str, str]] = []
    limit = getattr(session, "max_recent_turns", MAX_RECENT_TURNS) or MAX_RECENT_TURNS
    history_turns = session.recent_turns[-limit:]
    for turn in history_turns:
        events_content = format_turn_events(turn)
        turns.append({"role": "user", "content": events_content})

        response_content = format_turn_response(turn)
        turns.append({"role": "assistant", "content": response_content})

    return turns


def _format_emote_with_quotes(actor: str, content: str) -> str:
    """Format an emote, extracting any quoted speech.

    If the content contains double quotes, splits the action from the speech:
    - Input: 'looks at you. "Hello!"'
    - Output: '*Prax looks at you.* "Hello!"'

    Args:
        actor: The actor performing the emote.
        content: The emote content, possibly containing quoted speech.

    Returns:
        Formatted emote with action in *asterisks* and quotes preserved.
    """
    # Find the first double quote
    quote_idx = content.find('"')

    if quote_idx == -1:
        # No quotes - wrap the whole thing in asterisks
        return f"*{actor} {content}*"

    # Split into action and quoted speech
    action_part = content[:quote_idx].strip()
    speech_part = content[quote_idx:].strip()

    if action_part:
        return f"*{actor} {action_part}* {speech_part}"
    else:
        # Quote at the start - just wrap actor action
        return f"*{actor}* {speech_part}"


def format_event(event: MUDEvent) -> str:
    """Format a single event for display.

    Converts a MUDEvent into a human-readable string representation
    appropriate for including in chat context.

    Actions (emotes, object interactions, movement) are wrapped in *asterisks*.
    Emotes containing quoted speech have the action and speech separated.

    Args:
        event: The MUD event to format.

    Returns:
        Formatted event string with appropriate prefix and structure.
    """
    if event.event_type == EventType.SPEECH:
        return f'{event.actor} says, "{event.content}"'

    elif event.event_type == EventType.EMOTE:
        return _format_emote_with_quotes(event.actor, event.content)

    elif event.event_type == EventType.MOVEMENT:
        # Determine if this is an arrival or departure based on content
        content_lower = event.content.lower()
        if "enter" in content_lower or "arrive" in content_lower:
            return f"*{event.actor} has arrived.*"
        else:
            return f"*{event.actor} has left.*"

    elif event.event_type == EventType.OBJECT:
        return f"*{event.actor} {event.content}*"

    elif event.event_type == EventType.AMBIENT:
        return f"{event.content}"

    elif event.event_type == EventType.SYSTEM:
        return f"[System] {event.content}"

    else:
        # Fallback for unknown event types
        return f"[{event.event_type.value}] {event.actor}: {event.content}"


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


def entries_to_chat_turns(entries: list["MUDConversationEntry"]) -> list[dict[str, str]]:
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
        if entry.role == "assistant" and entry.think:
            content = f"<think>{entry.think}</think>\n{content}"
        turns.append({"role": entry.role, "content": content})
    return turns
