# andimud_worker/adapter/event.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""MUDEvent to chat turn adapter for MUD agent integration.

This module translates MUDEvent into a human-readable string representation
appropriate for including in chat context. Actions (emotes, object interactions,
movement) are wrapped in *asterisks*. Emotes containing quoted speech have the
action and speech separated.
"""

from aim_mud_types import EventType, MUDEvent


def format_event(event: MUDEvent, first_person: bool = False) -> str:
    """Format a single event for display.

    Converts a MUDEvent into a human-readable string representation
    appropriate for including in chat context.

    Actions (emotes, object interactions, movement) are wrapped in *asterisks*.
    Emotes containing quoted speech have the action and speech separated.

    Args:
        event: The MUD event to format.
        first_person: If True, format as first-person (for self-actions in documents).

    Returns:
        Formatted event string with appropriate prefix and structure.
    """
    if first_person:
        return format_self_event(event)

    if event.event_type == EventType.SPEECH:
        return f'{event.actor} says, "{event.content}"'

    elif event.event_type == EventType.EMOTE:
        return _format_emote_with_quotes(event.actor, event.content)

    elif event.event_type == EventType.MOVEMENT:
        # Determine if this is an arrival or departure based on content
        content_lower = event.content.lower()
        if "enter" in content_lower or "arrive" in content_lower:
            return f"*You see {event.actor} has arrived.*"
        else:
            return f"*You watch {event.actor} leave.*"

    elif event.event_type == EventType.OBJECT:
        return f"*{event.actor} {event.content}*"

    elif event.event_type == EventType.AMBIENT:
        return f"{event.content}"

    elif event.event_type == EventType.SYSTEM:
        return f"[System] {event.content}"

    elif event.event_type == EventType.NARRATIVE:
        return event.content

    else:
        # Fallback for unknown event types
        return f"[{event.event_type.value}] {event.actor}: {event.content}"

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

def format_self_event(event: MUDEvent) -> str:
    """Format a self-action event in first person for action guidance.

    Self-events are formatted as confirmations of actions the agent took,
    e.g., "You moved to The Kitchen." instead of "*Andi arrives.*"

    Args:
        event: The self-action MUDEvent to format.

    Returns:
        First-person formatted action confirmation string.
    """
    if event.event_type == EventType.MOVEMENT:
        # For movement, use room_name from event
        room_name = event.room_name or event.metadata.get("room_name", "somewhere")
        return f"You moved to {room_name}."

    elif event.event_type == EventType.OBJECT:
        content_lower = event.content.lower()
        target = event.target or ""

        if "pick" in content_lower or "get" in content_lower or "take" in content_lower:
            obj = target or _extract_object_from_content(event.content)
            container = event.metadata.get("container_name")
            if container:
                return f"You took {obj} from {container}."
            return f"You picked up {obj}."

        elif "drop" in content_lower:
            obj = target or _extract_object_from_content(event.content)
            return f"You dropped {obj}."

        elif "give" in content_lower:
            obj = target or "something"
            recipient = event.metadata.get("target_name", "someone")
            return f"You gave {obj} to {recipient}."

        elif "put" in content_lower:
            obj = target or "something"
            container = event.metadata.get("container_name", "somewhere")
            return f"You put {obj} in {container}."

        else:
            # Generic object manipulation - clean up the content
            return f"You {event.content.lower().strip('*').strip()}"

    elif event.event_type == EventType.EMOTE:
        # Self-emotes: convert third person to first person acknowledgment
        return f"You expressed: {event.content}"

    else:
        # Fallback for other self-events
        return f"You: {event.content}"


def _extract_object_from_content(content: str) -> str:
    """Extract object name from event content string."""
    # Remove asterisks and common action words
    cleaned = content.strip("*").strip()
    words = cleaned.split()
    # Skip actor name and action verb, return rest
    if len(words) > 2:
        return " ".join(words[2:]).rstrip(".")
    return "something"


def format_self_action_guidance(self_actions: list[MUDEvent]) -> str:
    """Format a list of self-actions as guidance block.

    Args:
        self_actions: List of self-action MUDEvent objects.

    Returns:
        Formatted guidance string with action confirmations,
        or empty string if no self-actions.
    """
    if not self_actions:
        return ""

    lines = []
    for event in self_actions:
        formatted = format_self_event(event)
        lines.append(f"[!! Action: {formatted} !!]")

    return "\n".join(lines)

