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
            # Arrival: rely on source_room_name metadata
            source = event.metadata.get("source_room_name")
            if source:
                return f"*You see {event.actor} arriving from {source}.*"
            return f"*You see {event.actor} has arrived.*"
        else:
            # Departure: rely on destination_room_name metadata
            destination = event.metadata.get("destination_room_name")
            if destination:
                return f"*You see {event.actor} leaving toward {destination}.*"
            return f"*You see {event.actor} leave.*"

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
        # For movement, use destination from metadata
        room_name = event.metadata.get("destination_room_name", "somewhere")
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


def format_self_action_guidance(self_actions: list[MUDEvent], world_state=None) -> str:
    """Format a list of self-actions as an enhanced guidance block.

    Creates a visually prominent notification box for self-actions with:
    - Clear visual separators
    - Action type labels
    - Explicit state confirmations (location/inventory)
    - Contextual reminders

    Args:
        self_actions: List of self-action MUDEvent objects.
        world_state: Optional WorldState for current inventory information.

    Returns:
        Formatted guidance string with enhanced action confirmations,
        or empty string if no self-actions.
    """
    if not self_actions:
        return ""

    separator = "═" * 60

    # Single action
    if len(self_actions) == 1:
        event = self_actions[0]
        formatted = format_self_event(event)

        lines = [
            separator,
            "!! IMPORTANT: YOUR RECENT ACTION !!",
            separator,
            ""
        ]

        # Add action type and details
        if event.event_type == EventType.MOVEMENT:
            source = event.metadata.get("source_room_name", "somewhere")
            destination = event.metadata.get("destination_room_name", "somewhere")
            lines.extend([
                "Action Type: MOVEMENT",
                "",
                f"You just moved from: {source}",
                f"You just moved to: {destination}",
                "",
                f"CURRENT LOCATION: {destination}",
                "",
                "This is your new location. You have physically moved and are now",
                "in a different room than before.",
            ])

        elif event.event_type == EventType.OBJECT:
            content_lower = event.content.lower()
            target = event.target or _extract_object_from_content(event.content)

            lines.extend([
                "Action Type: OBJECT INTERACTION",
                "",
            ])

            if "pick" in content_lower or "get" in content_lower or "take" in content_lower:
                lines.append(f"You picked up: {target}")
                lines.append("")
                inventory_summary = _get_current_inventory_summary(world_state)
                lines.append(f"CURRENT INVENTORY: {inventory_summary}")
            elif "drop" in content_lower:
                lines.append(f"You dropped: {target}")
                lines.append("")
                inventory_summary = _get_current_inventory_summary(world_state)
                lines.append(f"CURRENT INVENTORY: {inventory_summary}")
            elif "give" in content_lower:
                recipient = event.metadata.get("target_name", "someone")
                lines.append(f"You gave {target} to {recipient}")
                lines.append("")
                inventory_summary = _get_current_inventory_summary(world_state)
                lines.append(f"CURRENT INVENTORY: {inventory_summary}")
            elif "put" in content_lower:
                container = event.metadata.get("container_name", "somewhere")
                lines.append(f"You put {target} in {container}")
                lines.append("")
                inventory_summary = _get_current_inventory_summary(world_state)
                lines.append(f"CURRENT INVENTORY: {inventory_summary}")
            else:
                lines.append(formatted)

        elif event.event_type == EventType.EMOTE:
            lines.extend([
                "Action Type: EMOTE",
                "",
                formatted,
            ])

        else:
            lines.extend([
                f"Action Type: {event.event_type.value.upper()}",
                "",
                formatted,
            ])

        lines.extend([
            "",
            separator,
        ])

        return "\n".join(lines)

    # Multiple actions
    else:
        lines = [
            separator,
            "!! IMPORTANT: YOUR RECENT ACTIONS !!",
            separator,
            ""
        ]

        # Track state changes
        movement_occurred = False
        object_interaction = False
        final_location = None

        for idx, event in enumerate(self_actions, 1):
            formatted = format_self_event(event)

            if event.event_type == EventType.MOVEMENT:
                movement_occurred = True
                source = event.metadata.get("source_room_name", "somewhere")
                destination = event.metadata.get("destination_room_name", "somewhere")
                final_location = destination
                lines.append(f"{idx}. MOVEMENT: You moved from {source} to {destination}")
                lines.append(f"   → Current Location: {destination}")

            elif event.event_type == EventType.OBJECT:
                object_interaction = True
                target = event.target or _extract_object_from_content(event.content)
                content_lower = event.content.lower()

                if "pick" in content_lower or "get" in content_lower or "take" in content_lower:
                    lines.append(f"{idx}. OBJECT: You picked up {target}")
                elif "drop" in content_lower:
                    lines.append(f"{idx}. OBJECT: You dropped {target}")
                elif "give" in content_lower:
                    recipient = event.metadata.get("target_name", "someone")
                    lines.append(f"{idx}. OBJECT: You gave {target} to {recipient}")
                elif "put" in content_lower:
                    container = event.metadata.get("container_name", "somewhere")
                    lines.append(f"{idx}. OBJECT: You put {target} in {container}")
                else:
                    lines.append(f"{idx}. OBJECT: {formatted}")

                inventory_summary = _get_current_inventory_summary(world_state)
                lines.append(f"   → Current Inventory: {inventory_summary}")

            elif event.event_type == EventType.EMOTE:
                lines.append(f"{idx}. EMOTE: {formatted}")

            else:
                lines.append(f"{idx}. {event.event_type.value.upper()}: {formatted}")

            lines.append("")

        # Summary line
        summary_parts = []
        if movement_occurred and object_interaction:
            summary_parts.append("You have taken multiple actions. Your location and inventory")
            summary_parts.append("have changed.")
            if final_location:
                inventory_summary = _get_current_inventory_summary(world_state)
                summary_parts.append(f"You are now in {final_location}.")
                if inventory_summary != "none":
                    summary_parts.append(f"Carrying: {inventory_summary}")
        elif movement_occurred:
            summary_parts.append("You have moved between multiple locations.")
            if final_location:
                summary_parts.append(f"You are now in {final_location}.")
        elif object_interaction:
            summary_parts.append("You have performed multiple object interactions.")
            inventory_summary = _get_current_inventory_summary(world_state)
            if inventory_summary != "none":
                summary_parts.append(f"Carrying: {inventory_summary}")
        else:
            summary_parts.append(f"You have taken {len(self_actions)} actions.")

        lines.append(" ".join(summary_parts))
        lines.extend([
            "",
            separator,
        ])

        return "\n".join(lines)


def _get_current_inventory_summary(world_state) -> str:
    """Get a summary of current inventory items.

    Args:
        world_state: Optional WorldState object containing inventory.

    Returns:
        Comma-separated list of item names, or "none" if empty.
    """
    if not world_state or not hasattr(world_state, 'inventory') or not world_state.inventory:
        return "none"

    item_names = [item.name for item in world_state.inventory]
    return ", ".join(item_names)
