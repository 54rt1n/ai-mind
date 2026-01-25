# andimud_worker/adapter/session.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""MUD session state to chat turn adapter for MUD agent integration.

This module translates MUD session state into the chat API format
expected by LLM providers. It bridges the MUD world representation (rooms,
entities, events) with the conversational turn structure (system/user/assistant
messages).
"""

from typing import Optional

from aim_mud_types import MUDSession

from aim.agents.persona import Persona

from .event import format_event, format_you_see_guidance
from .turn import format_turn_events, format_turn_response

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
    include_format_guidance: bool = True,
    action_guidance: str = "",
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
        include_format_guidance: Whether to include ESH format reminder. Set to
            False for Phase 1 (decision) which expects JSON tool calls only.

    Returns:
        Formatted string containing room context, present entities,
        pending events (if include_events=True), and optional idle guidance.
    """
    parts: list[str] = []

    # Action guidance appears FIRST (results of agent's prior actions)
    if action_guidance:
        parts.append(action_guidance)
        parts.append("")  # Blank line separator

    # Coming online guidance for fresh sessions
    if coming_online:
        parts.append("[~~ Link Coming Online ~~]")
        parts.append("*chirp*Hello!")
        parts.append("Orient yourself in your apartment, feel your surroundings, greet whoever is present.")
        parts.append("[/~~Link~~/]")

    # "You See" grounding guidance (before events)
    you_see = format_you_see_guidance(session.world_state) if session else ""
    if you_see:
        parts.append(you_see)
        parts.append("")

    # Pending events are now in conversation history (pushed via _push_events_to_conversation)
    # This block only adds idle mode agency prompt when include_events=True and idle
    if include_events and idle_mode:
        # Events are already in conversation history, but idle mode needs agency prompt
        parts.append(
            "You don't see anything of note occuring. You have agency - what do you want to do?"
        )

    if guidance:
        parts.append(f"\n[Link Guidance: {guidance}]")

    # Formatting reminder for the agent (use persona_id from session)
    # Phase 1 (decision): No format guidance - expects JSON tool calls only
    # Phase 2 (response): Include ESH format guidance for expressive responses
    if include_format_guidance:
        persona_name = session.persona_id if session.persona_id else "Agent"
        parts.append(f"\n[~~ Link Format Guidance ~~]\n<think>...</think>\n[== {persona_name}'s Emotional State: <list of your +Emotions+> ==]\nBe sure to be expressive with details regarding what you are doing and saying.[/~~Link~~/]")

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
