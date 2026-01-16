# andimud_mediator/patterns.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Regular expression patterns for mediator commands."""

import re


def normalize_agent_id(name: str) -> str:
    """Normalize a display name to agent_id format.

    Converts "Lin Yu" → "linyu", "Nova" → "nova", etc.
    Removes spaces and converts to lowercase for Redis key lookups.

    Args:
        name: Display name or persona_id

    Returns:
        Normalized agent_id for Redis lookups
    """
    return name.strip().replace(" ", "").lower()


# Command patterns

# @dreamer <agent-id> on/off
DREAMER_PATTERN = re.compile(
    r"^@dreamer\s+([^=\s]+(?:\s+[^=\s]+)*)\s+(on|off)$",
    re.IGNORECASE,
)

# Analysis commands (require conversation_id)
# @analyze <agent> = <conversation_id>, <guidance>
ANALYZE_PATTERN = re.compile(
    r"^@analyze\s+([^=]+?)\s*=\s*([^,]+)(?:,(.*))?$",
    re.IGNORECASE,
)
# @summary <agent> = <conversation_id>
SUMMARY_PATTERN = re.compile(
    r"^@summary\s+([^=]+?)\s*=\s*(.+)$",
    re.IGNORECASE,
)

# Creative commands (query,guidance optional - create own conversation)
# @journal <agent> [= <query>, <guidance>]
JOURNAL_PATTERN = re.compile(
    r"^@journal\s+([^=]+?)(?:\s*=\s*([^,]*)(?:,(.*))?)?$",
    re.IGNORECASE,
)
# @ponder <agent> [= <query>, <guidance>]
PONDER_PATTERN = re.compile(
    r"^@ponder\s+([^=]+?)(?:\s*=\s*([^,]*)(?:,(.*))?)?$",
    re.IGNORECASE,
)
# @daydream <agent> [= <query>, <guidance>]
DAYDREAM_PATTERN = re.compile(
    r"^@daydream\s+([^=]+?)(?:\s*=\s*([^,]*)(?:,(.*))?)?$",
    re.IGNORECASE,
)
# @critique <agent> [= <query>, <guidance>]
CRITIQUE_PATTERN = re.compile(
    r"^@critique\s+([^=]+?)(?:\s*=\s*([^,]*)(?:,(.*))?)?$",
    re.IGNORECASE,
)
# @research <agent> [= <query>, <guidance>]
RESEARCH_PATTERN = re.compile(
    r"^@research\s+([^=]+?)(?:\s*=\s*([^,]*)(?:,(.*))?)?$",
    re.IGNORECASE,
)

# Map command names to scenario names
COMMAND_TO_SCENARIO = {
    "analyze": "analysis_dialogue",
    "summary": "summarizer",
    "journal": "journaler_dialogue",
    "ponder": "philosopher_dialogue",
    "daydream": "daydream_dialogue",
    "critique": "critique_dialogue",
    "research": "researcher_dialogue",
}

# Planner commands
# @planner <agent-id> on/off
PLANNER_PATTERN = re.compile(
    r"^@planner\s+([^=\s]+(?:\s+[^=\s]+)*)\s+(on|off)$",
    re.IGNORECASE,
)

# @plan <agent> = <objective>
PLAN_PATTERN = re.compile(
    r"^@plan\s+([^=]+?)\s*=\s*(.+)$",
    re.IGNORECASE,
)

# @update <agent> = <guidance>
UPDATE_PATTERN = re.compile(
    r"^@update\s+([^=]+?)\s*=\s*(.+)$",
    re.IGNORECASE,
)
