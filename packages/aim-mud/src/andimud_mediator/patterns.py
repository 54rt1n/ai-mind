# andimud_mediator/patterns.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Regular expression patterns for mediator commands."""

import re

# Command patterns

# @dreamer <agent-id> on/off
DREAMER_PATTERN = re.compile(
    r"^@dreamer\s+(\w+)\s+(on|off)$",
    re.IGNORECASE,
)

# Analysis commands (require conversation_id)
# @analyze <agent> = <conversation_id>, <guidance>
ANALYZE_PATTERN = re.compile(
    r"^@analyze\s+(\w+)\s*=\s*([^,]+)(?:,(.*))?$",
    re.IGNORECASE,
)
# @summary <agent> = <conversation_id>
SUMMARY_PATTERN = re.compile(
    r"^@summary\s+(\w+)\s*=\s*(.+)$",
    re.IGNORECASE,
)

# Creative commands (query,guidance optional - create own conversation)
# @journal <agent> [= <query>, <guidance>]
JOURNAL_PATTERN = re.compile(
    r"^@journal\s+(\w+)(?:\s*=\s*([^,]*)(?:,(.*))?)?$",
    re.IGNORECASE,
)
# @ponder <agent> [= <query>, <guidance>]
PONDER_PATTERN = re.compile(
    r"^@ponder\s+(\w+)(?:\s*=\s*([^,]*)(?:,(.*))?)?$",
    re.IGNORECASE,
)
# @daydream <agent> [= <query>, <guidance>]
DAYDREAM_PATTERN = re.compile(
    r"^@daydream\s+(\w+)(?:\s*=\s*([^,]*)(?:,(.*))?)?$",
    re.IGNORECASE,
)
# @critique <agent> [= <query>, <guidance>]
CRITIQUE_PATTERN = re.compile(
    r"^@critique\s+(\w+)(?:\s*=\s*([^,]*)(?:,(.*))?)?$",
    re.IGNORECASE,
)
# @research <agent> [= <query>, <guidance>]
RESEARCH_PATTERN = re.compile(
    r"^@research\s+(\w+)(?:\s*=\s*([^,]*)(?:,(.*))?)?$",
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
    r"^@planner\s+(\w+)\s+(on|off)$",
    re.IGNORECASE,
)

# @plan <agent> = <objective>
PLAN_PATTERN = re.compile(
    r"^@plan\s+(\w+)\s*=\s*(.+)$",
    re.IGNORECASE,
)

# @update <agent> = <guidance>
UPDATE_PATTERN = re.compile(
    r"^@update\s+(\w+)\s*=\s*(.+)$",
    re.IGNORECASE,
)
