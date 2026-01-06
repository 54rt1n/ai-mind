# aim/app/mud/worker/turns/response.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Response processing helpers for MUD worker turn processing.

Pure functions for normalizing, parsing, and validating LLM responses.
"""

import json
import re
from typing import Optional

from aim.dreamer.executor import extract_think_tags
from aim.tool.formatting import ToolUser
from ...utils import sanitize_response


def normalize_response(response: str) -> str:
    """Normalize a free-text response for emission.

    Removes extra blank lines and strips leading/trailing whitespace while
    preserving intentional paragraph breaks (single blank lines).

    Args:
        response: Raw response text from LLM

    Returns:
        Normalized response text
    """
    if not response:
        return ""

    stripped = response.strip()
    if not stripped:
        return ""

    lines = [line.rstrip() for line in stripped.splitlines()]
    normalized: list[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if not blank:
                normalized.append("")
                blank = True
            continue
        normalized.append(line)
        blank = False

    return "\n".join(normalized).strip()


def has_emotional_state_header(response: str) -> bool:
    """Check if response starts with emotional state header after think block.

    Validates that the response begins with the expected format:
    [== <name>'s Emotional State: <emotions> ==]

    Args:
        response: The response text to validate

    Returns:
        True if response has the required header format
    """
    # Remove think block first
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    # Check if it starts with [== ... Emotional State ... ==]
    return bool(re.match(r'\[==.*Emotional State.*==\]', cleaned, re.IGNORECASE))


def extract_speak_text_from_tool_call(response: str) -> Optional[str]:
    """Extract speak text if the response is a tool-call-like JSON blob.

    Attempts to parse the response as JSON and extract the speak action text.
    Useful for recovering when the model returns a tool call instead of free text.

    Args:
        response: The response text to extract from

    Returns:
        Extracted speak text, or None if not found or unparseable
    """
    if not response:
        return None

    stripped = response.strip()
    if not stripped:
        return None

    parsed = None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        try:
            parsed = ToolUser([])._extract_tool_call(stripped)
        except Exception:
            parsed = None

    if not isinstance(parsed, dict):
        return None

    if "speak" not in parsed:
        return None

    payload = parsed.get("speak")
    if isinstance(payload, str):
        return payload

    if isinstance(payload, dict):
        for key in ("text", "say", "message", "content"):
            value = payload.get(key)
            if isinstance(value, str):
                return value

    return None


def parse_agent_action_response(response: str) -> tuple[Optional[str], dict, str]:
    """Parse @agent JSON response into (action, args, error).

    Attempts to parse the response as JSON and extract an action specification.
    Supports two formats:
    1. {"action": "<name>", ...} - preferred format
    2. {"<action_name>": {...}} - alternate tool-call format

    Args:
        response: The raw response text from LLM

    Returns:
        Tuple of (action_name, arguments_dict, error_message).
        If parsing succeeds, error_message is empty.
    """
    cleaned, _think = extract_think_tags(response)
    cleaned = sanitize_response(cleaned)
    text = cleaned.strip()
    parsed = None
    json_text = None

    try:
        parsed = json.loads(text)
        json_text = text
    except json.JSONDecodeError:
        # Try to extract a JSON object from mixed text
        json_candidates = []
        brace_depth = 0
        start_idx = None
        for i, char in enumerate(text):
            if char == "{":
                if brace_depth == 0:
                    start_idx = i
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth == 0 and start_idx is not None:
                    json_candidates.append(text[start_idx : i + 1])
                    start_idx = None
        for candidate in reversed(json_candidates):
            try:
                parsed = json.loads(candidate)
                json_text = candidate.strip()
                break
            except json.JSONDecodeError:
                continue

    if not isinstance(parsed, dict):
        # Include truncated response in error for debugging
        preview = text[:200] + "..." if len(text) > 200 else text
        return None, {}, f"Could not parse JSON: {preview}"

    # Preferred format: {"action": "<name>", ...}
    if "action" in parsed:
        action = parsed.get("action")
        if not isinstance(action, str):
            return None, {}, "Action must be a string"
        args = {k: v for k, v in parsed.items() if k != "action"}
        return action.lower(), args, ""

    # Alternate tool-call format: {"describe": {...}}
    if len(parsed) == 1:
        action = next(iter(parsed))
        args = parsed.get(action)
        if isinstance(action, str) and isinstance(args, dict):
            return action.lower(), args, ""

    return None, {}, "Missing action field"
