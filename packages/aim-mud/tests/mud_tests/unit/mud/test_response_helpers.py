# tests/mud_tests/unit/mud/test_response_helpers.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for andimud_worker.turns.response helpers."""

from andimud_worker.turns.response import parse_agent_action_response, sanitize_response


def test_sanitize_response_truncates_you_see_artifact():
    text = "I nod.\n[~~ You See ~~]\nRoom details should not leak."
    assert sanitize_response(text) == "I nod."


def test_sanitize_response_truncates_link_guidance_artifact():
    text = (
        "[== Andi's Emotional State: +Calm+ ==]\n"
        "I smile warmly.\n"
        "[Link Guidance: Keep this hidden]\n"
        "Hidden content"
    )
    sanitized = sanitize_response(text)
    assert sanitized == "[== Andi's Emotional State: +Calm+ ==]\nI smile warmly."


def test_sanitize_response_preserves_second_think_truncation():
    text = "<think>first</think>\nvisible text</think>\ntrailing"
    assert sanitize_response(text) == "<think>first</think>\nvisible text"


def test_parse_agent_action_response_with_guidance_suffix():
    response = (
        '{"action": "move", "location": "Kitchen"}\n'
        "[Link Guidance: Internal rubric]"
    )
    action, args, error = parse_agent_action_response(response)
    assert error == ""
    assert action == "move"
    assert args == {"location": "Kitchen"}
