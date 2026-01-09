# tests/unit/test_events.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for MUDEvent model and methods."""

import pytest
from datetime import datetime, timezone

from aim_mud_types.events import MUDEvent
from aim_mud_types.enums import EventType, ActorType


class TestMUDEventIsSelfSpeechEcho:
    """Test MUDEvent.is_self_speech_echo() method."""

    def test_self_speech_returns_true(self):
        """Self-speech event should return True."""
        self_speech = MUDEvent(
            event_type=EventType.SPEECH,
            actor="Andi",
            actor_id="ai_andi",
            actor_type=ActorType.AI,
            room_id="room1",
            room_name="Test Room",
            content="Hello world",
            metadata={"is_self_action": True}
        )
        assert self_speech.is_self_speech_echo() is True

    def test_self_emote_returns_false(self):
        """Self-emote should return False (only speech is filtered)."""
        self_emote = MUDEvent(
            event_type=EventType.EMOTE,
            actor="Andi",
            actor_id="ai_andi",
            actor_type=ActorType.AI,
            room_id="room1",
            room_name="Test Room",
            content="waves enthusiastically",
            metadata={"is_self_action": True}
        )
        assert self_emote.is_self_speech_echo() is False

    def test_self_movement_returns_false(self):
        """Self-movement should return False (only speech is filtered)."""
        self_movement = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Andi",
            actor_id="ai_andi",
            actor_type=ActorType.AI,
            room_id="room2",
            room_name="New Room",
            content="arrives from the south",
            metadata={"is_self_action": True}
        )
        assert self_movement.is_self_speech_echo() is False

    def test_self_object_action_returns_false(self):
        """Self-object action should return False (only speech is filtered)."""
        self_object = MUDEvent(
            event_type=EventType.OBJECT,
            actor="Andi",
            actor_id="ai_andi",
            actor_type=ActorType.AI,
            room_id="room1",
            room_name="Test Room",
            content="picks up a flower",
            target="flower",
            metadata={"is_self_action": True}
        )
        assert self_object.is_self_speech_echo() is False

    def test_other_speech_returns_false(self):
        """Other's speech should return False."""
        other_speech = MUDEvent(
            event_type=EventType.SPEECH,
            actor="Bob",
            actor_id="player_bob",
            actor_type=ActorType.PLAYER,
            room_id="room1",
            room_name="Test Room",
            content="Hi there",
            metadata={"is_self_action": False}
        )
        assert other_speech.is_self_speech_echo() is False

    def test_speech_without_self_action_flag_returns_false(self):
        """Speech without is_self_action flag should return False."""
        speech_no_flag = MUDEvent(
            event_type=EventType.SPEECH,
            actor="Charlie",
            actor_id="npc_charlie",
            actor_type=ActorType.NPC,
            room_id="room1",
            room_name="Test Room",
            content="Greetings",
            metadata={}
        )
        assert speech_no_flag.is_self_speech_echo() is False

    def test_speech_with_self_action_false_returns_false(self):
        """Speech with explicit is_self_action=False should return False."""
        speech_not_self = MUDEvent(
            event_type=EventType.SPEECH,
            actor="Dave",
            actor_id="player_dave",
            actor_type=ActorType.PLAYER,
            room_id="room1",
            room_name="Test Room",
            content="Hello",
            metadata={"is_self_action": False}
        )
        assert speech_not_self.is_self_speech_echo() is False

    def test_system_event_returns_false(self):
        """System events should never be filtered."""
        system_event = MUDEvent(
            event_type=EventType.SYSTEM,
            actor="System",
            actor_id="system",
            actor_type=ActorType.SYSTEM,
            room_id="room1",
            room_name="Test Room",
            content="A notification appears.",
            metadata={"is_self_action": True}  # Even with flag
        )
        assert system_event.is_self_speech_echo() is False

    def test_self_speech_with_additional_metadata(self):
        """Self-speech with additional metadata should still return True."""
        self_speech = MUDEvent(
            event_type=EventType.SPEECH,
            actor="Andi",
            actor_id="ai_andi",
            actor_type=ActorType.AI,
            room_id="room1",
            room_name="Test Room",
            content="Hello",
            metadata={
                "is_self_action": True,
                "some_other_field": "value",
                "timestamp": "2025-01-09T12:00:00Z"
            }
        )
        assert self_speech.is_self_speech_echo() is True
