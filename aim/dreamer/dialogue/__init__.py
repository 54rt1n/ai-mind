# aim/dreamer/dialogue/__init__.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Dialogue flow module for AI-Mind scenarios.

This module provides a dialogue-based scenario execution system where
two speakers (persona and aspect) alternate turns with proper context
engineering - each participant is the assistant responding to the other as user.

Classes:
    DialogueStrategy: Loads and represents dialogue configuration from YAML
    DialogueScenario: Executes a dialogue strategy with role flipping
    DialogueState: Tracks execution state of a dialogue
    DialogueTurn: Represents a single turn in the dialogue
    DialogueSpeaker: Speaker configuration for a step
    DialogueStep: Definition of a single dialogue step
"""

from .models import (
    DialogueState,
    DialogueStep,
    DialogueSpeaker,
    DialogueTurn,
    DialogueConfig,
    SpeakerType,
    ScenarioContext,
    SeedAction,
)
from .strategy import DialogueStrategy
from .scenario import DialogueScenario

__all__ = [
    # Core classes
    'DialogueStrategy',
    'DialogueScenario',

    # State and tracking
    'DialogueState',
    'DialogueTurn',

    # Configuration
    'DialogueStep',
    'DialogueSpeaker',
    'DialogueConfig',
    'SpeakerType',
    'ScenarioContext',
    'SeedAction',
]
