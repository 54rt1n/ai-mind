# aim/dreamer/dialogue/models.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Pydantic models for dialogue flow execution."""

from datetime import datetime
from enum import Enum
from typing import Optional, Literal, Any
from pydantic import BaseModel, Field, field_serializer

from ..models import StepConfig, StepOutput, StepMemory, ContextAction


class SpeakerType(str, Enum):
    """Type of speaker in a dialogue step."""
    ASPECT = "aspect"
    PERSONA = "persona"


class DialogueSpeaker(BaseModel):
    """Speaker configuration for a dialogue step.

    Defines who speaks in a given step - either an aspect of the persona
    (like 'coder' or 'psychologist') or the persona themselves.
    """
    type: SpeakerType
    aspect_name: Optional[str] = None
    """Required when type is 'aspect'. Name of the aspect (e.g., 'coder', 'librarian')."""

    def get_speaker_id(self, persona_id: str) -> str:
        """Get unique speaker identifier string.

        Args:
            persona_id: The persona's ID for persona-type speakers

        Returns:
            Speaker ID in format 'aspect:{name}' or 'persona:{id}'
        """
        if self.type == SpeakerType.ASPECT:
            return f"aspect:{self.aspect_name}"
        return f"persona:{persona_id}"


class DialogueStep(BaseModel):
    """Definition of a single step in a dialogue strategy.

    Each step defines who speaks (speaker), what prompt template to use,
    optional guidance for output formatting, and standard step configuration.
    """
    id: str
    speaker: DialogueSpeaker
    prompt: str
    """Jinja2 template for the step prompt."""

    guidance: Optional[str] = None
    """Optional output guidance. Falls off after one turn (not accumulated)."""

    config: StepConfig = Field(default_factory=StepConfig)
    output: StepOutput
    memory: StepMemory = Field(default_factory=StepMemory)
    context: Optional[list[ContextAction]] = None
    """Context DSL actions for this step (same as standard scenarios)."""

    next: list[str] = Field(default_factory=list)
    """IDs of steps to execute after this one."""


class DialogueConfig(BaseModel):
    """Dialogue-specific configuration at the strategy level.

    Defines the primary aspect guiding the conversation, who speaks first,
    and the scene template that persists across all turns.
    """
    primary_aspect: str
    """Name of the primary aspect guiding the dialogue (e.g., 'coder')."""

    initial_speaker: SpeakerType
    """Who speaks first in the dialogue."""

    scene_template: str
    """Jinja2 template for the scene. Persists (prepended to each turn)."""


class ScenarioContext(BaseModel):
    """Scenario-level context configuration (shared with standard scenarios)."""
    required_aspects: list[str] = Field(default_factory=list)
    core_documents: list[str] = Field(default_factory=list)
    enhancement_documents: list[str] = Field(default_factory=list)
    location: str = ""
    thoughts: list[str] = Field(default_factory=list)


class SeedAction(BaseModel):
    """Initial data loading action."""
    action: Literal["load_conversation", "query_memories"]
    target: Optional[str] = "current"
    document_types: Optional[list[str]] = None
    exclude_types: Optional[list[str]] = None
    top_n: Optional[int] = None


class DialogueTurn(BaseModel):
    """A single turn in the dialogue history.

    Records who spoke, what they said, and metadata for tracking.
    """
    speaker_id: str
    """Speaker identifier: 'aspect:coder' or 'persona:andi'."""

    content: str
    """The generated response content."""

    think: Optional[str] = None
    """Extracted think content from model response."""

    step_id: str
    """ID of the step that generated this turn."""

    doc_id: str
    """Document ID for CVM storage."""

    document_type: str
    """Document type for storage. Aspect turns use 'dialogue:{aspect_name}',
    persona turns use the step's output.document_type."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer('timestamp')
    def serialize_timestamp(self, dt: datetime, _info: Any) -> str:
        """Serialize datetime to ISO format string."""
        return dt.isoformat()


class DialogueState(BaseModel):
    """Tracks execution state of a dialogue scenario.

    Maintains the accumulated turns, current position, and context
    needed for role flipping and turn construction.
    """
    # Identity
    pipeline_id: str
    strategy_name: str
    conversation_id: Optional[str] = None
    persona_id: str
    user_id: str

    # Model configuration
    model: str
    thought_model: Optional[str] = None
    codex_model: Optional[str] = None

    # Runtime context
    guidance: Optional[str] = None
    """User-provided guidance for the scenario."""

    query_text: Optional[str] = None
    """Query text for memory searches."""

    persona_mood: Optional[str] = None

    # Context accumulation (compatible with PipelineState for executor functions)
    context_doc_ids: list[str] = Field(default_factory=list)
    """Accumulated context document IDs from prior steps."""

    context_documents: Optional[list[dict]] = None
    """Pre-provided context documents (optional)."""

    # Execution state
    branch: int = 0
    """Branch number for conversation storage."""

    # Dialogue tracking
    turns: list[DialogueTurn] = Field(default_factory=list)
    """Accumulated dialogue turns."""

    current_step_id: Optional[str] = None
    """Currently executing step."""

    step_counter: int = 1
    """Step counter for sequence numbering."""

    completed_steps: list[str] = Field(default_factory=list)
    """IDs of completed steps."""

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer('created_at', 'updated_at')
    def serialize_datetime(self, dt: datetime, _info: Any) -> str:
        """Serialize datetime to ISO format string."""
        return dt.isoformat()

    def get_last_turn(self) -> Optional[DialogueTurn]:
        """Get the most recent turn, if any."""
        return self.turns[-1] if self.turns else None

    def add_turn(self, turn: DialogueTurn) -> None:
        """Add a turn to the dialogue history."""
        self.turns.append(turn)
        self.updated_at = datetime.utcnow()
