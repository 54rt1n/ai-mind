# aim/dreamer/core/models.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Pydantic models for pipeline state, steps, and scenarios."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Literal, Any, Union, Annotated, TYPE_CHECKING
from pydantic import BaseModel, Field, field_serializer, model_validator, Discriminator

if TYPE_CHECKING:
    from aim.llm.model_set import ModelSet


# --- Speaker types (shared with dialogue system) ---

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

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_serializer('timestamp')
    def serialize_timestamp(self, dt: datetime, _info: Any) -> str:
        """Serialize datetime to ISO format string."""
        return dt.isoformat()


class StepStatus(str, Enum):
    """Status of a pipeline step in the DAG."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class FormatValidation(BaseModel):
    """Configuration for response format validation.

    Enables automatic validation and retry for response formatting
    requirements like emotional state headers.
    """
    require_emotional_header: bool = True
    """Whether to require [== ... Emotional State ... ==] header."""

    max_retries: int = 3
    """Maximum retries with guidance injection before giving up."""

    fallback_model: Optional[str] = "fallback"
    """Model role to try after max_retries exhausted. None to disable fallback."""

    persona_name_override: Optional[str] = None
    """Override persona name for guidance messages.
    If None, uses persona.full_name from executor context."""


class StepConfig(BaseModel):
    """Step execution configuration."""
    max_tokens: int = 1024
    link_guidance: bool = False
    is_thought: bool = False  # DEPRECATED: Use model_role="thought" instead
    is_codex: bool = False    # DEPRECATED: Use model_role="codex" instead
    temperature: Optional[float] = None
    model_override: Optional[str] = None
    model_role: Optional[str] = None  # Model role from ModelSet (analysis, codex, writing, thought, etc.)
    max_iterations: Optional[int] = None  # Max times this step can execute
    on_limit: Optional[str] = None        # Where to go when max_iterations hit
    tool_retries: int = 3                 # Retries if LLM doesn't call a tool
    format_validation: FormatValidation = Field(default_factory=FormatValidation)
    """Format validation configuration for response retries. Enabled by default."""

    def get_model(self, model_set: "ModelSet") -> str:
        """Resolve which model this step will use given a ModelSet.

        Priority:
        1. model_override (explicit)
        2. model_role (resolved via ModelSet)
        3. is_thought (legacy)
        4. is_codex (legacy)
        5. model_set.default_model
        """
        if self.model_override:
            return self.model_override
        if self.model_role:
            return model_set.get_model_name(self.model_role)
        if self.is_thought:
            return model_set.thought_model
        if self.is_codex:
            return model_set.codex_model
        return model_set.default_model


class StepOutput(BaseModel):
    """Output configuration for a step."""
    document_type: str
    weight: float = 1.0
    add_to_turns: bool = True


class ScenarioTool(BaseModel):
    """Tool definition within a scenario."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema-like parameter definitions
    required: list[str] = Field(default_factory=list)


class Condition(BaseModel):
    """A single condition for conditional flow control.

    Used by tool_calling steps to determine next step based on tool results.

    Either:
    - source/condition/target/goto for conditional matching
    - default for fallback routing

    Optional:
    - collect_to: Collection name to append tool_result (independent of matching)
    """
    source: Optional[str] = None      # Field path: tool_result.accept, tool_name
    condition: Optional[str] = None   # Operator: ==, !=, in, not_in
    target: Optional[Any] = None      # Value to compare against (str or list[str])
    goto: Optional[str] = None        # Step ID, "end", or "abort"
    default: Optional[str] = None     # Default goto (mutually exclusive with above)
    collect_to: Optional[str] = None  # Collection name to append tool_result

    @model_validator(mode="after")
    def validate_condition(self) -> "Condition":
        """Ensure either source/condition/target/goto OR default is specified."""
        has_conditional = self.source and self.condition and self.goto
        has_default = self.default is not None

        if has_conditional and has_default:
            raise ValueError("Condition cannot have both conditional fields and default")
        if not has_conditional and not has_default:
            raise ValueError("Condition must have either source/condition/goto or default")

        # Validate operator
        if self.condition and self.condition not in ("==", "!=", "in", "not_in"):
            raise ValueError(f"Invalid condition operator: {self.condition}")

        return self


class MemoryAction(BaseModel):
    """Unified memory operation for the Memory DSL.

    Used in both seed-level and step-level context building.
    Actions execute sequentially, accumulating doc_ids.

    Action Categories:
    - Retrieval: load_conversation, get_memory, search_memories
    - Transform: sort, filter, truncate, drop
    - Meta: flush, clear
    """
    action: Literal[
        # Retrieval
        "load_conversation",  # Bulk load from conversation
        "get_memory",         # Direct retrieval by criteria (no semantic search)
        "search_memories",    # Semantic vector search
        # Transform
        "sort",               # Reorder accumulated docs
        "filter",             # Keep matching docs
        "truncate",           # Limit to N docs
        "drop",               # Remove docs/types
        # Meta
        "flush",              # Clear accumulated context
        "clear",              # Alias for flush
    ]

    # === Common params ===
    target: Optional[str] = "current"
    """Target conversation: 'current' uses state.conversation_id"""

    document_types: Optional[list[str]] = None
    """Document types to include (None = all)"""

    exclude_types: Optional[list[str]] = None
    """Document types to exclude"""

    top_n: Optional[int] = None
    """Maximum documents to return"""

    conversation_id: Optional[str] = None
    """Conversation to query: 'current', 'all', or specific ID"""

    # === search_memories params ===
    query_text: Optional[str] = None
    """Explicit query text for semantic search"""

    use_context: bool = False
    """If True, use accumulated content as query for search_memories"""

    temporal_decay: Optional[float] = None
    """Temporal decay factor 0.0-1.0 (default 0.5)"""

    diversity: Optional[float] = None
    """MMR diversity factor 0.0-1.0"""

    chunk_level: Optional[str] = None
    """Chunk level: 'chunk_256', 'chunk_768', 'full'"""

    # === sort params ===
    by: Optional[str] = None
    """Sort field: 'timestamp' or 'relevance'"""

    direction: Optional[str] = None
    """Sort direction: 'ascending' (oldest first) or 'descending' (newest first)"""

    # === truncate params ===
    limit: Optional[int] = None
    """Maximum documents to keep after truncate"""

    # === filter/drop params ===
    match: Optional[str] = None
    """Filter pattern for filter action"""

    doc_ids: Optional[list[str]] = None
    """Specific document IDs to drop"""

    # === min_memories threshold ===
    min_memories: Optional[int] = None
    """Minimum memories threshold: only execute if accumulated docs < this value"""




class StepDefinition(BaseModel):
    """Definition of a single pipeline step."""
    id: str
    prompt: str
    config: StepConfig = Field(default_factory=StepConfig)
    output: StepOutput
    next: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    input_refs: list[str] = Field(default_factory=list)
    """Step IDs whose outputs should be loaded from CVM as context.
    If empty, defaults to depends_on steps."""

    context: Optional[list[MemoryAction]] = None
    """Context preparation DSL - sequential actions to build step input.
    If provided, replaces seed_doc_ids for this step."""


# --- New Step Definition Types (for strategy pattern) ---

class BaseStepDefinition(BaseModel):
    """Base serializable typed step definition.

    Subclasses MUST override `type` with a narrow Literal and default value
    for Pydantic discriminated union parsing to work.
    """
    id: str
    type: str  # Discriminator field - subclasses narrow this
    depends_on: list[str] = Field(default_factory=list)  # Steps that must complete first


class ContextOnlyStepDefinition(BaseStepDefinition):
    """Memory actions only - no LLM, no output, no conditionals."""
    type: Literal["context_only"] = "context_only"  # Discriminator
    context: Optional[list[MemoryAction]] = None  # Memory DSL actions
    next: list[str]                       # Required: next step ID(s)


class StandardStepDefinition(BaseStepDefinition):
    """LLM prose generation - produces document output."""
    type: Literal["standard"] = "standard"  # Discriminator
    prompt: str
    context: Optional[list[MemoryAction]] = None  # Memory DSL actions
    config: StepConfig = Field(default_factory=StepConfig)
    output: StepOutput                    # Required: document_type, weight, add_to_turns
    next: list[str]                       # Required: next step ID(s)


class ToolCallingStepDefinition(BaseStepDefinition):
    """Tool-only LLM step - no output, conditionals required."""
    type: Literal["tool_calling"] = "tool_calling"  # Discriminator
    prompt: str
    context: Optional[list[MemoryAction]] = None  # Memory DSL actions
    tools: list[str]                      # Tool names (from framework.tools)
    next_conditions: list[Condition]      # Required - no 'next' field
    config: StepConfig = Field(default_factory=StepConfig)


class RenderingStepDefinition(BaseStepDefinition):
    """Template rendering - produces document from template, no LLM."""
    type: Literal["rendering"] = "rendering"  # Discriminator
    template: str                         # Jinja2 template
    output: StepOutput                    # Required: document_type, weight, add_to_turns
    next: list[str]                       # Required: next step ID(s)


class DialogueStepDefinition(BaseStepDefinition):
    """LLM dialogue with speaker-based role flipping.

    Used for persona/aspect dialogues where roles flip based on who speaks:
    - When ASPECT speaks: aspects='assistant', persona='user'
    - When PERSONA speaks: aspects='user', persona='assistant'
    """
    type: Literal["dialogue"] = "dialogue"  # Discriminator
    speaker: DialogueSpeaker              # Who speaks (aspect or persona)
    guidance: str = ""                    # Jinja2 template for step instructions
    scene_template: Optional[str] = None  # Optional scene template
    context: Optional[list[MemoryAction]] = None  # Memory DSL actions
    config: StepConfig = Field(default_factory=StepConfig)
    output: StepOutput                    # Required: document_type, weight
    next: list[str] = Field(default_factory=list)


# Type alias for discriminated union of step definitions
NewStepDefinition = Annotated[
    Union[
        ContextOnlyStepDefinition,
        StandardStepDefinition,
        ToolCallingStepDefinition,
        RenderingStepDefinition,
        DialogueStepDefinition,
    ],
    Discriminator('type')  # Pydantic uses 'type' field to pick correct class
]


class ScenarioContext(BaseModel):
    """Scenario-level context configuration."""
    required_aspects: list[str]
    core_documents: list[str] = Field(default_factory=list)
    enhancement_documents: list[str] = Field(default_factory=list)
    location: str = ""
    thoughts: list[str] = Field(default_factory=list)




class Scenario(BaseModel):
    """Complete scenario definition from YAML."""
    name: str
    version: int = 2
    flow: Optional[Literal["standard", "dialogue"]] = "standard"
    """Flow type: 'standard' uses executor, 'dialogue' uses DialogueScenario."""
    description: str = ""
    requires_conversation: bool = True
    """Whether this scenario requires an existing conversation.
    True for analyst/summarizer, False for journaler/dreamer/philosopher."""
    context: ScenarioContext
    seed: list[MemoryAction] = Field(default_factory=list)
    steps: dict[str, StepDefinition]

    def get_root_steps(self) -> list[str]:
        """Return steps with no dependencies (DAG entry points)."""
        roots = []
        for step_id, step in self.steps.items():
            # Check if this step has dependencies
            has_deps = bool(step.depends_on)

            # If no explicit depends_on, check if it's referenced in any next list
            if not has_deps:
                for other_step in self.steps.values():
                    if step_id in other_step.next:
                        has_deps = True
                        break

            # If no dependencies found, it's a root step
            if not has_deps:
                roots.append(step_id)

        return roots

    def get_downstream(self, step_id: str) -> list[str]:
        """Return steps that depend on the given step."""
        return [s_id for s_id, step_def in self.steps.items() if step_id in step_def.depends_on]

    def compute_dependencies(self) -> None:
        """Infer depends_on from next edges if not explicitly set."""
        for step_id, step in self.steps.items():
            if not step.depends_on:
                # Find all steps that have this step in their 'next'
                step.depends_on = [
                    s_id for s_id, s_def in self.steps.items()
                    if step_id in s_def.next
                ]

    def topological_order(self) -> list[str]:
        """Return steps in topological execution order."""
        # Build dependency graph
        in_degree = {step_id: len(step.depends_on) for step_id, step in self.steps.items()}
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Sort queue for deterministic ordering
            queue.sort()
            current = queue.pop(0)
            result.append(current)

            # Reduce in-degree for downstream steps
            for downstream_id in self.get_downstream(current):
                in_degree[downstream_id] -= 1
                if in_degree[downstream_id] == 0:
                    queue.append(downstream_id)

        # Check for cycles
        if len(result) != len(self.steps):
            raise ValueError("Scenario contains a cycle in step dependencies")

        return result


class StepResult(BaseModel):
    """Result of a single step execution."""
    step_id: str
    response: str
    think: Optional[str] = None
    tool_name: Optional[str] = None      # Which tool was called (tool_calling steps)
    tool_result: Optional[dict[str, Any]] = None   # Tool call arguments (tool_calling steps)
    doc_id: str
    document_type: str
    document_weight: float
    tokens_used: int
    timestamp: datetime

    @field_serializer('timestamp')
    def serialize_timestamp(self, dt: datetime, _info: Any) -> str:
        """Serialize datetime to ISO format string."""
        return dt.isoformat()


class StepJob(BaseModel):
    """Message format for the Redis step queue."""
    pipeline_id: str
    step_id: str
    attempt: int = 1
    max_attempts: int = 3
    enqueued_at: datetime
    priority: int = 0

    def increment_attempt(self) -> 'StepJob':
        """Return a new StepJob with incremented attempt counter."""
        return self.model_copy(update={'attempt': self.attempt + 1})

    @field_serializer('enqueued_at')
    def serialize_enqueued_at(self, dt: datetime, _info: Any) -> str:
        """Serialize datetime to ISO format string."""
        return dt.isoformat()


class PipelineState(BaseModel):
    """Serializable pipeline execution state stored in Redis."""

    # Identity
    pipeline_id: str
    scenario_name: str
    conversation_id: Optional[str] = None
    """Conversation ID. Required for analyst/summarizer, optional for journaler/dreamer/philosopher."""
    persona_id: str
    user_id: str

    # Config references
    model: str
    thought_model: Optional[str] = None
    codex_model: Optional[str] = None

    # Runtime context
    guidance: Optional[str] = None
    query_text: Optional[str] = None
    thought_content: Optional[str] = None  # Externally injected thought
    persona_mood: Optional[str] = None

    # Execution state
    branch: int
    step_counter: int = 1
    extra: list[str] = Field(default_factory=list)

    # DAG tracking - doc_id references to CVM
    completed_steps: list[str] = Field(default_factory=list)
    step_doc_ids: dict[str, str] = Field(default_factory=dict)
    """Maps step_id -> doc_id for loading outputs from CVM."""

    # Seed data references (message doc_ids loaded at pipeline start)
    seed_doc_ids: dict[str, list[str]] = Field(default_factory=dict)
    """Maps step_id -> list of doc_ids from seed actions."""

    # Accumulated context (passed from step to step)
    context_doc_ids: list[str] = Field(default_factory=list)
    """Accumulated context: initial context from first step's DSL + all step outputs.
    Each step receives this full context and appends its own output."""

    # Pre-provided context documents (from refiner or external source)
    context_documents: Optional[list[dict]] = None
    """Pre-provided context documents passed at pipeline start.
    If set, first step uses these instead of executing context DSL."""

    # Timestamps
    created_at: datetime
    updated_at: datetime

    @field_serializer('created_at', 'updated_at')
    def serialize_datetime(self, dt: datetime, _info: Any) -> str:
        """Serialize datetime to ISO format string."""
        return dt.isoformat()
