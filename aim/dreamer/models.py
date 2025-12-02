# aim/dreamer/models.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Pydantic models for pipeline state, steps, and scenarios."""

from datetime import datetime
from enum import Enum
from typing import Optional, Literal, Any
from pydantic import BaseModel, Field, field_serializer


class StepStatus(str, Enum):
    """Status of a pipeline step in the DAG."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class StepConfig(BaseModel):
    """Step execution configuration."""
    max_tokens: int = 1024
    use_guidance: bool = False
    is_thought: bool = False
    is_codex: bool = False
    temperature: Optional[float] = None
    model_override: Optional[str] = None


class StepMemory(BaseModel):
    """Memory query configuration for a step."""
    top_n: int = 0
    document_type: Optional[list[str]] = None
    flush_before: bool = False
    sort_by: str = "relevance"


class StepOutput(BaseModel):
    """Output configuration for a step."""
    document_type: str
    weight: float = 1.0
    add_to_turns: bool = True


class ContextAction(BaseModel):
    """Single action in context preparation DSL.

    Actions are executed sequentially to build up the step's input context.
    Supported actions:
    - load_conversation: Load documents from current conversation
    - query: Query documents across conversations
    - sort: Sort accumulated documents
    - filter: Filter accumulated documents (future)
    """
    action: Literal["load_conversation", "query", "sort", "filter"]

    # load_conversation params
    target: Optional[str] = "current"
    """Target conversation: 'current' uses state.conversation_id"""

    # Shared document type params
    document_types: Optional[list[str]] = None
    """Document types to include (None = all)"""

    exclude_types: Optional[list[str]] = None
    """Document types to exclude"""

    # query params
    top_n: Optional[int] = None
    """Maximum documents to return from query"""

    conversation_id: Optional[str] = None
    """Conversation to query: 'current', 'all', or specific ID"""

    # sort params
    by: Optional[str] = None
    """Sort field: 'timestamp' or 'relevance'"""

    direction: Optional[str] = None
    """Sort direction: 'ascending' (oldest first) or 'descending' (newest first)"""

    # filter params (future)
    match: Optional[str] = None
    """Filter pattern"""


class StepDefinition(BaseModel):
    """Definition of a single pipeline step."""
    id: str
    prompt: str
    config: StepConfig = Field(default_factory=StepConfig)
    output: StepOutput
    memory: StepMemory = Field(default_factory=StepMemory)
    next: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    input_refs: list[str] = Field(default_factory=list)
    """Step IDs whose outputs should be loaded from CVM as context.
    If empty, defaults to depends_on steps."""

    context: Optional[list[ContextAction]] = None
    """Context preparation DSL - sequential actions to build step input.
    If provided, replaces seed_doc_ids for this step."""


class ScenarioContext(BaseModel):
    """Scenario-level context configuration."""
    required_aspects: list[str]
    core_documents: list[str] = Field(default_factory=list)
    enhancement_documents: list[str] = Field(default_factory=list)
    location: str = ""
    thoughts: list[str] = Field(default_factory=list)


class SeedAction(BaseModel):
    """Initial data loading action."""
    action: Literal["load_conversation", "query_memories"]
    accumulate_to: str
    params: dict = Field(default_factory=dict)


class Scenario(BaseModel):
    """Complete scenario definition from YAML."""
    name: str
    version: int = 2
    description: str = ""
    requires_conversation: bool = True
    """Whether this scenario requires an existing conversation.
    True for analyst/summarizer, False for journaler/dreamer/philosopher."""
    context: ScenarioContext
    seed: list[SeedAction] = Field(default_factory=list)
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

    # Timestamps
    created_at: datetime
    updated_at: datetime

    @field_serializer('created_at', 'updated_at')
    def serialize_datetime(self, dt: datetime, _info: Any) -> str:
        """Serialize datetime to ISO format string."""
        return dt.isoformat()
