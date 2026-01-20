# aim/dreamer/core/state.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""ScenarioState - runtime execution state for strategy-based scenarios."""

from typing import Optional, Any
from pydantic import BaseModel, Field

from .models import StepResult, DialogueTurn, SpeakerType


class DocRef(BaseModel):
    """Reference to a document or chunk in the CVM.

    Can reference either a full document or a specific chunk within a document.
    Used for tracking memory context loaded during step execution.

    Attributes:
        doc_id: The document identifier
        document_type: Type of document (e.g., "conversation", "journal")
        parent_doc_id: If this is a chunk, the parent document's ID
        chunk_level: Chunk granularity ("full", "chunk_256", "chunk_768")
        chunk_index: Which chunk within the document (0-indexed)
    """
    doc_id: str
    document_type: Optional[str] = None
    parent_doc_id: Optional[str] = None  # If this is a chunk
    chunk_level: Optional[str] = None    # "full", "chunk_256", "chunk_768"
    chunk_index: Optional[int] = None    # Which chunk (0-indexed)

    @classmethod
    def from_row(cls, row: dict) -> "DocRef":
        """Create DocRef from a CVM query result row.

        Args:
            row: Dictionary with doc_id and optional chunk metadata

        Returns:
            DocRef instance
        """
        return cls(
            doc_id=row["doc_id"],
            document_type=row.get("document_type"),
            parent_doc_id=row.get("parent_doc_id"),
            chunk_level=row.get("chunk_level"),
            chunk_index=row.get("chunk_index"),
        )


class ScenarioTurn(BaseModel):
    """A turn in the scenario conversation - prompt/response pair.

    Each turn represents one LLM interaction during scenario execution.
    Stored in ScenarioState.turns to maintain conversation history.

    Attributes:
        step_id: Which step produced this turn
        prompt: The rendered prompt (becomes user turn)
        response: The LLM response (becomes assistant turn)
    """
    step_id: str
    prompt: str      # The rendered prompt (becomes user turn)
    response: str    # The LLM response (becomes assistant turn)


class ScenarioState(BaseModel):
    """Runtime execution state. Serialized to Redis, restored on resume.

    Together with ScenarioFramework, forms complete dream execution context.
    State is mutable during execution - framework is immutable.

    Attributes:
        current_step: Current step ID (or "end"/"abort" when complete)
        turns: Conversation history as prompt/response pairs
        memory_refs: Chunked docs from context DSL searches (refreshed per step)
        step_doc_ids: Document IDs created by this scenario
        step_results: Results keyed by step_id for template access
        collections: Accumulated tool results for iterating steps
        step_iterations: Iteration counts for max_iterations tracking
        guidance: External guidance text passed at start
        query_text: Query/topic text passed at start
        conversation_id: Target conversation ID
    """

    # Execution position
    current_step: str                    # Current step ID (or "end"/"abort")

    # Conversation history - each turn is a (prompt, response) pair
    # Guarantees proper user/assistant alternation
    turns: list[ScenarioTurn] = Field(default_factory=list)

    # Memory context - chunked docs from context DSL searches (refreshed per step)
    memory_refs: list[DocRef] = Field(default_factory=list)

    # Step outputs - doc IDs created by this scenario (for CVM persistence)
    step_doc_ids: list[str] = Field(default_factory=list)

    # Step results (for template access: {{ steps.select_topic.tool_result }})
    step_results: dict[str, StepResult] = Field(default_factory=dict)

    # Collections (for accumulating tool results: {{ collections.tasks }})
    collections: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

    # Iteration tracking (for max_iterations limit)
    step_iterations: dict[str, int] = Field(default_factory=dict)

    # Context passed at pipeline start
    guidance: Optional[str] = None       # External guidance text
    query_text: Optional[str] = None     # Query/topic text
    conversation_id: Optional[str] = None

    # Dialogue tracking (used only by dialogue steps)
    dialogue_turns: list[DialogueTurn] = Field(default_factory=list)
    """Accumulated dialogue turns with speaker metadata."""
    last_aspect_name: Optional[str] = None
    """Name of the most recent aspect speaker (for scene generation)."""

    @classmethod
    def initial(
        cls,
        first_step: str,
        conversation_id: Optional[str] = None,
        guidance: Optional[str] = None,
        query_text: Optional[str] = None,
    ) -> "ScenarioState":
        """Create initial state for starting a scenario.

        Args:
            first_step: Entry point step ID from ScenarioFramework
            conversation_id: Target conversation (optional)
            guidance: External guidance text (optional)
            query_text: Query/topic text (optional)

        Returns:
            New ScenarioState ready for execution
        """
        return cls(
            current_step=first_step,
            conversation_id=conversation_id,
            guidance=guidance,
            query_text=query_text,
        )

    def is_complete(self) -> bool:
        """Check if scenario has reached a terminal state.

        Returns:
            True if current_step is "end" or "abort"
        """
        return self.current_step in ("end", "abort")

    def is_aborted(self) -> bool:
        """Check if scenario was aborted.

        Returns:
            True if current_step is "abort"
        """
        return self.current_step == "abort"

    def record_turn(self, step_id: str, prompt: str, response: str) -> None:
        """Record a turn in the conversation history.

        Args:
            step_id: Step that produced this turn
            prompt: The rendered prompt
            response: The LLM response
        """
        self.turns.append(ScenarioTurn(
            step_id=step_id,
            prompt=prompt,
            response=response,
        ))

    def record_step_result(self, result: StepResult) -> None:
        """Record a step result for template access.

        Args:
            result: The StepResult to record
        """
        self.step_results[result.step_id] = result

    def increment_iteration(self, step_id: str) -> int:
        """Increment and return iteration count for a step.

        Args:
            step_id: Step to track

        Returns:
            New iteration count
        """
        current = self.step_iterations.get(step_id, 0)
        self.step_iterations[step_id] = current + 1
        return current + 1

    def collect_result(self, collection_name: str, result: dict[str, Any]) -> None:
        """Add a result to a named collection.

        Args:
            collection_name: Name of collection (e.g., "tasks")
            result: Tool result dict to collect
        """
        if collection_name not in self.collections:
            self.collections[collection_name] = []
        self.collections[collection_name].append(result)

    def add_doc_id(self, doc_id: str) -> None:
        """Add a document ID created by this scenario.

        Args:
            doc_id: The document ID to track
        """
        if doc_id not in self.step_doc_ids:
            self.step_doc_ids.append(doc_id)

    def clear_memory_refs(self) -> None:
        """Clear memory refs before loading new context."""
        self.memory_refs = []

    def build_template_context(self) -> dict[str, Any]:
        """Build Jinja2 context from state for template rendering.

        Returns:
            Dictionary with:
            - steps: Dict of step_id -> {tool_result, tool_name, response}
            - collections: Dict of collection_name -> list of results
            - guidance: External guidance
            - query_text: Query/topic
            - conversation_id: Current conversation
        """
        return {
            # Step results for cross-step references: {{ steps.select_topic.tool_result }}
            'steps': {
                step_id: {
                    'tool_result': result.tool_result,
                    'tool_name': result.tool_name,
                    'response': result.response,
                }
                for step_id, result in self.step_results.items()
            },
            # Collections: {{ collections.tasks }}
            'collections': self.collections,
            # Context
            'guidance': self.guidance,
            'query_text': self.query_text,
            'conversation_id': self.conversation_id,
        }

    # --- Dialogue-specific methods ---

    def add_dialogue_turn(self, turn: DialogueTurn) -> None:
        """Add a dialogue turn and track aspect changes.

        Args:
            turn: The DialogueTurn to add
        """
        self.dialogue_turns.append(turn)
        # Track last aspect for scene generation at aspect changes
        if turn.speaker_id.startswith("aspect:"):
            self.last_aspect_name = turn.speaker_id.split(":", 1)[1]

    def get_dialogue_role(self, speaker_id: str, current_is_persona: bool) -> str:
        """Determine user/assistant role based on speaker perspective (role flipping).

        When ASPECT speaks: aspects='assistant', persona='user'
        When PERSONA speaks: aspects='user', persona='assistant'

        Args:
            speaker_id: Speaker identifier (e.g., 'aspect:coder' or 'persona:andi')
            current_is_persona: Whether the current step's speaker is persona

        Returns:
            'user' or 'assistant'
        """
        turn_is_persona = speaker_id.startswith("persona:")
        if current_is_persona:
            return 'assistant' if turn_is_persona else 'user'
        return 'user' if turn_is_persona else 'assistant'
