# aim/dreamer/core/strategy/base.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Base classes for step execution strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING, Callable, Awaitable, Any

if TYPE_CHECKING:
    from ..framework import ScenarioFramework
    from ..state import ScenarioState
    from ..models import NewStepDefinition
    from aim.agents.persona import Persona
    from aim.config import ChatConfig
    from aim.conversation.model import ConversationModel
    from aim.llm.model_set import ModelSet
    from aim.llm.llm import LLMProvider


@dataclass
class ScenarioStepResult:
    """Result from executing a step.

    Executor uses this to determine dream status and next step.

    Attributes:
        success: Whether the step executed successfully
        next_step: Step ID, "end", or "abort"
        state_changed: Whether executor.state was mutated
        doc_created: Whether a document was created in CVM
        error: Optional error message if success is False
    """
    success: bool
    next_step: str  # Step ID, "end", or "abort"
    state_changed: bool = False  # Did we mutate executor.state?
    doc_created: bool = False    # Did we create a document?
    error: Optional[str] = None


@dataclass
class ScenarioExecutor:
    """Holds state, framework, and shared resources. Runs strategies.

    The executor is the context that strategies operate within. It provides
    access to all shared resources and maintains the mutable ScenarioState.

    Attributes:
        state: Mutable runtime state (current_step, turns, collections, etc.)
        framework: Immutable scenario definition (steps, tools)
        config: Chat configuration for LLM calls
        cvm: ConversationModel for memory operations
        persona: Agent persona for system prompts
        heartbeat_callback: Optional callback for liveness during LLM streaming
        model_set: ModelSet for resolving model roles
    """
    state: "ScenarioState"
    framework: "ScenarioFramework"
    config: "ChatConfig"
    cvm: "ConversationModel"
    persona: "Persona"
    heartbeat_callback: Optional[Callable[[], Awaitable[None]]] = None
    model_set: Optional["ModelSet"] = None

    def __post_init__(self):
        """Initialize model_set if not provided."""
        if self.model_set is None:
            from aim.llm.model_set import ModelSet
            self.model_set = ModelSet.from_config(self.config, self.persona)

    @classmethod
    def create(
        cls,
        state: "ScenarioState",
        framework: "ScenarioFramework",
        config: "ChatConfig",
        cvm: "ConversationModel",
        persona: "Persona",
        heartbeat_callback: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> "ScenarioExecutor":
        """Create executor with all required resources.

        Args:
            state: Runtime state (will be mutated by strategies)
            framework: Immutable scenario definition
            config: Chat configuration
            cvm: ConversationModel for memory
            persona: Agent persona
            heartbeat_callback: Optional liveness callback

        Returns:
            Configured ScenarioExecutor
        """
        return cls(
            state=state,
            framework=framework,
            config=config,
            cvm=cvm,
            persona=persona,
            heartbeat_callback=heartbeat_callback,
        )

    async def execute(self, strategy: "BaseStepStrategy") -> ScenarioStepResult:
        """Execute a strategy, update state, return result.

        Args:
            strategy: The strategy to execute (already initialized with step_def)

        Returns:
            ScenarioStepResult indicating success/failure and next step
        """
        from .functions import execute_seed_actions

        # Hydrate scenario-level seed context once before first step execution.
        execute_seed_actions(self)

        result = await strategy.execute()
        self.state.current_step = result.next_step
        return result

    def insert_message(self, message: "ConversationMessage") -> None:
        """Insert message to CVM with vectorizer lifecycle management.

        Automatically loads/releases vectorizer for chunking when skip_vectorizer=True.

        Args:
            message: ConversationMessage to insert.
        """
        self.cvm.load_vectorizer()
        try:
            self.cvm.insert(message)
        finally:
            self.cvm.release_vectorizer()


class BaseStepStrategy(ABC):
    """Abstract base class for step execution strategies.

    Strategies implement the execute() method to perform step-specific logic.
    Each strategy type handles different step types:
    - ContextOnlyStrategy: Memory DSL operations only
    - StandardStrategy: LLM prose generation
    - ToolCallingStrategy: LLM with tools and conditionals
    - RenderingStrategy: Jinja2 template rendering

    Attributes:
        executor: The ScenarioExecutor providing shared resources
        step_def: The step definition being executed
        llm_provider: Optional LLM provider for steps that need LLM calls
    """

    def __init__(
        self,
        executor: ScenarioExecutor,
        step_def: "NewStepDefinition",
        llm_provider: Optional["LLMProvider"] = None,
    ):
        """Initialize strategy with executor and step definition.

        Args:
            executor: ScenarioExecutor with state, framework, and resources
            step_def: The step definition to execute
            llm_provider: Optional LLM provider (some strategies don't need it)
        """
        self.executor = executor
        self.step_def = step_def
        self.llm_provider = llm_provider

    @abstractmethod
    async def execute(self) -> ScenarioStepResult:
        """Execute the step, mutate executor.state, return result with next step.

        Implementations should:
        1. Access executor.state, executor.framework, executor.cvm, executor.persona
        2. Use self.llm_provider for LLM calls (if applicable)
        3. Call heartbeat_callback during LLM streaming to keep turn alive
        4. Return ScenarioStepResult with next_step set appropriately

        Returns:
            ScenarioStepResult with success status and next step
        """
        pass

    def _get_next_step(self) -> str:
        """Get the next step from step_def.next.

        For steps with a 'next' field, returns the first item.

        Returns:
            Next step ID, or "end" if next is empty or ["end"]
        """
        if hasattr(self.step_def, 'next') and self.step_def.next:
            return self.step_def.next[0]
        return "end"

    async def _heartbeat(self) -> None:
        """Call heartbeat callback if configured."""
        if self.executor.heartbeat_callback:
            try:
                await self.executor.heartbeat_callback()
            except Exception:
                # Log but don't fail on heartbeat errors
                pass
