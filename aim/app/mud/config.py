# aim/app/mud/config.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Configuration for MUD agent workers."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MUDConfig:
    """Configuration for a MUD agent worker.

    Controls agent identity, Redis connections, timing parameters,
    memory settings, and LLM configuration.

    Attributes:
        agent_id: Unique identifier for this agent in the MUD.
        persona_id: ID of the persona configuration to use.
        redis_url: Redis connection URL.
        agent_stream: Redis stream for receiving events.
            Defaults to "agent:{agent_id}:events".
        action_stream: Redis stream for emitting actions.
        spontaneous_check_interval: Timeout in seconds for spontaneous
            action check when no events arrive.
        spontaneous_action_interval: Seconds of silence before
            spontaneous action is triggered.
        memory_path: Base path for memory storage.
            Defaults to "memory/{persona_id}".
        top_n_memories: Maximum memories to retrieve per query.
        max_recent_turns: Maximum recent turns to keep in session history.
        llm_provider: LLM provider name (e.g., "anthropic", "openai").
        model: Model identifier for LLM inference.
        temperature: Sampling temperature for LLM.
        max_tokens: Maximum tokens for LLM response.
    """

    # Identity
    agent_id: str
    persona_id: str

    # Redis
    redis_url: str = "redis://localhost:6379"
    agent_stream: Optional[str] = None
    action_stream: str = "mud:actions"

    # Timing
    spontaneous_check_interval: float = 60.0
    spontaneous_action_interval: float = 300.0

    # Memory
    memory_path: Optional[str] = None
    top_n_memories: int = 10
    max_recent_turns: int = 20

    # LLM
    llm_provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 2048

    # Pause control (Redis key for pause flag)
    pause_key: str = field(init=False)

    def __post_init__(self) -> None:
        """Set computed defaults after initialization."""
        if self.agent_stream is None:
            self.agent_stream = f"agent:{self.agent_id}:events"
        if self.memory_path is None:
            self.memory_path = f"memory/{self.persona_id}"
        self.pause_key = f"mud:agent:{self.agent_id}:paused"
