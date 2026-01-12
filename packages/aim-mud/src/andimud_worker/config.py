# aim/app/mud/config.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Configuration for MUD agent workers."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MUDConfig:
    """Configuration for a MUD agent worker.

    Controls agent identity, Redis connections, timing parameters,
    and memory settings. LLM configuration (model, temperature, max_tokens)
    comes from ChatConfig which loads from .env.

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
    event_settle_seconds: float = 15.0  # Wait time for event cascade settling

    # Memory
    memory_path: Optional[str] = None
    top_n_memories: int = 10
    max_recent_turns: int = 20
    bucket_max_tokens: int = 28000
    bucket_idle_flush_seconds: int = 600
    conversation_max_tokens: int = 50000

    # Phase 1 decision tools
    decision_tool_file: str = "config/tools/mud_phase1.yaml"
    decision_max_retries: int = 3
    agent_tool_file: str = "config/tools/mud_agent.yaml"

    # Turn request coordination
    turn_request_ttl_seconds: int = 0  # Deprecated: TTL no longer used (turn_request is persistent)
    turn_request_heartbeat_seconds: int = 20
    turn_request_poll_interval: float = 0.5

    # LLM failure retry configuration
    llm_failure_backoff_base_seconds: int = 30
    llm_failure_backoff_max_seconds: int = 600  # 10 minutes
    llm_failure_max_attempts: int = 3

    # Pause control (Redis key for pause flag)
    pause_key: str = field(init=False)

    def __post_init__(self) -> None:
        """Set computed defaults after initialization."""
        if self.agent_stream is None:
            self.agent_stream = f"agent:{self.agent_id}:events"
        if self.memory_path is None:
            self.memory_path = f"memory/{self.persona_id}"
        self.pause_key = f"mud:agent:{self.agent_id}:paused"
