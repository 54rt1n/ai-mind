# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Redis key conventions for MUD event streaming.

This module provides consistent key naming across all components:
- Evennia (event publisher, action consumer)
- Mediator (event router, action sequencer)
- AIM Workers (event consumers, action emitters)
"""


class RedisKeys:
    """Redis key name generator for MUD streams.

    All stream and control keys should be generated through this class
    to ensure consistency across Evennia, the mediator, and AIM workers.

    Stream Architecture:
        mud:events          <- Evennia publishes raw events
        agent:{id}:events   <- Mediator pushes enriched events per agent
        mud:actions         <- Agents emit actions for execution

    Control Keys:
        mud:agent:{id}:paused   <- Pause flag for individual agents
        mud:mediator:paused     <- Pause flag for mediator
        mediator:agent_rooms    <- Hash of agent_id -> room_id mappings
    """

    # Stream names
    MUD_EVENTS = "mud:events"
    MUD_ACTIONS = "mud:actions"

    @staticmethod
    def agent_events(agent_id: str) -> str:
        """Get the event stream key for a specific agent.

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            Redis stream key for the agent's events.
        """
        return f"agent:{agent_id}:events"

    @staticmethod
    def agent_profile(agent_id: str) -> str:
        """Get the profile hash key for a specific agent.

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            Redis hash key for the agent profile (identity + activity).
        """
        return f"agent:{agent_id}"

    @staticmethod
    def room_profile(room_id: str) -> str:
        """Get the profile hash key for a room.

        Args:
            room_id: Room dbref/id.

        Returns:
            Redis hash key for the room profile.
        """
        return f"room:{room_id}"

    @staticmethod
    def agent_turn_request(agent_id: str) -> str:
        """Get the turn request key for a specific agent.

        The turn request hash contains:
        - turn_id: Unique identifier for this turn
        - status: Current state (ready, assigned, in_progress, done, fail, crashed, abort_requested, aborted)
        - reason: Why turn was assigned (events, retry, idle, flush, clear, agent, choose)
        - status_reason: Why status changed (e.g., "Worker online", "Turn completed", "LLM call failed: TimeoutError")
        - event_count: Number of events in this turn
        - assigned_at: ISO timestamp when turn was assigned
        - heartbeat_at: ISO timestamp of last heartbeat
        - deadline_ms: Turn timeout in milliseconds
        - message: Optional error message (for fail/crashed states)
        - attempt_count: Number of retry attempts (increments on each failure)
        - next_attempt_at: ISO timestamp when to retry failed turn (empty if max attempts reached)

        Status lifecycle:
        - ready: Worker online and available
        - assigned -> in_progress -> done (successful completion)
        - in_progress -> fail (with next_attempt_at) -> assigned (retry after backoff)
        - in_progress -> crashed (heartbeat stale >5min)
        - assigned/in_progress -> abort_requested -> aborted (user abort)
        - done/aborted -> ready (worker ready for next turn)

        Field disambiguation:
        - reason: Why turn was assigned ("events", "retry", "agent", etc.)
        - status_reason: Why status changed ("Worker online", "Turn completed", "LLM timeout", etc.)
        - message: Error message (for "fail"/"crashed" statuses)

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            Redis key for the agent's turn request hash.
        """
        return f"agent:{agent_id}:turn_request"

    @staticmethod
    def agent_pause(agent_id: str) -> str:
        """Get the pause flag key for a specific agent.

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            Redis key for the agent's pause flag.
        """
        return f"mud:agent:{agent_id}:paused"

    # Mediator keys
    MEDIATOR_PAUSE = "mud:mediator:paused"
    AGENT_ROOMS = "mediator:agent_rooms"
    EVENTS_PROCESSED = "mud:events:processed"
    ACTIONS_PROCESSED = "mud:actions:processed"
    SEQUENCE_COUNTER = "mud:sequence_counter"
    LAST_PLAYER_ACTIVITY = "mud:last_player_activity"

    @staticmethod
    def agent_conversation(agent_id: str) -> str:
        """Get the conversation list key for a specific agent.

        The conversation list stores MUDConversationEntry objects
        as a Redis list, serving as the single source of truth for
        both LLM context and CVM persistence.

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            Redis list key for the agent's conversation history.
        """
        return f"mud:agent:{agent_id}:conversation"

    @staticmethod
    def agent_dreamer(agent_id: str) -> str:
        """Get the dreamer state key for a specific agent.

        The dreamer state hash contains configuration and status
        for the agent's dream processing system.

        Hash fields:
        - enabled: "true" | "false" - auto-dreaming on/off
        - last_dream_at: ISO timestamp of last dream completion
        - last_dream_scenario: name of last scenario run
        - idle_threshold_seconds: seconds idle before auto-dream triggers (default: 3600)
        - token_threshold: accumulated tokens before auto-dream triggers (default: 10000)
        - pending_pipeline_id: if dream is in progress, the pipeline ID

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            Redis hash key for the agent's dreamer state.
        """
        return f"agent:{agent_id}:dreamer"

    @staticmethod
    def agent_conversation_report(agent_id: str) -> str:
        """Get the conversation report key for a specific agent.

        The report contains conversation statistics generated
        from the persona's ConversationModel, updated on worker
        startup and after each CVM write/dreamer job.

        Value format (JSON stored as string):
        {
          "conv1": {"conversation": 10, "analysis": 1, "timestamp_max": 1704412800},
          "conv2": {"conversation": 5, "timestamp_max": 1704412900}
        }

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            Redis key for the agent's conversation report.
        """
        return f"agent:{agent_id}:conversation_report"

    @staticmethod
    def format_stream_id(timestamp_ms: int, sequence: int = 0) -> str:
        """Format a Redis stream message ID.

        Args:
            timestamp_ms: Unix timestamp in milliseconds.
            sequence: Sequence number within the millisecond.

        Returns:
            Formatted stream ID (e.g., "1704096000000-0").
        """
        return f"{timestamp_ms}-{sequence}"

    @staticmethod
    def agent_plan(agent_id: str) -> str:
        """Get the plan hash key for a specific agent.

        Hash fields:
        - plan_id: UUID of current plan
        - agent_id: Agent identifier
        - objective: High-level goal
        - summary: One-line summary
        - status: PlanStatus enum value
        - tasks: JSON array of PlanTask objects
        - current_task_id: Index into tasks
        - created_at: ISO timestamp
        - updated_at: ISO timestamp

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            Redis hash key for the agent's plan.
        """
        return f"agent:{agent_id}:plan"

    @staticmethod
    def agent_planner_enabled(agent_id: str) -> str:
        """Get the planner enabled flag for a specific agent.

        Value: "true" | "false"

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            Redis key for the agent's planner enabled flag.
        """
        return f"agent:{agent_id}:planner:enabled"

    @staticmethod
    def agent_dreaming_state(agent_id: str) -> str:
        """Get the dreaming state key for a specific agent.

        The dreaming state hash contains the serialized DreamingState
        for step-by-step dream execution.

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            Redis hash key for the agent's active dream state.
        """
        return f"agent:{agent_id}:dreaming"

    @staticmethod
    def agent_dreaming_history(agent_id: str) -> str:
        """Get the dreaming history key for a specific agent.

        The history list stores completed DreamingState objects
        for archival and debugging.

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            Redis list key for the agent's completed dream history.
        """
        return f"agent:{agent_id}:dreaming:history"

    @staticmethod
    def agent_thought(agent_id: str) -> str:
        """Get the thought injection key for a specific agent.

        The thought key stores external thought content to inject into
        the agent's processing. Value is JSON:
        {
            "content": str,          # The thought text
            "source": str,           # "manual" | "dreamer" | "system"
            "timestamp": int,        # Unix timestamp when set
        }

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            Redis key for the agent's injected thought.
        """
        return f"agent:{agent_id}:thought"
