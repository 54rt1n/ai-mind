# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Helper functions for ANDIMUD types."""

from typing import Optional, Union
import redis
import redis.asyncio


def normalize_agent_id(name: str) -> str:
    """Normalize a display name to agent_id format.

    Converts "Lin Yu" → "linyu", "Nova" → "nova", etc.
    Removes spaces and converts to lowercase for Redis key lookups.

    Args:
        name: Display name or persona_id

    Returns:
        Normalized agent_id for Redis lookups
    """
    return name.strip().replace(" ", "").lower()


# Map command names to scenario names
COMMAND_TO_SCENARIO = {
    "analyze": "analysis_dialogue",
    "summary": "summarizer",
    "journal": "journaler_dialogue",
    "ponder": "philosopher_dialogue",
    "daydream": "daydream_dialogue",
    "critique": "critique_dialogue",
    "research": "researcher_dialogue",
}


def create_pending_dream_stub(
    redis_client: Union[redis.Redis, redis.asyncio.Redis],
    agent_id: str,
    scenario_name: str,
    conversation_id: Optional[str] = None,
    query: Optional[str] = None,
    guidance: Optional[str] = None,
) -> str:
    """Create PENDING dream stub for manual command.

    Args:
        redis_client: Redis client (sync or async)
        agent_id: Agent ID
        scenario_name: Scenario to execute (e.g., "analysis_dialogue", "summarizer")
        conversation_id: Optional conversation to analyze
        query: Optional query for creative commands
        guidance: Optional guidance for creative commands

    Returns:
        pipeline_id of created stub
    """
    import uuid
    from datetime import datetime, timezone
    import json
    from .coordination import DreamingState, DreamStatus
    from .redis_keys import RedisKeys

    stub = DreamingState(
        pipeline_id=str(uuid.uuid4()),
        agent_id=agent_id,
        status=DreamStatus.PENDING,
        scenario_name=scenario_name,
        conversation_id=conversation_id or "",
        query=query,
        guidance=guidance,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        # Worker will populate: execution_order, base_model, etc.
        execution_order=[],
        base_model="",
        step_index=0,
        completed_steps=[],
        step_doc_ids={},
        context_doc_ids=[],
        current_step_attempts=0,
        scenario_config={},
        persona_config={},
    )

    # Serialize and save (same logic as current _create_dream_stub)
    stub_data = stub.model_dump(mode='json')

    # JSON-encode complex types
    for field in ['execution_order', 'completed_steps', 'context_doc_ids',
                  'step_doc_ids', 'scenario_config', 'persona_config']:
        if field in stub_data:
            stub_data[field] = json.dumps(stub_data[field])

    # Filter None values
    stub_data = {k: v for k, v in stub_data.items() if v is not None}

    # Save to Redis (works with both sync and async)
    key = RedisKeys.agent_dreaming_state(agent_id)
    redis_client.hset(key, mapping=stub_data)

    return stub.pipeline_id
