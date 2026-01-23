# aim-mud-types/client/async_client.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Composed async Redis MUD client with all type-specific operations.

This module provides the complete async Redis client that combines
the base client with all domain-specific mixins.
"""

from .base import BaseAsyncRedisMUDClient

# Import all async mixins
from .async_mixins.turn_request import TurnRequestMixin
from .async_mixins.mud_events import MudEventsStreamMixin
from .async_mixins.agent_events import AgentEventsStreamMixin
from .async_mixins.mud_actions import MudActionsStreamMixin
from .async_mixins.conversation import ConversationMixin
from .async_mixins.conversation_report import ConversationReportMixin
from .async_mixins.agent_profile import AgentProfileMixin
from .async_mixins.room_profile import RoomProfileMixin
from .async_mixins.dreamer_state import DreamerStateMixin
from .async_mixins.pause import PauseMixin
from .async_mixins.plan import PlanMixin
from .async_mixins.sequence import SequenceMixin
from .async_mixins.idle import IdleMixin
from .async_mixins.thought import ThoughtMixin


class AsyncRedisMUDClient(
    BaseAsyncRedisMUDClient,
    TurnRequestMixin,
    MudEventsStreamMixin,
    AgentEventsStreamMixin,
    MudActionsStreamMixin,
    ConversationMixin,
    ConversationReportMixin,
    AgentProfileMixin,
    RoomProfileMixin,
    DreamerStateMixin,
    PauseMixin,
    PlanMixin,
    SequenceMixin,
    IdleMixin,
    ThoughtMixin,
):
    """Complete async Redis MUD client with all type-specific operations.

    Combines base serialization/deserialization with domain-specific
    methods for turn requests, profiles, dreamer state, plans, and more.

    Usage:
        client = AsyncRedisMUDClient(redis_client)

        # Turn requests
        turn_req = await client.get_turn_request("andi")
        await client.create_turn_request("andi", new_request)
        await client.update_turn_request("andi", updated, expected_turn_id)
        await client.heartbeat_turn_request("andi")

        # Events/actions streams
        await client.append_mud_event({"data": "..."})
        await client.read_mud_events("0", block_ms=1000)

        # Conversation list
        await client.append_conversation_entry("andi", "entry_json")
        entries = await client.get_conversation_entries("andi", 0, -1)

        # Agent profiles
        profile = await client.get_agent_profile("andi")
        await client.create_agent_profile(profile)
        await client.update_agent_profile_fields("andi", conversation_id="conv2")

        # Room profiles
        room = await client.get_room_profile("room123")

        # Dreamer state
        state = await client.get_dreamer_state("andi")
        await client.update_dreamer_state_fields("andi", enabled=True)

        # Plans
        plan = await client.get_plan("andi")
        await client.create_plan(new_plan)
        await client.update_plan_fields("andi", status=PlanStatus.ACTIVE)
        await client.delete_plan("andi")
        await client.set_planner_enabled("andi", True)

        # Idle flag
        is_active = await client.is_idle_active("andi")
        await client.set_idle_active("andi", True)

        # Thought injection
        has_thought = await client.has_active_thought("andi")
        thought = await client.get_thought("andi")
    """
    pass


# Backward compatibility alias
RedisMUDClient = AsyncRedisMUDClient

__all__ = [
    "AsyncRedisMUDClient",
    "RedisMUDClient",  # Backward compatibility
]
