# aim-mud-types/client/sync_client.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Composed sync Redis MUD client with all type-specific operations.

This module provides the complete sync Redis client that combines
the base client with all domain-specific mixins. Use this client
in synchronous contexts like Evennia commands.
"""

from .base import BaseSyncRedisMUDClient

# Import all sync mixins
from .sync_mixins.turn_request import SyncTurnRequestMixin
from .sync_mixins.mud_events import SyncMudEventsStreamMixin
from .sync_mixins.agent_events import SyncAgentEventsStreamMixin
from .sync_mixins.mud_actions import SyncMudActionsStreamMixin
from .sync_mixins.conversation import SyncConversationMixin
from .sync_mixins.conversation_report import SyncConversationReportMixin
from .sync_mixins.agent_profile import SyncAgentProfileMixin
from .sync_mixins.room_profile import SyncRoomProfileMixin
from .sync_mixins.dreamer_state import SyncDreamerStateMixin
from .sync_mixins.pause import SyncPauseMixin
from .sync_mixins.plan import SyncPlanMixin
from .sync_mixins.sequence import SyncSequenceMixin
from .sync_mixins.idle import SyncIdleMixin
from .sync_mixins.thought import SyncThoughtMixin


class SyncRedisMUDClient(
    BaseSyncRedisMUDClient,
    SyncTurnRequestMixin,
    SyncMudEventsStreamMixin,
    SyncAgentEventsStreamMixin,
    SyncMudActionsStreamMixin,
    SyncConversationMixin,
    SyncConversationReportMixin,
    SyncAgentProfileMixin,
    SyncRoomProfileMixin,
    SyncDreamerStateMixin,
    SyncPauseMixin,
    SyncPlanMixin,
    SyncSequenceMixin,
    SyncIdleMixin,
    SyncThoughtMixin,
):
    """Complete sync Redis MUD client with all type-specific operations.

    Combines base serialization/deserialization with domain-specific
    methods for turn requests, profiles, dreamer state, plans, and more.

    This is the synchronous counterpart to AsyncRedisMUDClient.
    Use this client in synchronous contexts like Evennia commands.

    Usage:
        import redis
        client = SyncRedisMUDClient(redis.Redis(...))

        # Turn requests
        turn_req = client.get_turn_request("andi")
        client.create_turn_request("andi", new_request)
        client.update_turn_request("andi", updated, expected_turn_id)
        client.heartbeat_turn_request("andi")

        # Events/actions streams
        client.append_mud_event({"data": "..."})
        client.read_mud_events("0", block_ms=1000)

        # Conversation list
        client.append_conversation_entry("andi", "entry_json")
        entries = client.get_conversation_entries("andi", 0, -1)

        # Agent profiles
        profile = client.get_agent_profile("andi")
        client.create_agent_profile(profile)
        client.update_agent_profile_fields("andi", conversation_id="conv2")

        # Room profiles
        room = client.get_room_profile("room123")

        # Dreamer state
        state = client.get_dreamer_state("andi")
        client.update_dreamer_state_fields("andi", enabled=True)

        # Plans
        plan = client.get_plan("andi")
        client.create_plan(new_plan)
        client.update_plan_fields("andi", status=PlanStatus.ACTIVE)
        client.delete_plan("andi")
        client.set_planner_enabled("andi", True)

        # Idle flag
        is_active = client.is_idle_active("andi")
        client.set_idle_active("andi", True)

        # Thought injection
        has_thought = client.has_active_thought("andi")
        thought = client.get_thought("andi")
    """
    pass


__all__ = [
    "SyncRedisMUDClient",
]
