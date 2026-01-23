# aim-mud-types/client/sync_mixins/__init__.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Synchronous type-specific mixins for SyncRedisMUDClient.

Each mixin provides domain-specific sync CRUD operations for a MUD type:
- SyncTurnRequestMixin: MUDTurnRequest operations with CAS
- SyncMudEventsStreamMixin: mud:events stream helpers
- SyncAgentEventsStreamMixin: agent:{id}:events stream helpers
- SyncMudActionsStreamMixin: mud:actions stream helpers
- SyncConversationMixin: agent conversation list helpers
- SyncConversationReportMixin: cached conversation report helpers
- SyncAgentProfileMixin: AgentProfile operations
- SyncRoomProfileMixin: RoomProfile operations (read-only)
- SyncDreamerStateMixin: DreamerState operations
- SyncPauseMixin: pause flag helpers
- SyncPlanMixin: AgentPlan operations
- SyncSequenceMixin: sequence counter helpers
- SyncIdleMixin: idle active flag helpers
- SyncThoughtMixin: thought injection helpers (read-only)
"""

from .turn_request import SyncTurnRequestMixin
from .mud_events import SyncMudEventsStreamMixin
from .agent_events import SyncAgentEventsStreamMixin
from .mud_actions import SyncMudActionsStreamMixin
from .conversation import SyncConversationMixin
from .conversation_report import SyncConversationReportMixin
from .agent_profile import SyncAgentProfileMixin
from .room_profile import SyncRoomProfileMixin
from .dreamer_state import SyncDreamerStateMixin
from .pause import SyncPauseMixin
from .plan import SyncPlanMixin
from .sequence import SyncSequenceMixin
from .idle import SyncIdleMixin
from .thought import SyncThoughtMixin

__all__ = [
    "SyncTurnRequestMixin",
    "SyncMudEventsStreamMixin",
    "SyncAgentEventsStreamMixin",
    "SyncMudActionsStreamMixin",
    "SyncConversationMixin",
    "SyncConversationReportMixin",
    "SyncAgentProfileMixin",
    "SyncRoomProfileMixin",
    "SyncDreamerStateMixin",
    "SyncPauseMixin",
    "SyncPlanMixin",
    "SyncSequenceMixin",
    "SyncIdleMixin",
    "SyncThoughtMixin",
]
