# aim-mud-types/client/mixins/__init__.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Type-specific mixins for RedisMUDClient.

Each mixin provides domain-specific CRUD operations for a MUD type:
- TurnRequestMixin: MUDTurnRequest operations with CAS
- MudEventsStreamMixin: mud:events stream helpers
- AgentEventsStreamMixin: agent:{id}:events stream helpers
- MudActionsStreamMixin: mud:actions stream helpers
- ConversationMixin: agent conversation list helpers
- ConversationReportMixin: cached conversation report helpers
- AgentProfileMixin: AgentProfile operations
- RoomProfileMixin: RoomProfile operations (read-only)
- DreamerStateMixin: DreamerState operations
- PauseMixin: pause flag helpers
- PlanMixin: AgentPlan operations
- SequenceMixin: sequence counter helpers
"""

from .turn_request import TurnRequestMixin
from .mud_events import MudEventsStreamMixin
from .agent_events import AgentEventsStreamMixin
from .mud_actions import MudActionsStreamMixin
from .conversation import ConversationMixin
from .conversation_report import ConversationReportMixin
from .agent_profile import AgentProfileMixin
from .room_profile import RoomProfileMixin
from .dreamer_state import DreamerStateMixin
from .pause import PauseMixin
from .plan import PlanMixin
from .sequence import SequenceMixin

__all__ = [
    "TurnRequestMixin",
    "MudEventsStreamMixin",
    "AgentEventsStreamMixin",
    "MudActionsStreamMixin",
    "ConversationMixin",
    "ConversationReportMixin",
    "AgentProfileMixin",
    "RoomProfileMixin",
    "DreamerStateMixin",
    "PauseMixin",
    "PlanMixin",
    "SequenceMixin",
]
