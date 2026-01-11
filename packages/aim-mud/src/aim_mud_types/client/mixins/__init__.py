# aim-mud-types/client/mixins/__init__.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Type-specific mixins for RedisMUDClient.

Each mixin provides domain-specific CRUD operations for a MUD type:
- TurnRequestMixin: MUDTurnRequest operations with CAS
- AgentProfileMixin: AgentProfile operations
- RoomProfileMixin: RoomProfile operations (read-only)
- DreamerStateMixin: DreamerState operations
"""

from .turn_request import TurnRequestMixin
from .agent_profile import AgentProfileMixin
from .room_profile import RoomProfileMixin
from .dreamer_state import DreamerStateMixin

__all__ = [
    "TurnRequestMixin",
    "AgentProfileMixin",
    "RoomProfileMixin",
    "DreamerStateMixin",
]
