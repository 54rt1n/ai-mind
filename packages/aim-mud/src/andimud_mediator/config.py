# andimud_mediator/config.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

from dataclasses import dataclass
from aim_mud_types import RedisKeys

@dataclass
class MediatorConfig:
    """Configuration for the Mediator service.

    Attributes:
        redis_url: Redis connection URL.
        event_stream: Stream to read raw events from Evennia.
        event_poll_timeout: Timeout in seconds for event stream polling.
        evennia_api_url: URL for Evennia REST API (for room state queries).
        pause_key: Redis key for mediator pause flag.
    """

    redis_url: str = "redis://localhost:6379"
    event_stream: str = RedisKeys.MUD_EVENTS
    event_poll_timeout: float = 5.0
    evennia_api_url: str = "http://localhost:4001"
    pause_key: str = RedisKeys.MEDIATOR_PAUSE
    # Stream bounds and turn coordination
    mud_events_maxlen: int = 1000
    agent_events_maxlen: int = 200
    turn_request_ttl_seconds: int = 0  # Deprecated: TTL no longer used (turn_request is persistent)
    # Event processing hash
    events_processed_hash_max: int = 10000  # Max entries to keep in hash
    events_processed_cleanup_interval: int = 86400  # Seconds between cleanups (24h)
    # Semi-autonomous analysis mode
    auto_analysis_enabled: bool = True
    system_idle_seconds: int = 15  # System idle threshold for triggering idle turns
    auto_analysis_cooldown_seconds: int = 60  # Prevent rapid re-triggering
