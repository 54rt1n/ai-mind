# aim-mud-types/client/__init__.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Centralized Redis client for MUD types with Pydantic validation.

This module provides type-safe Redis clients for all ANDIMUD coordination
and state structures. All Redis hash operations go through Pydantic models
to ensure validation and prevent corruption.

Architecture:
    BaseAsyncRedisMUDClient: Core async serialization and CRUD operations
    BaseSyncRedisMUDClient: Core sync serialization and CRUD operations
    Async mixins: Domain-specific async methods for each MUD type
    Sync mixins: Domain-specific sync methods for each MUD type
    AsyncRedisMUDClient: Composed async client with all functionality
    SyncRedisMUDClient: Composed sync client with all functionality

Usage (async):
    import redis.asyncio as aioredis
    client = AsyncRedisMUDClient(aioredis.Redis(...))
    turn_request = await client.get_turn_request("andi")
    await client.update_turn_request("andi", updated_request, expected_turn_id)

Usage (sync):
    import redis
    client = SyncRedisMUDClient(redis.Redis(...))
    turn_request = client.get_turn_request("andi")
    client.update_turn_request("andi", updated_request, expected_turn_id)

Directory Structure:
    client/
    ├── __init__.py          # Re-exports (this file)
    ├── base.py              # BaseAsyncRedisMUDClient, BaseSyncRedisMUDClient
    ├── async_client.py      # AsyncRedisMUDClient composition
    ├── sync_client.py       # SyncRedisMUDClient composition
    ├── async_mixins/        # Async mixin implementations (14 mixins)
    │   ├── idle.py          # IdleMixin
    │   ├── thought.py       # ThoughtMixin
    │   └── ...              # Other mixins
    └── sync_mixins/         # Sync mixin implementations (14 mixins)
        ├── idle.py          # SyncIdleMixin
        ├── thought.py       # SyncThoughtMixin
        └── ...              # Other mixins
"""

# Import from base module
from .base import BaseAsyncRedisMUDClient, BaseSyncRedisMUDClient, BaseRedisMUDClient

# Import from async_client module
from .async_client import AsyncRedisMUDClient, RedisMUDClient

# Import from sync_client module
from .sync_client import SyncRedisMUDClient

# Backward compatibility: alias async_mixins as mixins for existing code
from . import async_mixins as mixins
from . import sync_mixins


__all__ = [
    # Primary exports (new names)
    "BaseAsyncRedisMUDClient",
    "BaseSyncRedisMUDClient",
    "AsyncRedisMUDClient",
    "SyncRedisMUDClient",
    # Backward compatibility aliases
    "BaseRedisMUDClient",
    "RedisMUDClient",
    "mixins",  # Deprecated: use async_mixins instead
    "sync_mixins",
]
