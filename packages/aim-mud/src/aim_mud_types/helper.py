# aim-mud-types/helper.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Helper functions for MUD types."""

import json
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


def _datetime_to_unix(dt: Any) -> Optional[int]:
    """Convert datetime to Unix timestamp (seconds since epoch).

    Handles datetime objects and strings (ISO format or Unix timestamp string).
    """
    if dt is None:
        return None
    if isinstance(dt, str):
        # Empty string means None
        if not dt:
            return None
        # If it's already a Unix timestamp string, convert directly
        try:
            return int(dt)
        except ValueError:
            # Otherwise parse as ISO format
            dt = datetime.fromisoformat(dt)
    return int(dt.timestamp())


def _unix_to_datetime(ts: Any) -> Optional[datetime]:
    """Convert Unix timestamp to datetime.

    Handles int, float, string, or datetime passthrough.
    Returns None for None, empty string, or "0".
    """
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        if not ts or ts == "0":
            return None
        # Try Unix timestamp first, then ISO format for backwards compatibility
        try:
            ts = int(ts)
        except ValueError:
            return datetime.fromisoformat(ts)
    if ts == 0:
        return None
    return datetime.fromtimestamp(int(ts), tz=timezone.utc)


def model_to_redis_hash(model: BaseModel) -> dict[str, str]:
    """Convert a Pydantic model to a Redis hash mapping.

    Uses model_dump(mode="json") to invoke field_serializers, then converts
    all values to strings suitable for Redis HSET.

    - None → ""
    - list/dict → JSON string
    - Other → str()

    Args:
        model: Pydantic model instance with proper field_serializers defined

    Returns:
        Dictionary with string keys and string values for Redis
    """
    dumped = model.model_dump(mode="json")
    data: dict[str, str] = {}
    for k, v in dumped.items():
        if v is None:
            data[k] = ""
        elif isinstance(v, (list, dict)):
            data[k] = json.dumps(v)
        else:
            data[k] = str(v)
    return data
