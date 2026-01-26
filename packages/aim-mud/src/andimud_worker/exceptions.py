# aim/app/mud/worker/exceptions.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Exceptions for the MUD worker."""

class AbortRequestedException(Exception):
    """Raised when a turn is aborted by user request."""
    pass


class TurnPreemptedException(Exception):
    """Raised when a turn is preempted by a newer turn_id."""
    pass


class ContextOverflowError(Exception):
    """Raised when context exceeds model limit after max retries.

    This occurs when the focus tool selects too much code and the
    immediate requery mechanism cannot recover within the retry limit.
    """
    pass
