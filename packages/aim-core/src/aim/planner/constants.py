# aim/planner/constants.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Planner constants."""

# Stage 2 limits
FORM_BUILDER_MAX_ITERATIONS = 50
FORM_BUILDER_MAX_TASKS = 20

# Timeouts (seconds)
DELIBERATION_TIMEOUT = 600.0  # 10 minutes
FORM_BUILDER_LLM_TIMEOUT = 60.0  # 1 minute per call

# Retry limits
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 2.0  # seconds, doubles each retry

# Redis TTLs (seconds) - for future use
PLAN_TTL_COMPLETED = 86400 * 30  # 30 days
PLAN_TTL_ABANDONED = 86400 * 7  # 7 days
