# aim/server/modules/dreamer/dto.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class StartPipelineRequest(BaseModel):
    """Request to start a new dreamer pipeline."""
    scenario_name: str
    conversation_id: str
    persona_id: str
    user_id: Optional[str] = None
    model_name: str
    query_text: Optional[str] = None
    guidance: Optional[str] = None
    mood: Optional[str] = None


class PipelineStatusResponse(BaseModel):
    """Status information for a dreamer pipeline."""
    pipeline_id: str
    scenario_name: str
    status: str
    current_step: Optional[str]
    completed_steps: List[str]
    failed_steps: List[str]
    step_errors: dict[str, str] = {}
    progress_percent: float
    created_at: datetime
    updated_at: datetime


class StartPipelineResponse(BaseModel):
    """Response after starting a pipeline."""
    status: str
    pipeline_id: str
    message: str


class CancelPipelineResponse(BaseModel):
    """Response after cancelling a pipeline."""
    status: str
    message: str


class ResumePipelineResponse(BaseModel):
    """Response after resuming a pipeline."""
    status: str
    message: str


class DeletePipelineResponse(BaseModel):
    """Response after deleting a pipeline."""
    status: str
    message: str


class ListPipelinesResponse(BaseModel):
    """Response containing list of pipelines."""
    status: str
    count: int
    pipelines: List[PipelineStatusResponse]
