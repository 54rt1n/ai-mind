# mindai/server/modules/pipeline/dto.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

from ....constants import PIPELINE_ANALYSIS, PIPELINE_JOURNAL, PIPELINE_PHILOSOPHER, PIPELINE_SUMMARIZER, PIPELINE_DAYDREAM


class PipelineType(str, Enum):
    ANALYST = PIPELINE_ANALYSIS
    JOURNALER = PIPELINE_JOURNAL
    PHILOSOPHER = PIPELINE_PHILOSOPHER
    SUMMARIZER = PIPELINE_SUMMARIZER
    DREAMER = PIPELINE_DAYDREAM


class BasePipelineSchema(BaseModel):
    user_id: str = Field(..., description="The user ID")
    persona_id: str = Field(..., description="The persona ID")
    conversation_id: str = Field(..., description="The conversation ID")
    model: str = Field(..., description="The model to use")
    mood: str = Field(..., description="The mood to use")
    no_retry: bool = Field(True, description="Whether to retry the pipeline")
    guidance: Optional[str] = Field(None, description="The guidance to use")
    top_n: Optional[int] = Field(None, description="The top N to use")
    query_text: Optional[str] = Field(None, description="The query text to use")


class PipelineTaskRequest(BaseModel):
    pipeline_type: PipelineType
    config: BasePipelineSchema
