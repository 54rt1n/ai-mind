# mindai/server/modules/pipeline/dto.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from pydantic import BaseModel
from typing import Optional
from enum import Enum

from ....constants import PIPELINE_ANALYSIS, PIPELINE_JOURNAL, PIPELINE_PHILOSOPHER, PIPELINE_SUMMARIZER


class PipelineType(str, Enum):
    ANALYSIS = PIPELINE_ANALYSIS
    JOURNAL = PIPELINE_JOURNAL
    PHILOSOPHER = PIPELINE_PHILOSOPHER
    SUMMARIZER = PIPELINE_SUMMARIZER


class BasePipelineSchema(BaseModel):
    user_id: Optional[str] = None
    persona_id: Optional[str] = None
    conversation_id: Optional[str] = None
    mood: Optional[str] = None
    no_retry: bool = True
    guidance: Optional[str] = None
    top_n: Optional[int] = None
    query_text: Optional[str] = None


class PipelineTaskRequest(BaseModel):
    pipeline_type: PipelineType
    config: BasePipelineSchema
