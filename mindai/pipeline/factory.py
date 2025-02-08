# mindai/pipeline/factory.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from typing import Callable, Awaitable, Any

from ..constants import (
    PIPELINE_ANALYSIS, PIPELINE_JOURNAL, PIPELINE_DAYDREAM, PIPELINE_PHILOSOPHER, PIPELINE_SUMMARIZER
)

from .base import BasePipeline
from .analyst import analysis_pipeline
from .daydream import daydream_pipeline
from .journaler import journal_pipeline
from .philosopher import ponder_pipeline
from .summarizer import summary_pipeline

def pipeline_factory(pipeline_type: str) -> Callable[[Any, Any], Awaitable[BasePipeline]]:
    if pipeline_type == PIPELINE_ANALYSIS:
        return analysis_pipeline
    elif pipeline_type == PIPELINE_DAYDREAM:
        return daydream_pipeline
    elif pipeline_type == PIPELINE_JOURNAL:
        return journal_pipeline
    elif pipeline_type == PIPELINE_PHILOSOPHER:
        return ponder_pipeline
    elif pipeline_type == PIPELINE_SUMMARIZER:
        return summary_pipeline
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")
