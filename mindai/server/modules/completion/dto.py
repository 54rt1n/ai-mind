# mindai/server/modules/completion/dto.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from typing import List, Optional
from pydantic import BaseModel, Field

class CompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    prompt: str = Field(..., description="The prompt to generate completions for")
    presence_penalty: Optional[float] = Field(None, description="Penalty for token presence")
    frequency_penalty: Optional[float] = Field(None, description="Penalty for token frequency")
    repetition_penalty: Optional[float] = Field(None, description="Penalty for token repetition")
    temperature: Optional[float] = Field(None, description="Sampling temperature")
    top_p: Optional[float] = Field(None, description="Top p sampling parameter")
    top_k: Optional[int] = Field(None, description="Top k sampling parameter")
    min_p: Optional[float] = Field(None, description="Minimum probability threshold")
    seed: Optional[int] = Field(None, description="Random seed")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stop_token_ids: Optional[List[int]] = Field(None, description="Stop token IDs")
    stream: Optional[bool] = Field(False, description="Stream output")
    include_stop_str_in_output: Optional[bool] = Field(False, description="Include stop strings in output")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict