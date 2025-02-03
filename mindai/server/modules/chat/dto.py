# mindai/server/modules/chat/dto.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from pydantic import BaseModel, Field
from typing import List, Optional


class ChatMessage(BaseModel):
    timestamp: int
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    user_id: str = Field(..., description="ID of the user")
    persona_id: str = Field(..., description="ID of the persona to use")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    model: Optional[str] = Field(..., description="Model to use")
    temperature: Optional[float] = Field(0.7, description="Temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(True, description="Stream output")
    location: Optional[str] = Field(None, description="Location")
    system_message: Optional[str] = Field(None, description="System message")
    pinned_messages: Optional[list[str]] = Field(None, description="Pinned messages")
    active_document: Optional[str] = Field(None, description="Active document")
    workspace_content: Optional[str] = Field(None, description="Workspace content")
    thought_content: Optional[str] = Field(None, description="Thought content")
    disable_guidance: Optional[bool] = Field(False, description="Disable guidance")
    disable_pif: Optional[bool] = Field(False, description="Disable pif")
    presence_penalty: Optional[float] = Field(None, description="Penalty for token presence")
    frequency_penalty: Optional[float] = Field(None, description="Penalty for token frequency")
    repetition_penalty: Optional[float] = Field(None, description="Penalty for token repetition")
    top_p: Optional[float] = Field(None, description="Top p sampling parameter")
    top_k: Optional[int] = Field(None, description="Top k sampling parameter")
    min_p: Optional[float] = Field(None, description="Minimum probability threshold")
    seed: Optional[int] = Field(None, description="Random seed")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stop_token_ids: Optional[List[int]] = Field(None, description="Stop token IDs")
    include_stop_str_in_output: Optional[bool] = Field(False, description="Include stop strings in output")
    min_tokens: Optional[int] = Field(None, description="Minimum tokens to generate")


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict

