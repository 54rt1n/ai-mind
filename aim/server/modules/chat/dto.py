# aim/server/modules/chat/dto.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from ....tool.dto import Tool


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message")
    content: str = Field(..., description="Content of the message")
    timestamp: Optional[int] = Field(None, description="Timestamp of the message")


class Metadata(BaseModel):
    user_id: str = Field(..., description="ID of the user")
    persona_id: str = Field(..., description="ID of the persona to use")
    pinned_messages: Optional[list[str]] = Field(None, description="Pinned messages")
    active_document: Optional[str] = Field(None, description="Active document")
    workspace_content: Optional[str] = Field(None, description="Workspace content")
    thought_content: Optional[str] = Field(None, description="Thought content")
    disable_guidance: Optional[bool] = Field(False, description="Disable guidance")
    disable_pif: Optional[bool] = Field(False, description="Disable pif")
    location: Optional[str] = Field(None, description="Location")


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of messages")
    metadata: Optional[Metadata] = Field(None, description="Metadata")
    model: Optional[str] = Field(..., description="Model to use")
    temperature: Optional[float] = Field(0.7, description="Temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(True, description="Stream output")
    system_message: Optional[str] = Field(None, description="System message")
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
    tools: Optional[List[Tool]] = Field(None, description="List of tools available for the model")
    tool_choice: Optional[str] = Field(None, description="Specific tool to use, if any")
    response_format: Optional[Dict[str, str]] = Field(None, description="Desired format of the response")


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict

