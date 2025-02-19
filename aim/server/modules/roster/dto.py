# aim/server/modules/roster/dto.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from typing import Dict, List
from pydantic import BaseModel, Field

class PersonaBase(BaseModel):
    """Base schema for persona attributes"""
    persona_id: str = Field(..., description="Unique identifier for the persona")
    persona_version: str = Field(
        default="0.1a",
        description="Persona version string"
    )
    chat_strategy: str = Field(..., description="Chat strategy to use (e.g. xmlmemory)")
    name: str = Field(..., description="Short name")
    full_name: str = Field(..., description="Full name")
    base_thoughts: List[str] = Field(..., description="List of base thoughts")
    pif: Dict[str, str] = Field(..., description="PIF (Persistent Identity Format) data")
    nshot: Dict[str, Dict[str, str]] = Field(..., description="NShot (N-Shot Examples) data")
    system_header: str = Field(
        default="Please follow directions, being precise and methodical, utilizing Chain of Thought, Self-RAG, and Semantic Keywords.",
        description="System prompt header"
    )
    wakeup: List[str] = Field(..., description="List of wakeup messages")
    attributes: Dict[str, str] = Field(..., description="Persona attributes dictionary")
    features: Dict[str, str] = Field(..., description="Persona features dictionary")
    default_location: str = Field(..., description="Default location")
    wardrobe: Dict[str, Dict[str, str]] = Field(..., description="Wardrobe configurations")
    current_outfit: str = Field(..., description="Current outfit key from wardrobe")
    birthday: str | None = Field(None, description="Optional birthday in YYYY-MM-DD format")
    include_date: bool = Field(
        default=True,
        description="Whether to include date in thoughts"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "persona_id": "simple_assistant",
                "chat_strategy": "xmlmemory",
                "name": "assistant",
                "full_name": "assistant",
                "base_thoughts": ["Ready to help"],
                "pif": {
                    "core_traits": "helpful and analytical"
                },
                "system_header": "",
                "wakeup": ["Hello! How can I assist you today?"],
                "attributes": {
                    "age": "24",
                    "occupation": "assistant"
                },
                "features": {
                    "voice": "melodic and clear",
                    "mannerisms": "precise and thoughtful"
                },
                "default_location": "digital workspace",
                "wardrobe": {
                    "casual_attire": {
                        "outfit": "comfortable and stylish",
                        "style": "modern casual"
                    }
                },
                "nshot": {
                    "example1": {
                        "human": "How do you feel about helping me today?",
                        "assistant": "I'm here to help you with any task or need you may have. How can I assist you today?"
                    }
                },
                "current_outfit": "casual_attire",
                "birthday": "2024-10-01"
            }
        }

class CreatePersonaRequest(PersonaBase):
    """Schema for creating a new persona. Inherits all fields from PersonaBase."""
    pass

class UpdatePersonaRequest(BaseModel):
    """Schema for updating an existing persona. All fields are optional for partial updates."""
    chat_strategy: str | None = None
    name: str | None = None
    full_name: str | None = None 
    attributes: Dict[str, str] | None = None
    features: Dict[str, str] | None = None
    wakeup: List[str] | None = None
    base_thoughts: List[str] | None = None
    pif: Dict[str, str] | None = None
    nshot: Dict[str, Dict[str, str]] | None = None
    default_location: str | None = None
    wardrobe: Dict[str, Dict[str, str]] | None = None
    current_outfit: str | None = None
    persona_version: str | None = None
    system_header: str | None = None
    include_date: bool | None = None

class PersonaResponse(PersonaBase):
    """Schema for persona response. Inherits all fields from PersonaBase."""
    pass

class PersonaListResponse(BaseModel):
    """Schema for listing personas"""
    personas: List[PersonaResponse] = Field(..., description="List of all personas")