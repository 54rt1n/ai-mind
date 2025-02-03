# mindai/server/modules/document/dto.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from pydantic import BaseModel, Field
from typing import List

class DocumentInfo(BaseModel):
    """Information about a document in the library"""
    name: str = Field(..., description="Name of the document")
    modified_time: float = Field(..., description="Last modified timestamp")
    size: int = Field(..., description="File size in bytes")


class DocumentListResponse(BaseModel):
    """Response model for listing documents"""
    status: str = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    documents: List[DocumentInfo] = Field(..., description="List of documents")
