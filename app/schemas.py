from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime

# Ingestion schemas
class IngestDocumentRequest(BaseModel):
    content: str = Field(..., description="Raw document text")
    source_type: str = Field(default="document", description="Type: 'document' or 'conversation'")
    source_id: Optional[str] = Field(None, description="Unique identifier for the source")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class IngestDocumentResponse(BaseModel):
    chunk_ids: List[int]
    num_chunks: int
    message: str

# Query schemas
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query or task description")
    token_budget: int = Field(default=4000, description="Maximum tokens for context")
    session_id: Optional[str] = Field(None, description="Session identifier")

class ContextChunk(BaseModel):
    chunk_id: int
    content: str
    score: float
    source_type: str
    created_at: datetime

class QueryResponse(BaseModel):
    context_chunks: List[ContextChunk]
    total_tokens: int
    retrieval_time_ms: float

# Agent schemas
class AgentRequest(BaseModel):
    query: str
    token_budget: int = Field(default=4000)
    session_id: Optional[str] = None

class AgentResponse(BaseModel):
    response: str
    context_used: List[ContextChunk]
    total_tokens: int

# Feedback schemas
class FeedbackRequest(BaseModel):
    interaction_id: int
    feedback: str = Field(..., description="'helpful' or 'unhelpful'")

# Entity schemas
class EntityResponse(BaseModel):
    id: int
    entity_type: str
    entity_name: str
    properties: Dict[str, Any]
    related_entities: List[Dict[str, Any]] = []
