from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from datetime import datetime
from app.database import Base
from app.config import settings

class DocumentChunk(Base):
    """Vector store for document chunks"""
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(settings.embedding_dim), nullable=False)
    
    # Metadata
    source_type = Column(String(50), nullable=False)  # 'document', 'conversation'
    source_id = Column(String(255), index=True)
    metadata = Column(JSONB, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Indexes for fast retrieval
    __table_args__ = (
        Index('idx_embedding_vector', 'embedding', postgresql_using='ivfflat'),
    )


class Entity(Base):
    """Graph nodes - entities extracted from documents"""
    __tablename__ = "entities"
    
    id = Column(Integer, primary_key=True, index=True)
    entity_type = Column(String(50), nullable=False, index=True)  # 'Person', 'Concept', etc.
    entity_name = Column(String(255), nullable=False, index=True)
    
    # For entity merging/deduplication
    embedding = Column(Vector(settings.embedding_dim))
    
    properties = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)


class Relationship(Base):
    """Graph edges - relationships between entities"""
    __tablename__ = "relationships"
    
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("entities.id"), nullable=False, index=True)
    target_id = Column(Integer, ForeignKey("entities.id"), nullable=False, index=True)
    
    relationship_type = Column(String(100), nullable=False)  # 'mentions', 'related_to', etc.
    strength = Column(Float, default=1.0)  # Co-occurrence weight
    
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Compound index for fast graph traversal
    __table_args__ = (
        Index('idx_source_target', 'source_id', 'target_id'),
    )


class InteractionLog(Base):
    """Log agent interactions for learning"""
    __tablename__ = "interaction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), index=True)
    
    query = Column(Text, nullable=False)
    retrieved_chunk_ids = Column(JSONB)  # List of chunk IDs used
    response = Column(Text)
    
    # Feedback signals
    feedback = Column(String(20))  # 'helpful', 'unhelpful', None
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
