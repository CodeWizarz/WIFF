import hashlib
import re
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.models import DocumentChunk
from app.services.embeddings import EmbeddingService
from app.config import settings

class DeDuplicationService:
    """
    Handles exact (hash-based) and semantic (embedding-based) deduplication.
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
    
    def compute_hash(self, text: str) -> str:
        """
        Compute SHA-256 hash of normalized text.
        """
        # Normalize: lower case, collapse whitespace
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    async def is_duplicate(self, db: AsyncSession, content_hash: str) -> bool:
        """
        Check if exact duplicate exists in DB.
        """
        result = await db.execute(
            select(DocumentChunk).where(DocumentChunk.content_hash == content_hash).limit(1)
        )
        return result.scalars().first() is not None
    
    async def is_near_duplicate(
        self, 
        db: AsyncSession, 
        embedding: List[float], 
        threshold: float = 0.95
    ) -> bool:
        """
        Check for semantic duplicates using vector similarity.
        """
        # Use pgvector cosine distance (cmp < 1 - threshold)
        # Cosine Distance = 1 - Cosine Similarity
        # Threshold 0.95 Sim => Distance < 0.05
        
        distance_threshold = 1.0 - threshold
        
        # Check against recent chunks to safe guard against duplicated document
        # being processed in chunks
        stmt = select(DocumentChunk.id).order_by(
            DocumentChunk.embedding.cosine_distance(embedding)
        ).limit(1)
        
        result = await db.execute(stmt)
        # Note: We'd need the actual distance to verify threshold. 
        # pgvector query usually returns the row, we need to add the distance calc or trust logic
        # For strictness, let's select with distance
        
        stmt = select(
            DocumentChunk.embedding.cosine_distance(embedding).label("distance")
        ).order_by(
            "distance"
        ).limit(1)
        
        result = await db.execute(stmt)
        row = result.first()
        
        if row and row.distance < distance_threshold:
            return True
            
        return False
