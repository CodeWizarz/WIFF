from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, or_
from app.models import Entity, Relationship
from app.services.embeddings import EmbeddingService
import asyncio

class MaintenanceService:
    """
    Background tasks for memory quality improvement:
    1. Entity Consolidation (Merge duplicates)
    2. Edge Weight Decay (Forget old connections)
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
    
    async def run_consolidation(self, db: AsyncSession, threshold: float = 0.92):
        """
        Merge entities that are semantically very similar.
        O(N^2) in naive impl - improved with vector search.
        """
        # Fetch all entities (for MVP scale)
        result = await db.execute(select(Entity))
        entities = result.scalars().all()
        
        merged_count = 0
        
        # Simple greedy clustering
        # In prod: Use DB-side clustering or specialized vector index queries
        skip_ids = set()
        
        for i, e1 in enumerate(entities):
            if e1.id in skip_ids:
                continue
                
            for e2 in entities[i+1:]:
                if e2.id in skip_ids:
                    continue
                
                # Check 1: Name Similarity (Exact or fuzzy)
                name_sim = self._string_similarity(e1.entity_name, e2.entity_name)
                
                # Check 2: Embedding Similarity
                vec_sim = self.embedding_service.cosine_similarity(e1.embedding, e2.embedding)
                
                if (name_sim > 0.8 and vec_sim > 0.85) or (vec_sim > threshold):
                    # Merge e2 into e1
                    await self._merge_entities(db, primary=e1, secondary=e2)
                    skip_ids.add(e2.id)
                    merged_count += 1
        
        await db.commit()
        return merged_count

    async def _merge_entities(self, db: AsyncSession, primary: Entity, secondary: Entity):
        """
        Move all edges from Secondary to Primary, then delete Secondary.
        """
        # 1. Update Relationships where secondary is source
        await db.execute(
            update(Relationship)
            .where(Relationship.source_id == secondary.id)
            .values(source_id=primary.id)
        )
        
        # 2. Update Relationships where secondary is target
        await db.execute(
            update(Relationship)
            .where(Relationship.target_id == secondary.id)
            .values(target_id=primary.id)
        )
        
        # 3. Merge Properties
        # (Simple merge for MVP)
        if hasattr(secondary, 'properties') and secondary.properties:
            if not primary.properties:
                primary.properties = {}
            primary.properties.update(secondary.properties)
            
        # 4. Delete Secondary
        await db.delete(secondary)
        
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Simple Jaccard similarity on character n-grams or words"""
        set1 = set(s1.lower().split())
        set2 = set(s2.lower().split())
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union
