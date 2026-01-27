from typing import List, Dict, Any, Tuple, Set
from datetime import datetime, timedelta
import math
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text, func, desc, or_
from app.models import DocumentChunk, Entity, Relationship
from app.services.embeddings import EmbeddingService
from app.services.chunker import DocumentChunker
from app.services.entity_extractor import EntityExtractor
from app.config import settings
from app.schemas import ContextChunk

class RetrievalService:
    """
    Hybrid retrieval engine combining vector search + graph traversal.
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.chunker = DocumentChunker()
        self.entity_extractor = EntityExtractor()
        
        # Tuning parameters
        self.alpha = settings.alpha_semantic
        self.beta = settings.beta_recency
        self.gamma = settings.gamma_graph
    
    async def retrieve_context(
        self,
        db: AsyncSession,
        query: str,
        token_budget: int = 4000,
        session_id: str = None
    ) -> Tuple[List[ContextChunk], int]:
        """
        Main entry point: Retrieve and pack context for a query.
        """
        # 1. Generate query embedding
        query_embedding = await self.embedding_service.embed_text(query)
        
        # 2. Extract entities from query (for graph search)
        query_entities = await self.entity_extractor.extract_entities(query)
        query_entity_names = [e["name"] for e in query_entities]
        
        # 3. Parallel Search: Vector + Graph
        # A. Vector Search (Semantic)
        vector_results = await self._vector_search(db, query_embedding, limit=settings.top_k * 2)
        
        # B. Graph Search (Relational)
        related_entity_names = await self._graph_search(db, query_entity_names)
        
        # 4. Hybrid Scoring
        scored_chunks = self._rank_chunks(
            vector_results, 
            related_entity_names,
            query_entity_names
        )
        
        # 5. Token Packing
        final_context, total_tokens = self._pack_context(scored_chunks, token_budget)
        
        return final_context, total_tokens

    async def _vector_search(
        self, 
        db: AsyncSession, 
        query_embedding: List[float],
        limit: int
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Perform vector similarity search using pgvector.
        Returns: List of (DocumentChunk, score 0..1)
        """
        # Distance metric: Cosine Distance (<=>)
        # We want Similarity = 1 - Distance
        distance_col = DocumentChunk.embedding.cosine_distance(query_embedding).label("distance")
        
        stmt = select(DocumentChunk, distance_col).order_by(
            distance_col
        ).limit(limit)
        
        result = await db.execute(stmt)
        # Returns list of (DocumentChunk, distance)
        rows = result.all()
        
        # Convert distance to similarity score
        return [(row[0], 1.0 - row[1]) for row in rows]

    async def _graph_search(
        self,
        db: AsyncSession,
        query_entity_names: List[str]
    ) -> Set[str]:
        """
        Find entities related to the query entities (1-hop).
        Returns: Set of related entity names.
        """
        if not query_entity_names:
            return set()
            
        # Find IDs of query entities
        stmt = select(Entity).where(Entity.entity_name.in_(query_entity_names))
        result = await db.execute(stmt)
        start_entities = result.scalars().all()
        start_ids = [e.id for e in start_entities]
        
        if not start_ids:
            return set()
            
        # Find 1-hop neighbors via Relationships
        stmt_related = select(Entity.entity_name).join(
            Relationship, Relationship.target_id == Entity.id
        ).where(
            Relationship.source_id.in_(start_ids)
        )
        
        result_related = await db.execute(stmt_related)
        return set(result_related.scalars().all())

    def _rank_chunks(
        self,
        vector_results: List[Tuple[DocumentChunk, float]],
        related_entity_names: Set[str],
        direct_entity_names: List[str]
    ) -> List[ContextChunk]:
        """
        Apply scoring formula:
        Score = α*Semantic + β*Recency + γ*Graph
        """
        scored = []
        now = datetime.utcnow()
        relevant_entities = related_entity_names.union(set(direct_entity_names))
        
        # Map DB objects to scored objects
        for chunk, semantic_sim in vector_results:
            # 1. Semantic Score (from vector search)
            semantic_score = max(0.0, semantic_sim)
            
            # 2. Recency Score (Exponential decay)
            # 1.0 for now, 0.5 after 30 days
            age_days = max(0, (now - chunk.created_at).days)
            recency_score = math.exp(-age_days / 30.0)
            
            # 3. Graph Score
            # Boost if chunk mentions relevant entities
            graph_score = 0.0
            found_entities = 0
            chunk_lower = chunk.content.lower()
            
            for name in relevant_entities:
                if name.lower() in chunk_lower:
                    found_entities += 1
            
            if found_entities > 0:
                # Diminishing returns for multiple matches
                graph_score = min(1.0, 0.2 * found_entities)
            
            # Total Score
            final_score = (
                self.alpha * semantic_score + 
                self.beta * recency_score + 
                self.gamma * graph_score
            )
            
            scored.append(ContextChunk(
                chunk_id=chunk.id,
                content=chunk.content,
                score=final_score,
                source_type=chunk.source_type,
                created_at=chunk.created_at
            ))
            
        # Sort by final score descending
        ranked = sorted(scored, key=lambda x: x.score, reverse=True)
        
        # Conflict Resolution / Diversity Filtering
        return self._resolve_conflicts(ranked)
        
    def _resolve_conflicts(self, ranked_chunks: List[ContextChunk]) -> List[ContextChunk]:
        """
        Filter out near-duplicates. Keep the highest scored one.
        Assumes chunks are ALREADY sorted by score descending.
        """
        filtered = []
        seen_content = [] # Store contents to check similarity
        
        for chunk in ranked_chunks:
            is_dup = False
            for seen in seen_content:
                # Simple containment or high Jaccard for text overlap
                # For strict performance, use Hash if available (Phase 7 added content_hash)
                # But here we only have ContextChunk... we assume distinct IDs.
                
                # Check 1: Exact subset?
                if chunk.content in seen or seen in chunk.content:
                    # If heavily overlapping, assumption: higher score (earlier in list) is better.
                    # Skip this one.
                    is_dup = True
                    break
            
            if not is_dup:
                filtered.append(chunk)
                seen_content.append(chunk.content)
                
        return filtered

    def _pack_context(
        self,
        ranked_chunks: List[ContextChunk],
        token_budget: int
    ) -> Tuple[List[ContextChunk], int]:
        """
        Greedily pack chunks until token budget is exhausted.
        """
        final_context = []
        current_tokens = 0
        headroom = int(token_budget * 0.1) # Leave 10% space
        limit = token_budget - headroom
        
        for chunk in ranked_chunks:
            chunk_tokens = self.chunker.count_tokens(chunk.content)
            
            if current_tokens + chunk_tokens <= limit:
                final_context.append(chunk)
                current_tokens += chunk_tokens
            
            if current_tokens >= limit:
                break
                
        return final_context, current_tokens
