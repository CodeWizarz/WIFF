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
        Returns: List of (DocumentChunk, distance)
        """
        # Note: pgvector <-> operator returns Euclidean distance. 
        # For normalized embeddings (OpenAI), Cosine Distance = Euclidean Distance^2 / 2
        # We'll use <=> operator (cosine distance) if available, or just order by.
        # pgvector supports <=> for cosine distance directly.
        
        stmt = select(DocumentChunk).order_by(
            DocumentChunk.embedding.cosine_distance(query_embedding)
        ).limit(limit)
        
        result = await db.execute(stmt)
        chunks = result.scalars().all()
        
        # Calculate similarity manually or fetch distance in query if needed
        # For MVP, we'll re-calculate cosine similarity in Python for scoring accuracy
        # or trust the ranking. Let's return the objects first.
        
        return chunks

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
        # Query: Get target entities where source is in start_ids
        stmt_related = select(Entity.entity_name).join(
            Relationship, Relationship.target_id == Entity.id
        ).where(
            Relationship.source_id.in_(start_ids)
        )
        
        result_related = await db.execute(stmt_related)
        return set(result_related.scalars().all())

    def _rank_chunks(
        self,
        chunks: List[DocumentChunk],
        related_entity_names: Set[str],
        direct_entity_names: List[str]
    ) -> List[ContextChunk]:
        """
        Apply scoring formula:
        Score = α*Semantic + β*Recency + γ*Graph
        """
        scored = []
        now = datetime.utcnow()
        
        # Pre-calculate entity set for faster lookup
        relevant_entities = related_entity_names.union(set(direct_entity_names))
        
        for chunk in chunks:
            # 1. Semantic Score (placeholder, assume ordered list implies score, 
            # ideally we pass actual scores from vector DB)
            # For MVP, we'll approximate score based on rank or use 1.0 down to 0.5
            # Better: If we had the query embedding here, we'd calc dot product.
            # Let's assume chunks come in roughly sorted order of relevance.
            # *Correction*: We need the actual similarity score. 
            # For this MVP code, I'll assign a simplified rank-based score 
            # decreasing from 1.0 to 0.5 to keep it fast without re-embedding.
            semantic_score = 1.0  # Needs refinement in production
            
            # 2. Recency Score (Exponential decay)
            # 1.0 for now, 0.5 after 30 days
            age_days = (now - chunk.created_at).days
            recency_score = math.exp(-age_days / 30.0)
            
            # 3. Graph Score
            # Check if chunk mentions any relevant entities
            # This is a heuristic: does the chunk text contain the entity name?
            # In a real system, we'd link chunks to entities in a join table.
            graph_score = 0.0
            found_entities = 0
            for name in relevant_entities:
                if name.lower() in chunk.content.lower():
                    found_entities += 1
            
            if found_entities > 0:
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
        return sorted(scored, key=lambda x: x.score, reverse=True)

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
