from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Dict, Any, Tuple
from app.models import DocumentChunk, Entity, Relationship
from app.services.chunker import DocumentChunker
from app.services.embeddings import EmbeddingService
from app.services.entity_extractor import EntityExtractor
from app.services.quality import QualityFilter
from app.services.dedup import DeDuplicationService

class IngestionService:
    """Orchestrates document ingestion pipeline with Q&A and Dedup"""
    
    def __init__(self):
        self.chunker = DocumentChunker()
        self.embedding_service = EmbeddingService()
        self.entity_extractor = EntityExtractor()
        self.quality_filter = QualityFilter()
        self.dedup_service = DeDuplicationService()
    
    async def ingest_document(
        self,
        db: AsyncSession,
        content: str,
        source_type: str,
        source_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> Tuple[List[int], str]:
        """
        Ingest a document: Quality -> Dedup -> Chunk -> Dedup(Chunk) -> Store -> Graph
        """
        # 1. Quality Gate
        if not self.quality_filter.is_worth_storing(content):
            return [], "Skipped: Low quality content"
        
        # 2. Global Deduplication (Exact Hash)
        doc_hash = self.dedup_service.compute_hash(content)
        # We don't have a "Document" table, but we can check if any chunk has this source_id 
        # or just rely on chunk-level dedup.
        # For MVP, let's skip doc-level check unless we store full docs.
        # Proceed to chunking.
        
        # 3. Chunking
        chunks = self.chunker.chunk(content)
        
        # 4. Process Chunks (Embed + Dedup)
        # Note: We need to embed first to check semantic duplicates
        embeddings = await self.embedding_service.embed_batch(chunks)
        
        chunk_ids = []
        skipped_count = 0
        
        for chunk_text, embedding in zip(chunks, embeddings):
            # A. Exact Chunk Deduplication
            chunk_hash = self.dedup_service.compute_hash(chunk_text)
            if await self.dedup_service.is_duplicate(db, chunk_hash):
                skipped_count += 1
                continue
                
            # B. Semantic Deduplication (Optional - expensive)
            # if await self.dedup_service.is_near_duplicate(db, embedding):
            #    skipped_count += 1
            #    continue
            
            # Store
            chunk = DocumentChunk(
                content=chunk_text,
                embedding=embedding,
                content_hash=chunk_hash,
                source_type=source_type,
                source_id=source_id,
                doc_metadata=metadata or {}
            )
            db.add(chunk)
            await db.flush()
            chunk_ids.append(chunk.id)
        
        await db.commit()
        
        if not chunk_ids:
            return [], f"Skipped: All {len(chunks)} chunks were duplicates"

        # 5. Extract Entities (Async/Background in production)
        # We pass the full content for extraction, or could pass chunks
        await self._extract_and_store_entities(db, content, chunk_ids)
        
        return chunk_ids, "Success"
    
    async def _extract_and_store_entities(
        self,
        db: AsyncSession,
        content: str,
        chunk_ids: List[int]
    ):
        """Extract entities and relationships from content"""
        # Extract entities using LLM
        extracted_entities = await self.entity_extractor.extract_entities(content)
        
        entity_map = {}  # name -> db_id
        
        for entity_data in extracted_entities:
            entity_name = entity_data["name"]
            entity_type = entity_data["type"]
            
            # Check if entity already exists (simple string match for MVP)
            result = await db.execute(
                select(Entity).where(
                    Entity.entity_name == entity_name,
                    Entity.entity_type == entity_type
                )
            )
            existing_entity = result.scalars().first()
            
            if existing_entity:
                entity_map[entity_name] = existing_entity.id
            else:
                # Create new entity with embedding
                embedding = await self.embedding_service.embed_text(entity_name)
                new_entity = Entity(
                    entity_type=entity_type,
                    entity_name=entity_name,
                    embedding=embedding,
                    properties=entity_data.get("properties", {})
                )
                db.add(new_entity)
                await db.flush()
                entity_map[entity_name] = new_entity.id
        
        # Create relationships (co-occurrence)
        entity_names = list(entity_map.keys())
        for i, name1 in enumerate(entity_names):
            for name2 in entity_names[i+1:]:
                # Create bidirectional relationship
                rel = Relationship(
                    source_id=entity_map[name1],
                    target_id=entity_map[name2],
                    relationship_type="co_occurrence",
                    strength=1.0
                )
                db.add(rel)
        
        await db.commit()
