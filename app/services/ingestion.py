from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Dict, Any
from app.models import DocumentChunk, Entity, Relationship
from app.services.chunker import DocumentChunker
from app.services.embeddings import EmbeddingService
from app.services.entity_extractor import EntityExtractor

class IngestionService:
    """Orchestrates document ingestion pipeline"""
    
    def __init__(self):
        self.chunker = DocumentChunker()
        self.embedding_service = EmbeddingService()
        self.entity_extractor = EntityExtractor()
    
    async def ingest_document(
        self,
        db: AsyncSession,
        content: str,
        source_type: str,
        source_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> List[int]:
        """
        Ingest a document: chunk, embed, store, extract entities.
        
        Returns:
            List of chunk IDs
        """
        # Step 1: Chunk the document
        chunks = self.chunker.chunk(content)
        
        # Step 2: Generate embeddings (batch)
        embeddings = await self.embedding_service.embed_batch(chunks)
        
        # Step 3: Store chunks in vector store
        chunk_ids = []
        for chunk_text, embedding in zip(chunks, embeddings):
            chunk = DocumentChunk(
                content=chunk_text,
                embedding=embedding,
                source_type=source_type,
                source_id=source_id,
                metadata=metadata or {}
            )
            db.add(chunk)
            await db.flush()
            chunk_ids.append(chunk.id)
        
        await db.commit()
        
        # Step 4: Extract entities (async, non-blocking for MVP)
        # In production, this would be a background job
        await self._extract_and_store_entities(db, content, chunk_ids)
        
        return chunk_ids
    
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
