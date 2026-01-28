from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models import Entity, Relationship
from app.schemas import EntityResponse

router = APIRouter()

@router.get("/entity/{entity_id}", response_model=EntityResponse)
async def get_entity(
    entity_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get entity details and its relationships.
    """
    # Get entity
    result = await db.execute(select(Entity).where(Entity.id == entity_id))
    entity = result.scalars().first()
    
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    # Get related entities (1-hop)
    rel_result = await db.execute(
        select(Relationship, Entity)
        .join(Entity, Relationship.target_id == Entity.id)
        .where(Relationship.source_id == entity_id)
    )
    
    related = []
    for rel, target_entity in rel_result:
        related.append({
            "id": target_entity.id,
            "name": target_entity.entity_name,
            "type": target_entity.entity_type,
            "relationship": rel.relationship_type,
            "strength": rel.strength
        })
    
    return EntityResponse(
        id=entity.id,
        entity_type=entity.entity_type,
        entity_name=entity.entity_name,
        properties=entity.properties,
        related_entities=related
    )

@router.get("/recent", response_model=list)
async def get_recent_memories(
    limit: int = 10,
    source_type: str = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Get most recent chunks/memories. Useful for verifying learning.
    """
    from app.models import DocumentChunk
    from sqlalchemy import desc
    
    query = select(DocumentChunk).order_by(desc(DocumentChunk.created_at)).limit(limit)
    
    if source_type:
        query = query.where(DocumentChunk.source_type == source_type)
        
    result = await db.execute(query)
    chunks = result.scalars().all()
    
    return [
        {
            "id": c.id,
            "content": c.content,
            "source_type": c.source_type,
            "created_at": c.created_at,
            "doc_metadata": c.doc_metadata
        }
        for c in chunks
    ]
