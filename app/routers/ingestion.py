from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.schemas import IngestDocumentRequest, IngestDocumentResponse
from app.services.ingestion import IngestionService

router = APIRouter()
ingestion_service = IngestionService()

@router.post("/document", response_model=IngestDocumentResponse)
async def ingest_document(
    request: IngestDocumentRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Ingest a document or conversation.
    
    This will:
    1. Chunk the content
    2. Generate embeddings
    3. Store in vector database
    4. Extract entities and build graph
    """
    try:
        chunk_ids = await ingestion_service.ingest_document(
            db=db,
            content=request.content,
            source_type=request.source_type,
            source_id=request.source_id,
            metadata=request.metadata
        )
        
        return IngestDocumentResponse(
            chunk_ids=chunk_ids,
            num_chunks=len(chunk_ids),
            message=f"Successfully ingested document with {len(chunk_ids)} chunks"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
