from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.schemas import QueryRequest, QueryResponse
from app.services.retrieval import RetrievalService
import time

router = APIRouter()
retrieval_service = RetrievalService()

@router.post("/", response_model=QueryResponse)
async def query_context(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve task-relevant context for a query.
    1. Embed query
    2. Vector search + Graph search
    3. Rank and pack results
    """
    start_time = time.time()
    
    try:
        context_chunks, total_tokens = await retrieval_service.retrieve_context(
            db=db,
            query=request.query,
            token_budget=request.token_budget,
            session_id=request.session_id
        )
        
        duration = (time.time() - start_time) * 1000
        
        return QueryResponse(
            context_chunks=context_chunks,
            total_tokens=total_tokens,
            retrieval_time_ms=duration
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
