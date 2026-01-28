from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.schemas import AgentRequest, AgentResponse

router = APIRouter()

@router.post("/", response_model=AgentResponse)
async def agent_query(
    request: AgentRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Agent endpoint with memory-augmented responses.
    
    This endpoint will be implemented in Phase 5.
    """
    # Placeholder for agent loop
    return AgentResponse(
        response="Agent not yet implemented",
        context_used=[],
        total_tokens=0
    )
