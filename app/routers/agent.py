from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.schemas import AgentRequest, AgentResponse
from app.services.agent import AgentService

router = APIRouter()
agent_service = AgentService()

@router.post("/", response_model=AgentResponse)
async def agent_query(
    request: AgentRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Agent endpoint with memory-augmented responses.
    """
    try:
        response_text, context_used, total_tokens = await agent_service.run_agent(
            db=db,
            query=request.query,
            token_budget=request.token_budget,
            session_id=request.session_id
        )
        
        return AgentResponse(
            response=response_text,
            context_used=context_used,
            total_tokens=total_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")
