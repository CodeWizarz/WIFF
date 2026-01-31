from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.schemas import DecisionRequest
from app.schemas_feedback import DecisionFeedback
from app.domain.decisions import DecisionResult
from app.services.decision_engine import DecisionEngine
from app.services.ingestion import IngestionService
from fastapi import BackgroundTasks

router = APIRouter()
decision_engine = DecisionEngine()
ingestion_service = IngestionService()

@router.post("/", response_model=DecisionResult)
async def analyze_decision(
    request: DecisionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Submit a problem statement and get a structured decision analysis
    based on the agent's memory.
    """
    try:
        result = await decision_engine.analyze_decision(
            db=db,
            query=request.query,
            token_budget=request.token_budget,
            session_id=request.session_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decision analysis failed: {str(e)}")

@router.post("/feedback")
async def submit_feedback(
    feedback: DecisionFeedback,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Provide human feedback on a decision to improve future performance.
    Rejected decisions with explanations become part of the agent's memory (Safe Deployment).
    """
    try:
        # Construct a memory document representing this feedback
        status_str = "APPROVED" if feedback.approved else "REJECTED"
        memory_content = f"""
[DECISION FEEDBACK]
Decision ID: {feedback.decision_id}
Original Query: {feedback.original_query}
Proposal: {feedback.selected_proposal_title}
User Status: {status_str}
Feedback Notes: {feedback.feedback_text}
"""
        # Ingest into memory (background task to not block response)
        # We need to wrap the async ingestion in a way that works with BackgroundTasks 
        # or just await it if it's fast. Ingestion involves embeddings, so might take a sec.
        # For simplicity in this demo, we await it.
        
        await ingestion_service.ingest_document(
            db=db,
            content=memory_content,
            source_type="governance_feedback",
            metadata={
                "decision_id": feedback.decision_id,
                "approved": feedback.approved,
                "user_id": feedback.user_id
            }
        )
        
        return {"message": "Feedback received and integrated into memory."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process feedback: {str(e)}")
