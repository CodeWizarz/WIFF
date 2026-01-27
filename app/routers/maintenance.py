from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.services.maintenance import MaintenanceService

router = APIRouter()
maintenance_service = MaintenanceService()

@router.post("/consolidate", response_model=dict)
async def run_consolidation(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger entity consolidation manually.
    Merges duplicate entities (e.g. 'OpenAI' and 'Open AI').
    """
    try:
        # For MVP, we run awaiting here to return result count.
        # In prod, use background_tasks.add_task(...)
        count = await maintenance_service.run_consolidation(db)
        return {"message": "Consolidation complete", "merged_entities": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
