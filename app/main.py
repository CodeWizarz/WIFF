from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager
from app.database import get_db, init_db
from app.routers import ingestion, query, agent, memory, maintenance, decision
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    await init_db()
    yield
    # Shutdown (if needed)

app = FastAPI(
    title="AI Agent with Persistent Memory",
    description="Memory-enabled AI agent with vector + graph storage",
    version="0.1.0",
    lifespan=lifespan
)


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Include routers
app.include_router(ingestion.router, prefix="/api/v1/ingest", tags=["Ingestion"])
app.include_router(query.router, prefix="/api/v1/query", tags=["Query"])
app.include_router(agent.router, prefix="/api/v1/agent", tags=["Agent"])
app.include_router(memory.router, prefix="/api/v1/memory", tags=["Memory"])
app.include_router(maintenance.router, prefix="/api/v1/maintenance", tags=["Maintenance"])
app.include_router(decision.router, prefix="/api/v1/decision", tags=["Decision"])

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def root():
    return FileResponse("app/static/index.html")

@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health check endpoint"""
    return {"status": "healthy", "database": "connected"}
