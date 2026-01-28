# AI Agent with Persistent Memory

Production-grade AI agent with hybrid vector + graph memory system for context-aware task execution.

## Architecture

- **Vector Store**: pgvector (Postgres extension) for semantic similarity search
- **Graph Store**: Postgres adjacency list for entity relationships
- **Backend**: FastAPI with async SQLAlchemy
- **Embeddings**: OpenAI `text-embedding-3-small`
- **LLM**: GPT-4o-mini

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API key

### 2. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env and add your OPENAI_API_KEY
```

### 3. Start Database

```bash
docker-compose up -d
```

This starts Postgres with pgvector extension on port 5432.

### 4. Run API Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`

Docs at `http://localhost:8000/docs`

## API Endpoints

### Ingestion
- `POST /api/v1/ingest/document` - Ingest documents or conversations

### Query
- `POST /api/v1/query` - Retrieve task-relevant context (coming in Phase 4)

### Agent
- `POST /api/v1/agent` - Get memory-augmented agent responses (coming in Phase 5)

### Memory
- `GET /api/v1/memory/entity/{id}` - Inspect entity graph

## Example Usage

```python
import httpx

# Ingest a document
response = httpx.post("http://localhost:8000/api/v1/ingest/document", json={
    "content": "OpenAI released GPT-4 in March 2023...",
    "source_type": "document",
    "metadata": {"author": "OpenAI"}
})

print(response.json())
# {"chunk_ids": [1, 2], "num_chunks": 2, "message": "Successfully ingested..."}
```

## Project Structure

```
app/
├── main.py              # FastAPI app
├── config.py            # Settings
├── database.py          # DB connection
├── models.py            # SQLAlchemy models
├── schemas.py           # Pydantic schemas
├── routers/             # API endpoints
│   ├── ingestion.py
│   ├── query.py
│   ├── agent.py
│   └── memory.py
└── services/            # Business logic
    ├── chunker.py
    ├── embeddings.py
    ├── entity_extractor.py
    └── ingestion.py
```

## Development Roadmap

- [x] Phase 1: Backend scaffolding
- [x] Phase 2: Ingestion pipeline
- [ ] Phase 3: Database migrations
- [ ] Phase 4: Context retrieval engine
- [ ] Phase 5: Agent orchestration
- [ ] Phase 6: Memory quality improvements

## Configuration

See `.env.example` for all configurable parameters:

- Embedding model and dimensions
- Chunk size and overlap
- Retrieval scoring weights (α, β, γ)
- Token budget

## License

MIT
