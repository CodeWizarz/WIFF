# Setup Guide

## Quick Start (3 steps)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Your OpenAI API Key

Edit `.env` file and replace `sk-your-key-here` with your actual OpenAI API key.

### 3. Start the Database

**Option A: Using Docker (Recommended)**
```bash
docker-compose up -d
```

**Option B: Using Existing Postgres**
If you have Postgres installed locally:
1. Install pgvector extension: `CREATE EXTENSION vector;`
2. Update `DATABASE_URL` in `.env` to point to your Postgres instance

### 4. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

## Testing the API

### 1. Ingest a Document

```bash
curl -X POST "http://localhost:8000/api/v1/ingest/document" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "OpenAI released GPT-4 in March 2023. GPT-4 is a large multimodal model that can process both text and images. It was trained on a massive dataset and exhibits human-level performance on various professional benchmarks.",
    "source_type": "document",
    "metadata": {"source": "openai_blog"}
  }'
```

### 2. View Entity Graph

After ingestion, entities are extracted. View them:

```bash
curl "http://localhost:8000/api/v1/memory/entity/1"
```

## Troubleshooting

### Docker Compose Error on Windows

If you see "docker-compose: The term 'docker-compose' is not recognized", try:

1. **Use Docker Compose V2**:
   ```bash
   docker compose up -d
   ```

2. **Or install Docker Desktop** which includes Docker Compose

3. **Or use manual Postgres setup** (see Option B above)

### Missing OpenAI API Key

If you get authentication errors, make sure `.env` has your real API key:
```
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

### Database Connection Error

Make sure Postgres is running:
```bash
docker ps  # Should show memory_agent_db container
```

If not running:
```bash
docker compose up -d
```

## Quick Demos

### 1. Standalone Agent Flow (Mocked)
Run the end-to-end flow demo without any database dependencies:
```bash
python app/services/agent_flow_demo.py
```

### 2. Real Agent Verification
Run the actual agent against the local database (requires Docker):
```bash
python verify_agent_real.py
```

## What's Implemented
✅ **Core Architecture:**
- **Hybrid Retrieval**: Vector (pgvector) + Graph (Entities) search.
- **Agent Orchestration**: Specialized `AgentService` with memory context injection.
- **Context Drift Mitigation**: `QueryTransformer` rewrites queries based on history.
- **Learning Loop**: Background synthesis of interactions into new facts.

🚧 **In Progress:**
- Advanced conflict resolution for conflicting memories.
- User-level partitioning and multi-tenancy.

## Project Status

See `task.md` in the artifacts for detailed progress tracking.
