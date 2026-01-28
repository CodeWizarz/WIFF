# System Failure Modes & Mitigation Strategies

This document outlines realistic failure modes for the implementation of the memory-augmented agent system, covering retrieval, context management, and failure.

## 1. Context Drift / Hallucination of Relevance
**Description**: The system retrieves memories that are semantically similar but contextually irrelevant to the *current* goal, causing the agent to "drift" away from the user's immediate intent. Or, as the conversation grows, the "working memory" is flooded with old turns, and the agent forgets the original instruction.

**Example**: 
- User: "Fix the bug in `main.py`."
- *Later*: "Now optimize the database."
- Agent retrieves `main.py` bugs because "optimize" semantically overlaps with "fix code" + previous context, ignoring the switch to "database".

**Mitigations**:
- **Query Rewriting**: Before retrieval, use an LLM call to rewrite the user's query into a standalone statement that incorporates necessary context from the immediate chat history (e.g., "Optimize the database configuration in `database.py`").
- **Strict Recency Weighting**: Apply a decay function to interaction logs so that very old conversation turns have significantly lower retrieval scores unless they are flagged as "core facts."
- **Goal Anchor**: Maintain a persistent "Goal" slot in the system prompt that doesn't cycle out with standard context window sliding.

## 2. Stale Memory
**Description**: The system retains and retrieves information that was once true but is now outdated.

**Example**: 
- *Day 1*: "We are using Postgres 14." (Stored as Fact)
- *Day 30*: "We migrated to Postgres 16." (Stored as New Fact)
- *Query*: "What DB version?" -> Agent retrieves *both* chunks. It might answer "14", "16", or "14 and 16".

**Mitigations**:
- **Memory Consolidation (Background Job)**: Periodically run a job that clusters similar facts and asks an LLM to resolve conflicts ("Merge these two facts into the latest truth").
- **Time-Weighted Retrieval**: In `retrieval.py`, ensure the scoring formula heavily penalizes snippets older than $X$ days if a conflicting newer snippet exists (hard to detect conflict without read-time logic).
- **"Valid_Until" Metadata**: When storing volatile facts (e.g., versions, current tasks), ask the extraction layer to estimate a `valid_until` date or tag it as `volatile`.

## 3. Over-Retrieval (Noise Flooding)
**Description**: The vector search returns the requested top-k (e.g., 20) chunks because they *exist*, not because they are *good*. This dilutes the context window with low-relevance noise, distracting the LLM.

**Example**: 
- Query: "How do I logging?"
- Retrieval: Returns 10 snippets of code that *contain* the word "log" (interaction logs, login logic, logical operators) instead of the actual logging utility.

**Mitigations**:
- **Relevance Thresholding**: In `retrieval.py`, drop any chunk with a similarity score $< 0.75$ (tuned threshold). Return *nothing* if no chunks meet the bar.
- **Cross-Encoder Re-ranking**: Use a potentially slower but more accurate "Cross-Encoder" model to re-rank the top 20 candidate chunks from the vector store and take only the top 5.
- **Maximum Marginal Relevance (MMR)**: During ranking, penalize chunks that are too similar to *already selected* chunks to ensure diversity in the context.

## 4. Latency Spikes
**Description**: The standard RAG loop (Embed -> Vector Search -> Graph Search -> Rank -> Generate) adds significant latency, potentially making the chat feel sluggish (e.g., > 3-5 seconds).

**Example**: 
- Vector database cold start.
- Large context injection slows down 'Time to First Token' (TTFT) on the LLM.

**Mitigations**:
- **Async/Parallel Retrieval**: Ensure Vector Search and Graph Search run primarily in parallel `asyncio.gather()`.
- **Hybrid Cache**: Cache the "Retrieval Result" for identical or highly similar queries (Semantic Cache). If the user asks the same question twice, skip the DB lookup.
- **Speculative Execution**: Start generating a preliminary answer based on generic knowledge *while* retrieval happens (complex UI required).
- **Tiered Context**: Use a faster, cheaper model to summarize the retrieved chunks *before* feeding them to the main reasoning model.

## 5. Privacy Leaks (PII & ACLs)
**Description**: The system stores sensitive data (API keys, passwords, personal names) in the vector store. This data might be retrieved for a user who shouldn't see it, or leaked into logs.

**Example**: 
- User A pastes AWS keys to debug. System ingests it.
- User B asks "Show me configuration examples." System retrieves User A's chunk with the live keys.

**Mitigations**:
- **PII Redaction Layer**: run a Presidio/Regex pass *before* `ingestion_service.ingest_document()` to replace emails, keys, and phones with `<REDACTED>`.
- **User-Level Partitioning**: Store a `user_id` or `org_id` in the `metadata` of every chunk. Enforce a strict filter in `_vector_search` (`where user_id == current_user`).
- **LLM Output Filter**: A final lightweight regex pass on the *generated response* to catch any accidental leakage before sending to client.
