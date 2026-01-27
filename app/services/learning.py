import json
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, Optional
from app.config import settings
from app.services.ingestion import IngestionService
from app.services.embeddings import EmbeddingService
from app.models import DocumentChunk

class LearnerService:
    """
    Reflects on agent interactions to learn new facts, procedures, or episodes.
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model
        # Reuse existing services for storage
        self.ingestion_service = IngestionService()
        self.embedding_service = EmbeddingService()

    async def learn_from_interaction(
        self,
        db: AsyncSession,
        query: str,
        response: str,
        session_id: str = None
    ):
        """
        Main entry point: Evaluate -> Distill -> Store
        """
        # 1. Fast Filter: Skip chit-chat or very short interactions
        if not self._passes_fast_filter(query, response):
            return

        # 2. Evaluate: Is this worth remembering?
        evaluation = await self._evaluate_signal(query, response)
        
        if not evaluation.get("should_remember", False):
            return

        # 3. Distill & Store based on type
        memory_type = evaluation.get("memory_type", "episodic")
        importance = evaluation.get("importance", 0.5)
        
        if memory_type == "semantic":
            await self._learn_semantic(db, query, response)
        elif memory_type == "procedural":
            await self._learn_procedural(db, query, response, importance)
        else:
            await self._learn_episodic(db, query, response, session_id, importance)

    def _passes_fast_filter(self, query: str, response: str) -> bool:
        """Heuristic filter to avoid LLM calls for obvious noise"""
        # Too short?
        if len(query.split()) < 3 and len(response.split()) < 5:
            return False
            
        # Phatic conversation?
        phatic_triggers = ["thank", "thanks", "ok", "cool", "bye", "hello", "hi"]
        if query.lower().strip() in phatic_triggers:
            return False
            
        return True

    async def _evaluate_signal(self, query: str, response: str) -> Dict[str, Any]:
        """Ask LLM if this interaction contains valuable long-term info"""
        prompt = f"""Analyze this interaction.
User: {query}
Agent: {response}

Criteria for keeping:
1. Contains new facts about the user or world (Semantic)
2. Is a significant task completion (Episodic)
3. Contains a reusable solution or workflow (Procedural)

Criteria for discarding:
1. Chit-chat ("hi", "thanks")
2. Failed attempts with no clear lesson
3. Information already widely known or generic

Return JSON:
{{ 
    "should_remember": bool, 
    "memory_type": "semantic" | "procedural" | "episodic",
    "importance": 0.0-1.0,
    "reason": "explanation" 
}}"""

        try:
            res = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(res.choices[0].message.content)
        except Exception:
            return {"should_remember": False}

    async def _learn_semantic(self, db: AsyncSession, query: str, response: str):
        """Extract facts and update graph"""
        # Leverage IngestionService's entity extraction logic
        # We treat the synthesis of Q+A as a "fact source"
        text_to_ingest = f"Fact derived from interaction:\nUser asked: {query}\nAnswer: {response}"
        
        # We can use the ingestion service to extract entities directly
        # Note: IngestionService expects source_id. We can use a special ID.
        await self.ingestion_service.ingest_document(
            db=db,
            content=text_to_ingest,
            source_type="learned_fact",
            metadata={"origin": "interaction_reflection"}
        )

    async def _learn_procedural(self, db: AsyncSession, query: str, response: str, importance: float):
        """Extract workflow/recipe"""
        prompt = f"""Extract the reusable workflow or solution from this interaction.
User: {query}
Agent: {response}

Summarize as a clear step-by-step recipe.
"""
        res = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        recipe = res.choices[0].message.content
        
        # Store as a chunk with high importance
        # We manually create the chunk to set specific metadata
        embedding = await self.embedding_service.embed_text(recipe)
        
        chunk = DocumentChunk(
            content=recipe,
            embedding=embedding,
            source_type="procedure",
            metadata={
                "importance": importance,
                "trigger_query": query
            }
        )
        db.add(chunk)
        await db.commit()

    async def _learn_episodic(self, db: AsyncSession, query: str, response: str, session_id: str, importance: float):
        """Summarize the episode"""
        prompt = f"""Summarize this interaction in 1 sentence for future context.
User: {query}
Agent: {response}
"""
        res = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        summary = res.choices[0].message.content
        
        # Store as episode
        embedding = await self.embedding_service.embed_text(summary)
        
        chunk = DocumentChunk(
            content=summary,
            embedding=embedding,
            source_type="episode",
            source_id=session_id,
            metadata={
                "importance": importance,
                "timestamp": "now" # In real app, use datetime
            }
        )
        db.add(chunk)
        await db.commit()
