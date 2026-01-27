from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Tuple
from app.config import settings
from app.services.retrieval import RetrievalService
from app.models import InteractionLog
from app.schemas import ContextChunk

class AgentService:
    """
    Orchestrates the agent loop:
    Retrieval -> Prompt Engineering -> LLM -> Response -> Logging
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.retrieval_service = RetrievalService()
        self.model = settings.llm_model
    
    async def run_agent(
        self,
        db: AsyncSession,
        query: str,
        token_budget: int = 4000,
        session_id: str = None
    ) -> Tuple[str, List[ContextChunk], int]:
        """
        Run the full agent loop.
        """
        # 1. Retrieve Context
        context_chunks, total_tokens = await self.retrieval_service.retrieve_context(
            db, query, token_budget, session_id
        )
        
        # 2. Construct Prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, context_chunks)
        
        # 3. Call LLM
        response_text = await self._call_llm(system_prompt, user_prompt)
        
        # 4. Log Interaction (Async/Fire-and-forget in production)
        await self._log_interaction(db, query, context_chunks, response_text, session_id)
        
        return response_text, context_chunks, total_tokens
    
    def _build_system_prompt(self) -> str:
        return """You are a memory-augmented AI assistant. 
Your goal is to answer the user's query accurately using the provided context.
- Use the provided context chunks to answer.
- If the context mentions specific entities or past events, reference them.
- If the context is empty or irrelevant, generic knowledge is acceptable but prioritize memory.
- Be concise and direct."""

    def _build_user_prompt(self, query: str, context_chunks: List[ContextChunk]) -> str:
        context_str = "\n\n".join([
            f"--- [Timestamp: {c.created_at}] [Source: {c.source_type}] ---\n{c.content}"
            for c in context_chunks
        ])
        
        if not context_chunks:
            context_str = "(No relevant memory found)"
            
        return f"""Relevant Context from Memory:
{context_str}

User Query:
{query}"""

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content

    async def _log_interaction(
        self,
        db: AsyncSession,
        query: str,
        context: List[ContextChunk],
        response: str,
        session_id: str
    ):
        """Log the interaction for future reinforcement learning"""
        chunk_ids = [c.chunk_id for c in context]
        
        log = InteractionLog(
            session_id=session_id,
            query=query,
            retrieved_chunk_ids=chunk_ids,
            response=response
        )
        db.add(log)
        await db.commit()
