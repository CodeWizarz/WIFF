from app.services.learning import LearnerService
import asyncio

class AgentService:
    """
    Orchestrates the agent loop:
    Retrieval -> Prompt Engineering -> LLM -> Response -> Logging -> Learning
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.retrieval_service = RetrievalService()
        self.learning_service = LearnerService()
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
        
        # 4. Log Interaction & Learn (Pseudo-background)
        await self._log_interaction(db, query, context_chunks, response_text, session_id)
        
        # Trigger learning service
        # In production, use BackgroundTasks or Celery. Here we await it 
        # or use asyncio.create_task if loop allows.
        # To avoid blocking the response significantly, we should ideally background it.
        # But we need the DB session... which is a challenge if we spawn.
        # For this prototype, we'll run it purely sequentially to ensure data integrity.
        try:
            await self.learning_service.learn_from_interaction(
                db, query, response_text, session_id
            )
        except Exception as e:
            print(f"Learning failed: {e}")
        
        return response_text, context_chunks, total_tokens
    
    def _build_system_prompt(self) -> str:
        return """You are a memory-augmented AI assistant. 
Your goal is to answer the user's query accurately using the provided context.
- Use the provided context chunks to answer.
- If the context mentions specific entities or past events, reference them.
- If the context is empty or irrelevant, generic knowledge is acceptable but prioritize memory.
- Be concise and direct."""

    def _build_user_prompt(self, query: str, context_chunks: List[ContextChunk]) -> str:
        if not context_chunks:
            return f"""Relevant Context from Memory:
(No relevant memory found)

User Query:
{query}"""

        formatted_context = self._format_context(context_chunks)
            
        return f"""Relevant Context from Memory:
{formatted_context}

User Query:
{query}"""

    def _format_context(self, chunks: List[ContextChunk]) -> str:
        """
        Format context chunks into structured XML sections.
        Groups by source_type and sorts by date/importance.
        """
        # Buckets
        facts = []
        history = []
        procedures = []
        others = []
        
        for c in chunks:
            # Format: [YYYY-MM-DD] Content
            date_str = c.created_at.strftime("%Y-%m-%d")
            entry = f"[{date_str}] {c.content.strip()}"
            
            st = c.source_type
            if st == "learned_fact":
                facts.append(entry)
            elif st == "episode" or st == "conversation":
                history.append(entry)
            elif st == "procedure":
                procedures.append(entry)
            else:
                others.append(entry)
        
        # Build XML
        sections = []
        
        if facts:
            sections.append("<facts>\n- " + "\n- ".join(facts) + "\n</facts>")
            
        if history:
            sections.append("<relevant_history>\n- " + "\n- ".join(history) + "\n</relevant_history>")
            
        if procedures:
            sections.append("<suggested_procedures>\n- " + "\n- ".join(procedures) + "\n</suggested_procedures>")
            
        if others:
            sections.append("<general_context>\n- " + "\n- ".join(others) + "\n</general_context>")
            
        return "<memory_context>\n" + "\n\n".join(sections) + "\n</memory_context>"

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
