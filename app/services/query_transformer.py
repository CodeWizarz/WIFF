from openai import AsyncOpenAI
from app.config import settings

class QueryTransformer:
    """
    Reformulates user queries to be standalone and context-aware,
    mitigating 'Context Drift' and improving retrieval quality.
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model
        
    async def rewrite_query(self, original_query: str, chat_history: list[dict]) -> str:
        """
        Rewrite the query based on history.
        chat_history should be a list of {"role": "user"|"assistant", "content": "..."}
        """
        if not chat_history:
            return original_query
            
        # Keep only the last few turns to avoid excessive context
        recent_history = chat_history[-4:]
        
        history_text = ""
        for msg in recent_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            history_text += f"{role.title()}: {content}\n"
            
        prompt = f"""You are a query rewriting assistant. 
Your specific and only task is to rewrite the LAST User query to be a standalone sentence that fully incorporates the context from the conversation history.
- Resolve pronouns (it, he, they, that) to their specific referents.
- Fill in missing details implied by the previous turns.
- Do NOT answer the query. Just rewrite it.
- If the query is already standalone, return it exactly as is.

Conversation History:
{history_text}

User's Last input:
{original_query}

Rewritten Query:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200
            )
            rewritten = response.choices[0].message.content.strip()
            # Safety: if LLM returns something weird or empty, fallback
            return rewritten if rewritten else original_query
        except Exception as e:
            print(f"Query rewriting failed: {e}")
            return original_query
