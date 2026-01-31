import json
from typing import List
from app.config import settings
from app.schemas import ContextChunk
from app.domain.decisions import DecisionProposal, DecisionResult
from openai import AsyncOpenAI

class DecisionSynthesisAgent:
    """
    Specialized agent for synthesizing actionable options from memory.
    
    Why Decision Synthesis is different from Q&A:
    - Q&A targets a single 'correct' answer based on facts.
    - Decision synthesis explores the 'possibility space'. It requires generating *multiple* viable paths
      (options) and evaluating them against conflicting criteria (trade-offs), rather than converging on one truth.
    
    Why multiple options?
    - Rarely is there one perfect decision. Providing ranked options empowers the human-in-the-loop to 
      apply their own judgment/values that might not be captured in the memory.
    - It forces the system to consider alternatives, reducing tunnel vision.
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model

    async def synthesize_decisions(
        self, 
        query: str, 
        context_chunks: List[ContextChunk]
    ) -> List[DecisionProposal]:
        """
        Synthesize actionable proposals based on the query and retrieved context.
        """
        
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, context_chunks)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3, # Slightly higher than 0 to allow creative option generation
                response_format={"type": "json_object"}
            )
            
            raw_json = response.choices[0].message.content
            data = json.loads(raw_json)
            
            # Expecting a wrapper object with 'proposals' based on prompt
            proposals_data = data.get("proposals", [])
            proposals = [DecisionProposal(**p) for p in proposals_data]
            
            return proposals

        except Exception as e:
            # In a real system, we'd log this and maybe return a fallback "Need more info" proposal
            raise RuntimeError(f"Decision Synthesis failed: {str(e)}")

    def _build_system_prompt(self) -> str:
        return """You are a Decision Synthesis Agent.
Your goal is to synthesize 2-4 actionable decision proposals based on the user's situation and memory context.

Output a JSON object with this structure:
{
  "proposals": [
    {
      "id": "short_snake_case_id",
      "title": "Actionable Title",
      "rationale": "Why this is a good idea...",
      "impact": "high|medium|low",
      "scores": [
        {
          "value": 0.0-1.0,
          "confidence": 0.0-1.0,
          "dimension": "overall"
        }
      ],
      "supporting_evidence": [
        {
          "evidence_id": "chunk_id_from_context",
          "content_snippet": "exact quote or summary",
          "relevance_score": 0.0-1.0
        }
      ]
    }
  ]
}

RULES:
1. **Grounding**: You MUST strictly reference 'evidence_id' from the provided Context. Do not hallucinatie IDs.
2. **Impact**: Estimate impact (high/medium/low) based on the stakes implied in memory.
3. **Multiplicity**: Generate at least 2 distinct options if the potential for choice exists.
4. **Confidence**: Initial confidence score should reflect how complete the memory context is for this option.
"""

    def _build_user_prompt(self, query: str, context_chunks: List[ContextChunk]) -> str:
        context_text = "\n\n".join([f"[Chunk ID: {c.chunk_id}] {c.content}" for c in context_chunks])
        if not context_text:
            context_text = "(No relevant memory. Generate generic proposals with low confidence.)"

        return f"""
Context / Evidence:
{context_text}

User Query / Context:
{query}

Synthesize actionable decision proposals now.
"""
