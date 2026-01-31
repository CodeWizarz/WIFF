import json
from typing import List
from app.config import settings
from app.schemas import ContextChunk
from app.domain.decisions import DecisionProposal, DecisionScore
from openai import AsyncOpenAI

class CriticAgent:
    """
    Reviewer agent that validates proposed decisions against evidence.
    
    Why challenging AI output is required:
    - LLMs are prone to 'hallucination' and 'sycophancy' (agreeing with the user).
    - A production decision system must have an adversarial step to catch weak reasoning 
      or unsupported claims before they reach the user.
      
    Why confidence reduction vs rejection:
    - Rejection ("This is wrong") is binary and brittle. 
    - Confidence reduction ("This might be risky") preserves the option but signals caution, 
      allowing the human to make the final call with full awareness of the uncertainty.
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model

    async def critique_proposals(
        self, 
        query: str, 
        context_chunks: List[ContextChunk],
        proposals: List[DecisionProposal]
    ) -> List[DecisionProposal]:
        """
        Review proposals, attach critiques, and adjusting confidence scores if necessary.
        """
        if not proposals:
            return []

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, context_chunks, proposals)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1, # Low temp for critical analysis
                response_format={"type": "json_object"}
            )
            
            raw_json = response.choices[0].message.content
            data = json.loads(raw_json)
            reviews = data.get("reviews", [])
            
            # Map reviews back to proposals
            # We assume the list order is preserved or use IDs if possible. 
            # For robustness, we'll try to match by ID.
            
            review_map = {r["proposal_id"]: r for r in reviews}
            
            for p in proposals:
                review = review_map.get(p.id)
                if review:
                    # Attach critique
                    p.critique = review.get("critique_text")
                    
                    # Apply confidence adjustment if defined
                    # We will look for the 'overall' score and modify it.
                    original_score = next((s for s in p.scores if s.dimension == "overall"), None)
                    if original_score:
                        # The critic outputs a modifier scalar (e.g. 0.9 for "mostly okay", 0.5 for "big doubts")
                        # Or a new confidence level directly.
                        # Let's say the critic outputs a "confidence_adjustment_factor" (0.0 to 1.0).
                        factor = review.get("confidence_adjustment_factor", 1.0)
                        
                        # We apply this factor to the model's confidence
                        original_score.confidence *= factor
                        
                        # Add a score entry for the penalty if significant
                        if factor < 0.9:
                            p.scores.append(DecisionScore(
                                value=factor,
                                confidence=1.0,
                                dimension="critic_confidence_factor"
                            ))

            return proposals

        except Exception as e:
            # If critic fails, return proposals as-is but log error
            print(f"Critic failed: {e}")
            return proposals

    def _build_system_prompt(self) -> str:
        return """You are a Critic Agent.
Your goal is to REVIEW proposed decisions for logical fallacies, weak evidence, or missing context.

Output a JSON object with this structure:
{
  "reviews": [
    {
      "proposal_id": "id_of_proposal",
      "critique_text": "Detailed critique of why this might be wrong or risky...",
      "confidence_adjustment_factor": 0.0-1.0
    }
  ]
}

RULES:
1. **Skepticism**: Look for assumptions not backed by the provided 'Memory Context'.
2. **Adjustment**: 
   - Return 1.0 if the proposal is solid.
   - Return < 1.0 if there are risks (e.g., 0.8 for minor gaps, 0.5 for major hallucinations).
3. **Evidence**: If a proposal cites evidence that doesn't actually support it, critique it heavily.
"""

    def _build_user_prompt(
        self, 
        query: str, 
        context_chunks: List[ContextChunk], 
        proposals: List[DecisionProposal]
    ) -> str:
        context_text = "\n\n".join([f"[Chunk ID: {c.chunk_id}] {c.content}" for c in context_chunks])
        
        proposals_text = ""
        for p in proposals:
            evidence_str = ", ".join([e.content_snippet for e in p.supporting_evidence])
            proposals_text += f"""
---
ID: {p.id}
Title: {p.title}
Rationale: {p.rationale}
Cited Evidence: {evidence_str}
---
"""

        return f"""
Context / Truth:
{context_text}

Query: {query}

Proposals to Critique:
{proposals_text}

Provide your critical review in JSON.
"""
