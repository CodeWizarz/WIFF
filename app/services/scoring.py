from typing import Dict, List
from datetime import datetime, timezone
import math

from app.domain.decisions import DecisionProposal, DecisionScore
from app.schemas import ContextChunk

class ScoringService:
    """
    Module responsible for normalizing and calibrating confidence scores.
    
    Why raw LLM confidence is meaningless:
    - LLMs are trained to be persuasive, not truthful. A model can be 99% 'confident' 
      in a completely hallucinated fact because the sentence structure is highly probable.
    - Raw probability logits do not map linearly to real-world probability (probability of correctness).
    
    Why calibration matters:
    - Decision makers need to compare apples to apples. A 0.8 from one run must mean the same risk profile 
      as a 0.8 from another.
    - We must mechanically penalize structural weaknesses (e.g., lack of evidence, old data) 
      that the LLM might overlook or be unaware of.
    """

    def calibrate_proposal(
        self, 
        proposal: DecisionProposal, 
        context_map: Dict[str, ContextChunk]
    ) -> DecisionProposal:
        """
        Apply penalties and normalization to the proposal's scores.
        """
        # Find the overall score to adjust
        overall_score = next((s for s in proposal.scores if s.dimension == 'overall'), None)
        if not overall_score:
            return proposal
            
        raw_confidence = overall_score.confidence
        
        # 1. Evidence Penalties
        # Calculate heuristics based on the underlying evidence
        evidence_penalty = self._calculate_evidence_penalty(proposal, context_map)
        recency_penalty = self._calculate_recency_penalty(proposal, context_map)
        
        # 2. Apply Calibration
        # We start with the raw confidence and subtract penalties.
        # This effectively 'damps' the LLM's enthusiasm.
        
        calibrated_confidence = raw_confidence * evidence_penalty * recency_penalty
        
        # 3. Normalize
        # Ensure it stays within 0.0 - 1.0 (though multiplication guarantees it if inputs are <1)
        calibrated_confidence = max(0.0, min(1.0, calibrated_confidence))
        
        # Update the score
        overall_score.confidence = calibrated_confidence
        
        # Add breakdown for explainability
        if overall_score.breakdown is None:
            overall_score.breakdown = {}
        
        overall_score.breakdown.update({
            "raw_llm_confidence": raw_confidence,
            "evidence_penalty_factor": evidence_penalty,
            "recency_penalty_factor": recency_penalty,
            "calibration_note": "Confidence adjusted based on evidence density and age."
        })
        
        return proposal

    def _calculate_evidence_penalty(self, proposal: DecisionProposal, context_map: Dict[str, ContextChunk]) -> float:
        """
        Penalize if there is very little evidence supporting the decision.
        Returns a factor (0.0 to 1.0).
        """
        # Count unique evidence chunks cited
        unique_chunks = set(e.evidence_id for e in proposal.supporting_evidence if e.evidence_id in context_map)
        count = len(unique_chunks)
        
        if count == 0:
            return 0.5  # Heavy penalty for zero grounded evidence
        elif count == 1:
            return 0.8  # Slight penalty for single-source
        else:
            return 1.0  # Sufficient evidence density

    def _calculate_recency_penalty(self, proposal: DecisionProposal, context_map: Dict[str, ContextChunk]) -> float:
        """
        Penalize if the evidence is old.
        Returns a factor (0.0 to 1.0).
        """
        now = datetime.now(timezone.utc)
        ages = []
        
        for e in proposal.supporting_evidence:
            chunk = context_map.get(e.evidence_id)
            if chunk and chunk.created_at:
                # Ensure chunk.created_at is timezone aware or handle naive
                created_at = chunk.created_at
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                    
                age_days = (now - created_at).days
                ages.append(age_days)
        
        if not ages:
            return 1.0 # No dates, assume neutral OR penalty? Let's assume neutral 1.0 to avoid noise.
            
        # Calculate average age
        avg_age = sum(ages) / len(ages)
        
        # Heuristic: Decay confidence as data gets older
        # e.g., < 30 days = 1.0
        # > 365 days = 0.5
        # Exponential decay? or simple thresholds?
        # Simple threshold for explainability.
        
        if avg_age < 30:
            return 1.0
        elif avg_age < 90:
            return 0.95
        elif avg_age < 180:
            return 0.9
        elif avg_age < 365:
            return 0.8
        else:
            return 0.7
