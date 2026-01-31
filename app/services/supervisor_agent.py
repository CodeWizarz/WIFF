from typing import List
import uuid
from app.domain.decisions import DecisionProposal, DecisionResult

class SupervisorAgent:
    """
    Arbitrator that finalizes the decision process.
    
    Responsibilities:
    - Review the Critic's findings.
    - Rank proposals based on risk-adjusted confidence.
    - Flag high-stakes/low-confidence items for human review.
    - Produce the final DecisionResult.
    
    Why arbitration is necessary:
    - Individual agents (Builder, Critic) have narrow views. The Builder wants to solve; the Critic wants to find faults.
    - A Supervisor balances these opposing forces to find a pragmatic path forward.
    
    Real enterprise workflows:
    - This mirrors a 'Committee Review' or 'Manager Sign-off'. 
    - In an enterprise, raw analysis (Synthesis) is vetted by risk/compliance (Critic), 
      and a leader (Supervisor) makes the final call based on strategic alignment and risk appetite.
    """

    def arbitrate(
        self,
        query: str,
        proposals: List[DecisionProposal],
        context_summary: str
    ) -> DecisionResult:
        """
        Consolidate synthesis and critique into a final result.
        """
        
        if not proposals:
            return self._create_empty_result(query)

        # 1. Rank Decisions
        # Prioritize high confidence (adjusted by critic) and check impact.
        # Score = Confidence * (1.0 if Impact != High else 1.1) (Prefer high impact if confident?)
        # For simplicity, strict ranking by Overall Confidence.
        
        ranked_proposals = sorted(
            proposals,
            key=lambda p: self._get_confidence(p),
            reverse=True
        )
        
        selected_proposal = ranked_proposals[0]
        
        # 2. Flag for Human Approval
        # Logic: If Impact is HIGH and Confidence < 0.8, requires approval.
        # We can add a flag to the result or analysis text.
        
        selected_confidence = self._get_confidence(selected_proposal)
        needs_approval = selected_proposal.impact == "high" and selected_confidence < 0.8
        
        # 3. Formulate Meta-Analysis
        meta_analysis = self._generate_meta_analysis(selected_proposal, ranked_proposals, needs_approval)

        return DecisionResult(
            decision_id=str(uuid.uuid4()),
            context_summary=context_summary,
            proposals=ranked_proposals,
            selected_proposal_id=selected_proposal.id,
            meta_analysis=meta_analysis
        )

    def _get_confidence(self, proposal: DecisionProposal) -> float:
        return next((s.confidence for s in proposal.scores if s.dimension == 'overall'), 0.0)

    def _generate_meta_analysis(
        self, 
        selected: DecisionProposal, 
        all_proposals: List[DecisionProposal],
        needs_approval: bool
    ) -> str:
        count = len(all_proposals)
        confidence = self._get_confidence(selected)
        
        analysis = (
            f"Evaluated {count} potential options. "
            f"Selected '{selected.title}' as the optimal path with {confidence:.0%} confidence."
        )
        
        if selected.critique:
            analysis += f" Note: The Critic raised valid points regarding: {selected.critique[:50]}..."
            
        if needs_approval:
            analysis += (
                "\n\n⚠️ HUMAN APPROVAL REQUIRED: "
                "This decision has HIGH impact but lower confidence (<80%). "
                "Review evidence manually before proceeding."
            )
            
        return analysis

    def _create_empty_result(self, query: str) -> DecisionResult:
        return DecisionResult(
            decision_id=str(uuid.uuid4()),
            context_summary=f"No viable options found for: {query}",
            proposals=[],
            selected_proposal_id=None,
            meta_analysis="Insufficient information to generate proposals."
        )
