from pydantic import BaseModel, Field
from typing import List, Optional, Any
from enum import Enum

class ImpactLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class DecisionEvidence(BaseModel):
    """
    Represents a piece of evidence used to support a decision.
    
    Why structured?
    Decisions must be traceable back to specific facts or documents (grounding).
    Unstructured reasoning allows for hallucinations; explicit evidence references enforce truthfulness.
    """
    evidence_id: str = Field(..., description="Unique ID of the memory chunk or document")
    content_snippet: str = Field(..., description="Brief snippet of the evidence for display")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="How strongly this supports the decision")
    source_uri: Optional[str] = Field(None, description="Original source location (file, url, etc.)")

class DecisionScore(BaseModel):
    """
    Quantitative evaluation of a proposal.
    
    Why structured?
    Textual 'good/bad' evaluations are hard to compare programmatically.
    Structured scores allow for sorting, filtering, and threshold-based automation.
    """
    value: float = Field(..., ge=0.0, le=1.0, description="Normalized score (0-1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model's certainty in this score")
    dimension: str = Field(default="overall", description="Aspect being scored (e.g., 'cost', 'feasibility')")
    breakdown: Optional[dict] = Field(None, description="Detailed components of the score")

class DecisionProposal(BaseModel):
    """
    A specific course of action proposed by the system.
    
    Why structured?
    Proposals need to be actionable objects that can be accepted, rejected, or modified by a human.
    """
    id: str = Field(..., description="Unique identifier for this proposal")
    title: str = Field(..., description="Actionable summary")
    rationale: str = Field(..., description="Why this is being proposed")
    impact: ImpactLevel = Field(..., description="Estimated magnitude of effect")
    scores: List[DecisionScore] = Field(..., description="Evaluations of this proposal")
    supporting_evidence: List[DecisionEvidence] = Field(..., description="Evidence backing this proposal")
    critique: Optional[str] = Field(None, description="Validation notes and risks identified by the Critic Agent")

from datetime import datetime, timezone

class DecisionStatus(str, Enum):
    APPROVED = "approved"     # High confidence, auto-approved
    PENDING = "pending"       # Low confidence, requires human review
    REJECTED = "rejected"     # Rejected by user

class AuditEvent(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When this event occurred")
    agent: str = Field(..., description="Name of the agent (e.g., 'Critic', 'Supervisor')")
    action: str = Field(..., description="Action taken (e.g., 'Critique', 'Ranking')")
    details: str = Field(..., description="Description of the action and its reasoning")

class DecisionResult(BaseModel):
    """
    The final output of a decision process.
    
    Why structured?
    Encapsulates the entire decision context, allowing for audit trails and replayability.
    """
    decision_id: str = Field(..., description="Unique ID for this decision event")
    status: DecisionStatus = Field(..., description="Current governance state")
    context_summary: str = Field(..., description="Summary of the inputs/state")
    proposals: List[DecisionProposal] = Field(..., description="Ranked options")
    selected_proposal_id: Optional[str] = Field(None, description="The recommendation or user selection")
    meta_analysis: str = Field(..., description="High-level reasoning over the options")
    audit_trail: List[AuditEvent] = Field(default_factory=list, description="Log of agent actions for trust and governance")
