from pydantic import BaseModel, Field

class DecisionFeedback(BaseModel):
    decision_id: str
    original_query: str
    selected_proposal_title: str
    approved: bool
    feedback_text: str = Field(..., description="User's explanation for their rating/rejection")
    user_id: str = "system_user"
