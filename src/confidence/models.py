"""
Pydantic Data Models for Confidence Scoring & Human Review Workflow.
"""
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

class ConfidenceTier(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class Recommendation(str, Enum):
    AUTO_ACCEPT = "AUTO_ACCEPT"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    REJECT_OR_REPROCESS = "REJECT_OR_REPROCESS"

class ReviewStatus(str, Enum):
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REPROCESS_REQUIRED = "REPROCESS_REQUIRED"

class ReviewerAction(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REQUEST_REPROCESSING = "REQUEST_REPROCESSING"

class FactorScore(BaseModel):
    """Detailed score breakdown for an individual confidence dimension."""
    model_config = ConfigDict(extra="ignore")
    name: str = Field(description="Name of the factor (e.g. ocr_quality, financial_validation)")
    score: float = Field(ge=0.0, le=100.0, description="Factor score from 0 to 100")
    weight: float = Field(ge=0.0, le=1.0, description="Configured weight for this factor")
    weighted_score: float = Field(ge=0.0, le=100.0, description="Score multiplied by weight")
    reason: str = Field(description="Human-readable explanation of the score")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic parameters")

class TriggeredRule(BaseModel):
    """Details of a critical safety override or risk rule triggered during evaluation."""
    model_config = ConfigDict(extra="ignore")
    rule_id: str = Field(description="Identifier of the rule (e.g. OVERRIDE_FINANCIAL_ERROR)")
    category: str = Field(description="OVERRIDE, RISK, or PENALTY")
    severity: str = Field(description="CRITICAL, HIGH, MEDIUM, or LOW")
    description: str = Field(description="Explanation of why the rule fired")
    override_effect: str = Field(description="Action taken (e.g. Capped tier to LOW, Forced REVIEW_REQUIRED)")

class ConfidenceResult(BaseModel):
    """Comprehensive multi-factor confidence evaluation output."""
    model_config = ConfigDict(extra="ignore")
    invoice_id: str
    overall_score: float = Field(ge=0.0, le=100.0, description="Composite confidence score (0-100)")
    confidence_tier: str = Field(description="HIGH, MEDIUM, or LOW")
    recommendation: str = Field(description="AUTO_ACCEPT, REVIEW_REQUIRED, or REJECT_OR_REPROCESS")
    requires_review: bool = Field(description="True if human intervention is required before downstream use")
    factors: Dict[str, FactorScore] = Field(default_factory=dict, description="Individual factor scores and weights")
    triggered_rules: List[TriggeredRule] = Field(default_factory=list, description="List of safety overrides triggered")
    explanation: str = Field(description="Executive summary of confidence determination")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ReviewRecord(BaseModel):
    """Immutable audit trail of human review actions for an invoice."""
    model_config = ConfigDict(extra="ignore")
    invoice_id: str
    review_status: str = Field(default=ReviewStatus.PENDING_REVIEW.value, description="PENDING_REVIEW, APPROVED, REJECTED, REPROCESS_REQUIRED")
    confidence_score: float = Field(ge=0.0, le=100.0)
    confidence_tier: str
    recommendation: str
    requires_review: bool
    reviewer_action: Optional[str] = None
    issues: List[Dict[str, Any]] = Field(default_factory=list, description="Validation issues present at time of review")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reviewed_at: Optional[str] = None
    notes: Optional[str] = None

class ConfidenceSummary(BaseModel):
    """Batch-level aggregate metrics for confidence scoring."""
    model_config = ConfigDict(extra="ignore")
    total_invoices: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    auto_accept_count: int = 0
    review_required_count: int = 0
    reprocess_reject_count: int = 0
    average_confidence_score: float = 0.0
    min_confidence_score: float = 0.0
    max_confidence_score: float = 0.0
    validation_error_correlation: Dict[str, Any] = Field(default_factory=dict)
