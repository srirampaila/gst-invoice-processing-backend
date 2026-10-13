"""
Configuration models and defaults for Phase 7 Confidence Scoring Engine.
"""
from pydantic import BaseModel, Field, model_validator
from src.confidence.exceptions import ConfidenceConfigError

class ConfidenceConfig(BaseModel):
    """Configuration settings, factor weights, and thresholds for confidence scoring."""
    
    # Factor Weights (Must sum to 1.0)
    ocr_weight: float = Field(default=0.15, ge=0.0, le=1.0, description="Weight for OCR recognition certainty")
    field_completeness_weight: float = Field(default=0.15, ge=0.0, le=1.0, description="Weight for general field completeness")
    critical_fields_weight: float = Field(default=0.20, ge=0.0, le=1.0, description="Weight for mandatory legal invoice fields")
    gst_validation_weight: float = Field(default=0.20, ge=0.0, le=1.0, description="Weight for GSTIN structure and state checks")
    financial_validation_weight: float = Field(default=0.20, ge=0.0, le=1.0, description="Weight for deterministic arithmetic reconciliation")
    line_item_weight: float = Field(default=0.10, ge=0.0, le=1.0, description="Weight for line item consistency")

    # Score Thresholds
    high_threshold: float = Field(default=85.0, ge=0.0, le=100.0, description="Minimum score required for HIGH confidence tier")
    medium_threshold: float = Field(default=60.0, ge=0.0, le=100.0, description="Minimum score required for MEDIUM confidence tier")
    
    # OCR Guardrail Thresholds
    min_ocr_for_auto_accept: float = Field(default=65.0, ge=0.0, le=100.0, description="Minimum OCR confidence required for AUTO_ACCEPT")
    min_ocr_for_medium: float = Field(default=40.0, ge=0.0, le=100.0, description="OCR confidence below this forces LOW tier")

    # Error & Warning Penalties (subtracted from composite score)
    error_penalty: float = Field(default=15.0, ge=0.0, description="Score penalty deduction per validation error")
    warning_penalty: float = Field(default=5.0, ge=0.0, description="Score penalty deduction per validation warning")

    # Safety Overrides Configuration
    enforce_safety_overrides: bool = Field(default=True, description="Enforce non-negotiable regulatory and arithmetic overrides")
    max_errors_for_review: int = Field(default=2, description="If error count exceeds this, recommend REJECT_OR_REPROCESS rather than REVIEW")

    @model_validator(mode="after")
    def validate_weights(self) -> "ConfidenceConfig":
        total_weight = (
            self.ocr_weight +
            self.field_completeness_weight +
            self.critical_fields_weight +
            self.gst_validation_weight +
            self.financial_validation_weight +
            self.line_item_weight
        )
        if abs(total_weight - 1.0) > 1e-4:
            raise ConfidenceConfigError(f"Factor weights must sum to 1.0, currently sum to {total_weight:.4f}")
        
        if self.high_threshold <= self.medium_threshold:
            raise ConfidenceConfigError(
                f"high_threshold ({self.high_threshold}) must be strictly greater than medium_threshold ({self.medium_threshold})"
            )
        return self
