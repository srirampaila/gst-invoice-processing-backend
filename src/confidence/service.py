"""
Service Orchestrator for Phase 7 Confidence Scoring & Human Review Workflow.
"""
from typing import Optional, Dict, Any, Union
from datetime import datetime, timezone
from pathlib import Path
import json

from src.extraction.models import CanonicalInvoice
from src.validation.models import ValidationResult
from src.confidence.config import ConfidenceConfig
from src.confidence.models import (
    ConfidenceResult,
    ConfidenceTier,
    Recommendation,
    ReviewRecord,
    ReviewStatus,
    ReviewerAction
)
from src.confidence.scoring import (
    evaluate_ocr_quality,
    evaluate_field_completeness,
    evaluate_critical_fields,
    evaluate_gst_validation,
    evaluate_financial_validation,
    evaluate_line_item_quality
)
from src.confidence.rules import evaluate_safety_overrides
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ConfidenceService:
    """Calculates deterministic multi-factor confidence scores and manages human review records."""

    def __init__(self, config: Optional[ConfidenceConfig] = None):
        self.config = config or ConfidenceConfig()

    def evaluate_confidence(
        self,
        invoice: CanonicalInvoice,
        validation_result: Optional[ValidationResult] = None,
        ocr_metadata: Optional[Dict[str, Any]] = None,
        invoice_id: Optional[str] = None
    ) -> ConfidenceResult:
        """
        Executes multi-factor confidence scoring and evaluates safety overrides.
        """
        inv_id = invoice_id or invoice.invoice_number or "UNKNOWN_INVOICE"
        
        # 1. Compute Individual Factors
        f_ocr = evaluate_ocr_quality(invoice, validation_result, ocr_metadata, self.config)
        f_field = evaluate_field_completeness(invoice, validation_result, self.config)
        f_crit = evaluate_critical_fields(invoice, validation_result, self.config)
        f_gst = evaluate_gst_validation(invoice, validation_result, self.config)
        f_fin = evaluate_financial_validation(invoice, validation_result, self.config)
        f_item = evaluate_line_item_quality(invoice, validation_result, self.config)

        factors = {
            "ocr_quality": f_ocr,
            "field_completeness": f_field,
            "critical_fields": f_crit,
            "gst_validation": f_gst,
            "financial_validation": f_fin,
            "line_item_quality": f_item
        }

        # 2. Compute Base Weighted Score
        base_score = sum(f.weighted_score for f in factors.values())

        # 3. Deduct Penalties for Validation Findings
        error_count = validation_result.error_count if validation_result else 0
        warning_count = validation_result.warning_count if validation_result else 0
        
        penalties = (error_count * self.config.error_penalty) + (warning_count * self.config.warning_penalty)
        composite_score = max(0.0, min(100.0, base_score - penalties))
        composite_score = round(composite_score, 2)

        # 4. Evaluate Safety Overrides
        tier, recommendation, requires_review, triggered_rules = evaluate_safety_overrides(
            invoice=invoice,
            validation_result=validation_result,
            calculated_score=composite_score,
            factors=factors,
            config=self.config
        )

        # 5. Formulate Human-Readable Executive Explanation
        explanation_parts = [
            f"Overall score {composite_score}/100 placed in {tier} tier (Recommendation: {recommendation})."
        ]
        if triggered_rules:
            override_descs = [r.description for r in triggered_rules]
            explanation_parts.append(f"Safety Overrides Applied: {'; '.join(override_descs)}.")
        else:
            explanation_parts.append("All deterministic validation rules and critical field checks passed cleanly.")

        explanation = " ".join(explanation_parts)

        logger.info(
            f"Confidence evaluated for {inv_id}: score={composite_score}, tier={tier}, "
            f"rec={recommendation}, overrides={len(triggered_rules)}"
        )

        return ConfidenceResult(
            invoice_id=inv_id,
            overall_score=composite_score,
            confidence_tier=tier,
            recommendation=recommendation,
            requires_review=requires_review,
            factors=factors,
            triggered_rules=triggered_rules,
            explanation=explanation
        )

    def create_review_record(
        self,
        confidence_result: ConfidenceResult,
        validation_result: Optional[ValidationResult] = None
    ) -> ReviewRecord:
        """
        Creates a new immutable ReviewRecord in PENDING_REVIEW state.
        """
        issues_list = []
        if validation_result and validation_result.issues:
            issues_list = [i.model_dump() for i in validation_result.issues]

        return ReviewRecord(
            invoice_id=confidence_result.invoice_id,
            review_status=ReviewStatus.PENDING_REVIEW.value,
            confidence_score=confidence_result.overall_score,
            confidence_tier=confidence_result.confidence_tier,
            recommendation=confidence_result.recommendation,
            requires_review=confidence_result.requires_review,
            reviewer_action=None,
            issues=issues_list,
            created_at=datetime.now(timezone.utc).isoformat(),
            reviewed_at=None,
            notes=None
        )

    def apply_review_action(
        self,
        record: ReviewRecord,
        action: Union[ReviewerAction, str],
        notes: Optional[str] = None
    ) -> ReviewRecord:
        """
        Applies a reviewer action (APPROVE, REJECT, REQUEST_REPROCESSING)
        and returns a newly updated ReviewRecord.
        """
        action_str = action.value if isinstance(action, ReviewerAction) else str(action).upper()
        
        if action_str == ReviewerAction.APPROVE.value:
            new_status = ReviewStatus.APPROVED.value
        elif action_str == ReviewerAction.REJECT.value:
            new_status = ReviewStatus.REJECTED.value
        elif action_str == ReviewerAction.REQUEST_REPROCESSING.value:
            new_status = ReviewStatus.REPROCESS_REQUIRED.value
        else:
            new_status = action_str

        # Create updated record without mutating input
        record_dict = record.model_dump()
        record_dict.update({
            "review_status": new_status,
            "reviewer_action": action_str,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "notes": notes or record.notes
        })

        return ReviewRecord.model_validate(record_dict)

def evaluate_invoice_confidence(
    invoice: CanonicalInvoice,
    validation_result: Optional[ValidationResult] = None,
    ocr_metadata: Optional[Dict[str, Any]] = None,
    config: Optional[ConfidenceConfig] = None,
    invoice_id: Optional[str] = None
) -> ConfidenceResult:
    """Convenience helper to evaluate confidence for an invoice."""
    service = ConfidenceService(config)
    return service.evaluate_confidence(
        invoice=invoice,
        validation_result=validation_result,
        ocr_metadata=ocr_metadata,
        invoice_id=invoice_id
    )
