"""
Deterministic Safety Overrides and Hard Guardrails for Phase 7 Confidence Engine.
"""
from typing import List, Tuple, Dict, Any, Optional
from src.extraction.models import CanonicalInvoice
from src.validation.models import ValidationResult
from src.confidence.config import ConfidenceConfig
from src.confidence.models import (
    ConfidenceTier,
    Recommendation,
    TriggeredRule,
    FactorScore
)

def evaluate_safety_overrides(
    invoice: CanonicalInvoice,
    validation_result: Optional[ValidationResult],
    calculated_score: float,
    factors: Dict[str, FactorScore],
    config: ConfidenceConfig
) -> Tuple[str, str, bool, List[TriggeredRule]]:
    """
    Evaluates hard regulatory, arithmetic, and OCR safety rules.
    Returns: (final_tier, final_recommendation, requires_review, triggered_rules)
    """
    triggered_rules: List[TriggeredRule] = []

    # Initial Tier & Recommendation based purely on numerical threshold
    if calculated_score >= config.high_threshold:
        tier = ConfidenceTier.HIGH.value
        recommendation = Recommendation.AUTO_ACCEPT.value
        requires_review = False
    elif calculated_score >= config.medium_threshold:
        tier = ConfidenceTier.MEDIUM.value
        recommendation = Recommendation.REVIEW_REQUIRED.value
        requires_review = True
    else:
        tier = ConfidenceTier.LOW.value
        recommendation = Recommendation.REVIEW_REQUIRED.value
        requires_review = True

    if not config.enforce_safety_overrides:
        return tier, recommendation, requires_review, triggered_rules

    # 1. Check Missing Invoice Number
    if not invoice.invoice_number or not str(invoice.invoice_number).strip():
        tier = ConfidenceTier.LOW.value
        requires_review = True
        if recommendation == Recommendation.AUTO_ACCEPT.value:
            recommendation = Recommendation.REVIEW_REQUIRED.value
        triggered_rules.append(TriggeredRule(
            rule_id="OVERRIDE_MISSING_INVOICE_NUMBER",
            category="OVERRIDE",
            severity="CRITICAL",
            description="Mandatory invoice number is missing; confidence capped at LOW",
            override_effect="Capped tier to LOW, set requires_review=True"
        ))

    # 2. Check Missing Invoice Date
    if not invoice.invoice_date or not str(invoice.invoice_date).strip():
        tier = ConfidenceTier.LOW.value
        requires_review = True
        if recommendation == Recommendation.AUTO_ACCEPT.value:
            recommendation = Recommendation.REVIEW_REQUIRED.value
        triggered_rules.append(TriggeredRule(
            rule_id="OVERRIDE_MISSING_INVOICE_DATE",
            category="OVERRIDE",
            severity="CRITICAL",
            description="Mandatory invoice date is missing; confidence capped at LOW",
            override_effect="Capped tier to LOW, set requires_review=True"
        ))

    # 3. Check Seller GSTIN Missing or Invalid
    seller_gstin_invalid = False
    if not invoice.seller or not invoice.seller.gstin or not str(invoice.seller.gstin).strip():
        seller_gstin_invalid = True
    elif validation_result and validation_result.seller_gstin_result and not validation_result.seller_gstin_result.valid:
        seller_gstin_invalid = True

    if seller_gstin_invalid:
        tier = ConfidenceTier.LOW.value
        requires_review = True
        if recommendation == Recommendation.AUTO_ACCEPT.value:
            recommendation = Recommendation.REVIEW_REQUIRED.value
        triggered_rules.append(TriggeredRule(
            rule_id="OVERRIDE_INVALID_SELLER_GSTIN",
            category="OVERRIDE",
            severity="CRITICAL",
            description="Seller GSTIN is missing or structurally invalid; confidence capped at LOW",
            override_effect="Capped tier to LOW, set requires_review=True"
        ))

    # 4. Check Deterministic Validation Failures & Errors
    if validation_result:
        # Check specific financial rules
        total_001_failed = any(i.rule_id == "TOTAL-001" for i in validation_result.issues)
        math_001_failed = any(i.rule_id == "MATH-001" for i in validation_result.issues)
        has_tax_error = any(i.rule_id.startswith("TAX-") for i in validation_result.issues)
        financial_error = total_001_failed or math_001_failed or has_tax_error

        if financial_error or validation_result.error_count > 0:
            requires_review = True
            if tier == ConfidenceTier.HIGH.value:
                tier = ConfidenceTier.MEDIUM.value
            if recommendation == Recommendation.AUTO_ACCEPT.value:
                recommendation = Recommendation.REVIEW_REQUIRED.value

            triggered_rules.append(TriggeredRule(
                rule_id="OVERRIDE_FINANCIAL_ERROR",
                category="OVERRIDE",
                severity="CRITICAL",
                description=f"Validation engine reported {validation_result.error_count} error(s); cannot AUTO_ACCEPT or receive HIGH confidence",
                override_effect="Capped tier from HIGH to MEDIUM/LOW, set requires_review=True, recommendation=REVIEW_REQUIRED"
            ))

        if total_001_failed:
            tier = ConfidenceTier.LOW.value
            triggered_rules.append(TriggeredRule(
                rule_id="OVERRIDE_TOTAL_001_FAILED",
                category="OVERRIDE",
                severity="CRITICAL",
                description="Grand total failed deterministic arithmetic reconciliation (TOTAL-001)",
                override_effect="Capped tier to LOW, forced manual review"
            ))

        # Check for multiple critical validation errors
        if validation_result.error_count >= config.max_errors_for_review:
            tier = ConfidenceTier.LOW.value
            recommendation = Recommendation.REJECT_OR_REPROCESS.value
            requires_review = True
            triggered_rules.append(TriggeredRule(
                rule_id="OVERRIDE_MULTIPLE_CRITICAL_ERRORS",
                category="OVERRIDE",
                severity="CRITICAL",
                description=f"Multiple critical errors detected ({validation_result.error_count} >= {config.max_errors_for_review}); recommendation escalated to REJECT_OR_REPROCESS",
                override_effect="Set tier=LOW, recommendation=REJECT_OR_REPROCESS"
            ))

        # Check for validation warnings
        if validation_result.warning_count > 0 or validation_result.requires_review:
            requires_review = True
            if recommendation == Recommendation.AUTO_ACCEPT.value and validation_result.warning_count > 0:
                recommendation = Recommendation.REVIEW_REQUIRED.value
                triggered_rules.append(TriggeredRule(
                    rule_id="OVERRIDE_WARNINGS_DETECTED",
                    category="RISK",
                    severity="MEDIUM",
                    description=f"Validation engine reported {validation_result.warning_count} warning(s)",
                    override_effect="Set requires_review=True, recommendation=REVIEW_REQUIRED"
                ))

    # 5. Check OCR Quality Guardrails
    ocr_factor = factors.get("ocr_quality")
    if ocr_factor:
        ocr_score = ocr_factor.score
        if ocr_score < config.min_ocr_for_auto_accept:
            if recommendation == Recommendation.AUTO_ACCEPT.value:
                recommendation = Recommendation.REVIEW_REQUIRED.value
                requires_review = True
            triggered_rules.append(TriggeredRule(
                rule_id="OVERRIDE_DEGRADED_OCR",
                category="RISK",
                severity="HIGH",
                description=f"OCR quality ({ocr_score:.1f}%) is below minimum threshold for auto-acceptance ({config.min_ocr_for_auto_accept}%)",
                override_effect="Forced human review"
            ))

        if ocr_score < config.min_ocr_for_medium:
            tier = ConfidenceTier.LOW.value
            recommendation = Recommendation.REJECT_OR_REPROCESS.value
            requires_review = True
            triggered_rules.append(TriggeredRule(
                rule_id="OVERRIDE_SEVERE_OCR_FAILURE",
                category="OVERRIDE",
                severity="CRITICAL",
                description=f"OCR recognition quality ({ocr_score:.1f}%) is critically degraded",
                override_effect="Set tier=LOW, recommendation=REJECT_OR_REPROCESS"
            ))

    return tier, recommendation, requires_review, triggered_rules
