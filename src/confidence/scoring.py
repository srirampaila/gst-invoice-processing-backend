"""
Deterministic Multi-Factor Scoring Functions for Phase 7 Confidence Engine.
"""
from typing import Dict, Any, Optional
from decimal import Decimal
from src.extraction.models import CanonicalInvoice
from src.validation.models import ValidationResult
from src.confidence.config import ConfidenceConfig
from src.confidence.models import FactorScore

def evaluate_ocr_quality(
    invoice: CanonicalInvoice,
    validation_result: Optional[ValidationResult],
    ocr_metadata: Optional[Dict[str, Any]],
    config: ConfidenceConfig
) -> FactorScore:
    """
    Evaluates raw OCR recognition confidence and character fidelity.
    """
    weight = config.ocr_weight
    
    # Priority: direct OCR metadata -> ExtractionMetadata confidence
    ocr_conf = None
    if ocr_metadata and "average_confidence" in ocr_metadata:
        ocr_conf = float(ocr_metadata["average_confidence"])
    elif invoice.extraction_metadata and invoice.extraction_metadata.confidence is not None:
        ocr_conf = float(invoice.extraction_metadata.confidence * 100.0)
    else:
        ocr_conf = 80.0  # Reasonable fallback

    # Check for noisy / degraded indicators
    details = {"raw_ocr_confidence": round(ocr_conf, 2)}
    score = max(0.0, min(100.0, ocr_conf))

    if ocr_conf >= 85.0:
        reason = f"High-fidelity OCR recognition certainty ({ocr_conf:.1f}%)"
    elif ocr_conf >= 65.0:
        reason = f"Acceptable OCR recognition certainty ({ocr_conf:.1f}%) with minor character degradation"
    else:
        reason = f"Low OCR recognition certainty ({ocr_conf:.1f}%), elevated risk of character misread"

    weighted_score = round(score * weight, 2)
    return FactorScore(
        name="ocr_quality",
        score=round(score, 2),
        weight=weight,
        weighted_score=weighted_score,
        reason=reason,
        details=details
    )

def evaluate_field_completeness(
    invoice: CanonicalInvoice,
    validation_result: Optional[ValidationResult],
    config: ConfidenceConfig
) -> FactorScore:
    """
    Evaluates presence of contextual and non-critical standard invoice fields.
    """
    weight = config.field_completeness_weight
    total_points = 0.0
    evaluated_fields = {}

    # Seller Name & Address
    if invoice.seller and invoice.seller.name:
        total_points += 15.0
        evaluated_fields["seller_name"] = True
    else:
        evaluated_fields["seller_name"] = False

    if invoice.seller and invoice.seller.address:
        total_points += 10.0
        evaluated_fields["seller_address"] = True
    else:
        evaluated_fields["seller_address"] = False

    # Buyer Name & Address
    if invoice.buyer and invoice.buyer.name:
        total_points += 15.0
        evaluated_fields["buyer_name"] = True
    else:
        evaluated_fields["buyer_name"] = False

    if invoice.buyer and invoice.buyer.address:
        total_points += 10.0
        evaluated_fields["buyer_address"] = True
    else:
        evaluated_fields["buyer_address"] = False

    # Place of Supply
    if invoice.place_of_supply:
        total_points += 15.0
        evaluated_fields["place_of_supply"] = True
    else:
        evaluated_fields["place_of_supply"] = False

    # Tax Summary presence
    if invoice.tax_summary and invoice.tax_summary.taxable_value is not None:
        total_points += 15.0
        evaluated_fields["tax_summary"] = True
    else:
        evaluated_fields["tax_summary"] = False

    # Currency
    if invoice.currency:
        total_points += 10.0
        evaluated_fields["currency"] = True
    else:
        evaluated_fields["currency"] = False

    # Non-empty items
    if invoice.items and len(invoice.items) > 0:
        total_points += 10.0
        evaluated_fields["has_items"] = True
    else:
        evaluated_fields["has_items"] = False

    score = min(100.0, total_points)
    populated_count = sum(1 for v in evaluated_fields.values() if v)
    reason = f"Populated {populated_count} of 8 standard contextual invoice fields"
    weighted_score = round(score * weight, 2)

    return FactorScore(
        name="field_completeness",
        score=round(score, 2),
        weight=weight,
        weighted_score=weighted_score,
        reason=reason,
        details=evaluated_fields
    )

def evaluate_critical_fields(
    invoice: CanonicalInvoice,
    validation_result: Optional[ValidationResult],
    config: ConfidenceConfig
) -> FactorScore:
    """
    Evaluates presence of mandatory core legal fields (Invoice Number, Date, Seller GSTIN, Grand Total).
    """
    weight = config.critical_fields_weight
    critical_status = {}
    points = 0.0

    # Invoice Number (25 pts)
    if invoice.invoice_number and str(invoice.invoice_number).strip():
        points += 25.0
        critical_status["invoice_number"] = "PRESENT"
    else:
        critical_status["invoice_number"] = "MISSING"

    # Invoice Date (25 pts)
    if invoice.invoice_date and str(invoice.invoice_date).strip():
        points += 25.0
        critical_status["invoice_date"] = "PRESENT"
    else:
        critical_status["invoice_date"] = "MISSING"

    # Seller GSTIN (25 pts)
    if invoice.seller and invoice.seller.gstin and str(invoice.seller.gstin).strip():
        points += 25.0
        critical_status["seller_gstin"] = "PRESENT"
    else:
        critical_status["seller_gstin"] = "MISSING"

    # Grand Total (25 pts)
    if invoice.grand_total is not None and invoice.grand_total > 0:
        points += 25.0
        critical_status["grand_total"] = "PRESENT"
    else:
        critical_status["grand_total"] = "MISSING"

    score = points
    missing_fields = [k for k, v in critical_status.items() if v == "MISSING"]
    if not missing_fields:
        reason = "All 4 critical mandatory invoice fields are present"
    else:
        reason = f"Missing critical mandatory fields: {', '.join(missing_fields)}"

    weighted_score = round(score * weight, 2)
    return FactorScore(
        name="critical_fields",
        score=round(score, 2),
        weight=weight,
        weighted_score=weighted_score,
        reason=reason,
        details=critical_status
    )

def evaluate_gst_validation(
    invoice: CanonicalInvoice,
    validation_result: Optional[ValidationResult],
    config: ConfidenceConfig
) -> FactorScore:
    """
    Evaluates GSTIN format integrity, state consistency, and checksum from deterministic validation.
    """
    weight = config.gst_validation_weight
    
    if not validation_result:
        # Default neutral score if validation result not available
        return FactorScore(
            name="gst_validation",
            score=50.0,
            weight=weight,
            weighted_score=round(50.0 * weight, 2),
            reason="Validation result not provided",
            details={}
        )

    score = 100.0
    details = {}
    reasons = []

    # Seller GSTIN checks
    seller_res = validation_result.seller_gstin_result
    if seller_res:
        details["seller_gstin_valid"] = seller_res.valid
        details["seller_gstin_checksum_valid"] = seller_res.checksum_valid
        if not seller_res.valid:
            score -= 50.0
            reasons.append("Seller GSTIN structurally invalid")
        elif not seller_res.checksum_valid:
            score -= 10.0
            reasons.append("Seller GSTIN checksum variance")
    else:
        score -= 50.0
        reasons.append("Seller GSTIN missing or unverified")

    # Buyer GSTIN checks
    buyer_res = validation_result.buyer_gstin_result
    if buyer_res and invoice.buyer and invoice.buyer.gstin:
        details["buyer_gstin_valid"] = buyer_res.valid
        if not buyer_res.valid:
            score -= 25.0
            reasons.append("Buyer GSTIN structurally invalid")
    else:
        details["buyer_gstin"] = "B2C / Unregistered"

    # Specific GST rule failures from validation issues
    gst_issues = [i for i in validation_result.issues if i.rule_id.startswith("GSTIN") or i.rule_id.startswith("STATE")]
    if any(i.severity == "ERROR" for i in gst_issues):
        score = min(score, 40.0)

    score = max(0.0, min(100.0, score))
    if not reasons:
        reason = "Seller GSTIN and state configuration verified successfully"
    else:
        reason = "; ".join(reasons)

    weighted_score = round(score * weight, 2)
    return FactorScore(
        name="gst_validation",
        score=round(score, 2),
        weight=weight,
        weighted_score=weighted_score,
        reason=reason,
        details=details
    )

def evaluate_financial_validation(
    invoice: CanonicalInvoice,
    validation_result: Optional[ValidationResult],
    config: ConfidenceConfig
) -> FactorScore:
    """
    Evaluates independent mathematical reconciliation of taxable values, taxes, and grand total.
    """
    weight = config.financial_validation_weight

    if not validation_result:
        return FactorScore(
            name="financial_validation",
            score=50.0,
            weight=weight,
            weighted_score=round(50.0 * weight, 2),
            reason="Validation result not provided",
            details={}
        )

    score = 100.0
    details = {
        "error_count": validation_result.error_count,
        "warning_count": validation_result.warning_count,
        "grand_total_discrepancy": validation_result.calculated_values.grand_total_discrepancy
    }
    reasons = []

    # Check for arithmetic issues
    arithmetic_issues = [
        i for i in validation_result.issues 
        if i.rule_id in ("TOTAL-001", "MATH-001", "TAX-CGST-001", "TAX-SGST-001", "TAX-IGST-001")
    ]

    has_total_001 = any(i.rule_id == "TOTAL-001" for i in arithmetic_issues)
    has_math_001 = any(i.rule_id == "MATH-001" for i in arithmetic_issues)
    tax_errors = [i for i in arithmetic_issues if i.rule_id.startswith("TAX-")]

    if has_total_001:
        disc = validation_result.calculated_values.grand_total_discrepancy or 0.0
        score -= 70.0
        reasons.append(f"Grand total reconciliation failed (TOTAL-001 discrepancy: ₹{disc:,.2f})")

    if has_math_001:
        score -= 30.0
        reasons.append("Item taxable sum does not match summary taxable value (MATH-001)")

    if tax_errors:
        score -= (len(tax_errors) * 20.0)
        reasons.append(f"Tax rate calculation error ({', '.join(i.rule_id for i in tax_errors)})")

    score = max(0.0, min(100.0, score))
    if not reasons:
        reason = "Deterministic financial arithmetic and grand total reconciled perfectly"
    else:
        reason = "; ".join(reasons)

    weighted_score = round(score * weight, 2)
    return FactorScore(
        name="financial_validation",
        score=round(score, 2),
        weight=weight,
        weighted_score=weighted_score,
        reason=reason,
        details=details
    )

def evaluate_line_item_quality(
    invoice: CanonicalInvoice,
    validation_result: Optional[ValidationResult],
    config: ConfidenceConfig
) -> FactorScore:
    """
    Evaluates itemized consistency, descriptions, unit rates, and individual calculations.
    """
    weight = config.line_item_weight

    if not invoice.items or len(invoice.items) == 0:
        return FactorScore(
            name="line_item_quality",
            score=30.0,
            weight=weight,
            weighted_score=round(30.0 * weight, 2),
            reason="No itemized lines present (summary-level extraction only)",
            details={"item_count": 0}
        )

    item_scores = []
    for item in invoice.items:
        item_pts = 0.0
        # Description
        if item.description and len(str(item.description).strip()) > 1:
            item_pts += 25.0
        # Quantity & Unit Price
        if item.quantity is not None and item.quantity > 0:
            item_pts += 25.0
        if item.unit_price is not None and item.unit_price >= 0:
            item_pts += 25.0
        # Taxable Value
        if item.taxable_value is not None and item.taxable_value >= 0:
            item_pts += 25.0
        item_scores.append(item_pts)

    avg_item_score = sum(item_scores) / len(item_scores)
    score = max(0.0, min(100.0, avg_item_score))
    reason = f"Reconciled {len(invoice.items)} line item(s) with consistent attributes"
    
    weighted_score = round(score * weight, 2)
    return FactorScore(
        name="line_item_quality",
        score=round(score, 2),
        weight=weight,
        weighted_score=weighted_score,
        reason=reason,
        details={"item_count": len(invoice.items), "item_scores": item_scores}
    )
