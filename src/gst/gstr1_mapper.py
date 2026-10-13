"""
Deterministic Mapper for GSTR-1 Draft Preparation.
Transforms validated CanonicalInvoice instances into GSTR-1 B2B / B2C records.
"""
from typing import Optional, Tuple, Dict, Any, Union
from datetime import datetime
import re

from src.extraction.models import CanonicalInvoice
from src.validation.models import ValidationResult
from src.confidence.models import (
    ConfidenceResult,
    ReviewRecord,
    ConfidenceTier,
    Recommendation,
    ReviewStatus
)
from src.gst.gstr1_models import (
    GSTR1B2BInvoice,
    GSTR1B2CSInvoice,
    GSTR1ExclusionRecord
)
from src.gst.exceptions import GSTR1MappingError

def resolve_applicable_tax_rate(invoice: CanonicalInvoice) -> float:
    """
    Determines the applicable GST tax rate percentage from line items or tax totals.
    """
    # 1. Check if line items specify tax rates
    if invoice.items and len(invoice.items) > 0:
        rates = []
        for item in invoice.items:
            item_rate = 0.0
            if item.cgst_rate is not None and item.sgst_rate is not None:
                item_rate = float(item.cgst_rate) + float(item.sgst_rate)
            elif item.igst_rate is not None:
                item_rate = float(item.igst_rate)
            if item_rate > 0:
                rates.append(item_rate)
        if rates:
            return round(rates[0], 2)

    # 2. Derive rate from tax amounts and taxable value
    taxable = invoice.tax_summary.taxable_value if invoice.tax_summary else None
    if taxable and taxable > 0:
        cgst = invoice.tax_summary.cgst or 0.0
        sgst = invoice.tax_summary.sgst or 0.0
        igst = invoice.tax_summary.igst or 0.0
        total_taxes = cgst + sgst + igst
        if total_taxes > 0:
            derived_rate = (total_taxes / taxable) * 100.0
            # Common GST slab matching
            for standard_slab in [0.0, 5.0, 12.0, 18.0, 28.0]:
                if abs(derived_rate - standard_slab) <= 0.5:
                    return standard_slab
            return round(derived_rate, 2)

    return 18.0  # Default standard GST rate if indeterminate

def normalize_invoice_date(date_str: Optional[str]) -> str:
    """
    Normalizes invoice date into YYYY-MM-DD representation.
    """
    if not date_str or not str(date_str).strip():
        return "1970-01-01"
    
    clean_str = str(date_str).strip()
    
    # Standard YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", clean_str):
        return clean_str

    # DD-MM-YYYY or DD/MM/YYYY
    match_dmy = re.match(r"^(\d{2})[-/](\d{2})[-/](\d{4})$", clean_str)
    if match_dmy:
        d, m, y = match_dmy.groups()
        return f"{y}-{m}-{d}"

    # Try datetime parse fallback
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d %b %Y", "%d %B %Y"):
        try:
            dt = datetime.strptime(clean_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    return clean_str

def check_invoice_eligibility(
    invoice: CanonicalInvoice,
    validation_result: Optional[ValidationResult],
    confidence_result: Optional[ConfidenceResult],
    review_record: Optional[ReviewRecord],
    invoice_id: str
) -> Tuple[bool, Optional[GSTR1ExclusionRecord]]:
    """
    Evaluates whether an invoice satisfies all Phase 8 eligibility criteria.
    Returns: (is_eligible, exclusion_record_if_ineligible)
    """
    reasons = []
    
    # 1. Review status gate
    is_approved = review_record and review_record.review_status == ReviewStatus.APPROVED.value
    is_auto_accept = (
        confidence_result and 
        confidence_result.confidence_tier == ConfidenceTier.HIGH.value and 
        confidence_result.recommendation == Recommendation.AUTO_ACCEPT.value
    )

    if review_record and review_record.review_status == ReviewStatus.REJECTED.value:
        reasons.append("Invoice manually REJECTED by reviewer")
    elif review_record and review_record.review_status == ReviewStatus.REPROCESS_REQUIRED.value:
        reasons.append("Invoice flagged as REPROCESS_REQUIRED")
    elif not is_approved and not is_auto_accept:
        reasons.append(
            f"Confidence tier is {confidence_result.confidence_tier if confidence_result else 'UNKNOWN'} "
            f"(Recommendation: {confidence_result.recommendation if confidence_result else 'UNKNOWN'}), "
            f"requires explicit human approval"
        )

    # 2. Validation error gate
    if validation_result:
        critical_issues = [
            i for i in validation_result.issues 
            if i.rule_id in ("TOTAL-001", "MATH-001", "FIELD-001", "FIELD-002", "FIELD-003", "GSTIN-001") or i.severity == "ERROR"
        ]
        if critical_issues:
            issue_summaries = [f"{i.rule_id} ({i.message})" for i in critical_issues]
            reasons.append(f"Critical validation failures: {'; '.join(issue_summaries)}")

    # 3. Core mandatory fields
    if not invoice.invoice_number:
        reasons.append("Missing mandatory invoice number")
    if not invoice.invoice_date:
        reasons.append("Missing mandatory invoice date")
    if not invoice.place_of_supply:
        reasons.append("Missing place of supply")

    if reasons:
        exclusion = GSTR1ExclusionRecord(
            invoice_id=invoice_id,
            reason=" | ".join(reasons),
            confidence_score=confidence_result.overall_score if confidence_result else 0.0,
            confidence_tier=confidence_result.confidence_tier if confidence_result else "LOW",
            recommendation=confidence_result.recommendation if confidence_result else "REJECT_OR_REPROCESS",
            review_status=review_record.review_status if review_record else "PENDING_REVIEW",
            validation_status=validation_result.status if validation_result else "FAIL",
            error_count=validation_result.error_count if validation_result else 1,
            warning_count=validation_result.warning_count if validation_result else 0,
            triggered_rules=[r.rule_id for r in confidence_result.triggered_rules] if confidence_result else []
        )
        return False, exclusion

    return True, None

def map_canonical_to_gstr1(
    invoice: CanonicalInvoice,
    validation_result: Optional[ValidationResult] = None,
    confidence_result: Optional[ConfidenceResult] = None,
    review_record: Optional[ReviewRecord] = None,
    invoice_id: Optional[str] = None
) -> Tuple[Optional[Union[GSTR1B2BInvoice, GSTR1B2CSInvoice]], Optional[GSTR1ExclusionRecord]]:
    """
    Transforms a CanonicalInvoice into a validated GSTR-1 record (B2B Table 4A or B2C Table 7),
    or produces an exclusion record if ineligible.
    """
    inv_id = invoice_id or invoice.invoice_number or "UNKNOWN_INVOICE"

    # Evaluate eligibility gate
    is_eligible, exclusion = check_invoice_eligibility(
        invoice=invoice,
        validation_result=validation_result,
        confidence_result=confidence_result,
        review_record=review_record,
        invoice_id=inv_id
    )

    if not is_eligible:
        return None, exclusion

    # Extract financial values with explicit None checks
    if invoice.tax_summary and invoice.tax_summary.taxable_value is not None:
        taxable_val = float(invoice.tax_summary.taxable_value)
    elif validation_result and validation_result.calculated_values and validation_result.calculated_values.calculated_item_taxable_sum is not None:
        taxable_val = float(validation_result.calculated_values.calculated_item_taxable_sum)
    else:
        taxable_val = 0.0

    if invoice.tax_summary and invoice.tax_summary.cgst is not None:
        cgst_val = float(invoice.tax_summary.cgst)
    elif validation_result and validation_result.calculated_values and validation_result.calculated_values.calculated_cgst is not None:
        cgst_val = float(validation_result.calculated_values.calculated_cgst)
    else:
        cgst_val = 0.0

    if invoice.tax_summary and invoice.tax_summary.sgst is not None:
        sgst_val = float(invoice.tax_summary.sgst)
    elif validation_result and validation_result.calculated_values and validation_result.calculated_values.calculated_sgst is not None:
        sgst_val = float(validation_result.calculated_values.calculated_sgst)
    else:
        sgst_val = 0.0

    if invoice.tax_summary and invoice.tax_summary.igst is not None:
        igst_val = float(invoice.tax_summary.igst)
    elif validation_result and validation_result.calculated_values and validation_result.calculated_values.calculated_igst is not None:
        igst_val = float(validation_result.calculated_values.calculated_igst)
    else:
        igst_val = 0.0

    if invoice.grand_total is not None:
        inv_total = float(invoice.grand_total)
    elif validation_result and validation_result.calculated_values and validation_result.calculated_values.calculated_grand_total is not None:
        inv_total = float(validation_result.calculated_values.calculated_grand_total)
    else:
        inv_total = taxable_val + cgst_val + sgst_val + igst_val
    
    tax_rate = resolve_applicable_tax_rate(invoice)
    inv_date = normalize_invoice_date(invoice.invoice_date)
    pos = str(invoice.place_of_supply or "").strip()
    inv_num = str(invoice.invoice_number or "").strip()
    val_status = validation_result.status if validation_result else "PASS"
    conf_score = confidence_result.overall_score if confidence_result else 100.0

    # Determine B2B vs B2C
    has_buyer_gstin = bool(invoice.buyer and invoice.buyer.gstin and str(invoice.buyer.gstin).strip())

    if has_buyer_gstin:
        # Map to Table 4A (B2B)
        b2b_rec = GSTR1B2BInvoice(
            source_invoice_id=inv_id,
            recipient_gstin=str(invoice.buyer.gstin).strip().upper(),
            recipient_name=invoice.buyer.name or None,
            invoice_number=inv_num,
            invoice_date=inv_date,
            invoice_value=round(inv_total, 2),
            place_of_supply=pos,
            reverse_charge="N",
            invoice_type="Regular",
            taxable_value=round(taxable_val, 2),
            rate=tax_rate,
            cgst_amount=round(cgst_val, 2),
            sgst_amount=round(sgst_val, 2),
            igst_amount=round(igst_val, 2),
            cess_amount=0.0,
            source_validation_status=val_status,
            source_confidence_score=conf_score
        )
        return b2b_rec, None
    else:
        # Map to Table 7 (B2C Small)
        b2c_rec = GSTR1B2CSInvoice(
            source_invoice_id=inv_id,
            place_of_supply=pos,
            invoice_number=inv_num,
            invoice_date=inv_date,
            invoice_value=round(inv_total, 2),
            taxable_value=round(taxable_val, 2),
            rate=tax_rate,
            cgst_amount=round(cgst_val, 2),
            sgst_amount=round(sgst_val, 2),
            igst_amount=round(igst_val, 2),
            cess_amount=0.0,
            source_validation_status=val_status,
            source_confidence_score=conf_score
        )
        return b2c_rec, None
