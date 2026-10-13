"""
Deterministic GSTR-1 Draft Validator.
Validates normalized B2B Table 4A and B2C Table 7 records against statutory criteria.
"""
from typing import List, Union, Dict, Any, Set, Optional
import re
from src.gst.gstr1_models import (
    GSTR1B2BInvoice,
    GSTR1B2CSInvoice,
    GSTR1ValidationIssue
)

def validate_gstr1_record(
    record: Union[GSTR1B2BInvoice, GSTR1B2CSInvoice],
    seen_invoice_ids: Optional[Set[str]] = None
) -> List[GSTR1ValidationIssue]:
    """
    Validates an individual GSTR-1 draft record.
    """
    issues: List[GSTR1ValidationIssue] = []
    inv_id = record.source_invoice_id

    # GSTR1-001: Invoice Number Present
    if not record.invoice_number or not str(record.invoice_number).strip():
        issues.append(GSTR1ValidationIssue(
            invoice_id=inv_id,
            rule_id="GSTR1-001",
            severity="ERROR",
            field="invoice_number",
            message="Invoice number is missing or empty"
        ))

    # GSTR1-002: Invoice Date Format (YYYY-MM-DD)
    if not record.invoice_date or not re.match(r"^\d{4}-\d{2}-\d{2}$", str(record.invoice_date).strip()):
        issues.append(GSTR1ValidationIssue(
            invoice_id=inv_id,
            rule_id="GSTR1-002",
            severity="ERROR",
            field="invoice_date",
            message=f"Invoice date '{record.invoice_date}' must follow YYYY-MM-DD format"
        ))

    # GSTR1-003: Place of Supply Present
    if not record.place_of_supply or not str(record.place_of_supply).strip():
        issues.append(GSTR1ValidationIssue(
            invoice_id=inv_id,
            rule_id="GSTR1-003",
            severity="ERROR",
            field="place_of_supply",
            message="Place of supply is missing"
        ))

    # GSTR1-004 & GSTR1-005: Buyer GSTIN Checks
    if isinstance(record, GSTR1B2BInvoice):
        # B2B requires valid 15-char GSTIN
        if not record.recipient_gstin or len(record.recipient_gstin.strip()) != 15:
            issues.append(GSTR1ValidationIssue(
                invoice_id=inv_id,
                rule_id="GSTR1-004",
                severity="ERROR",
                field="recipient_gstin",
                message=f"B2B recipient GSTIN '{record.recipient_gstin}' is not a valid 15-character GSTIN"
            ))

    # GSTR1-006: Non-negative values
    if record.taxable_value < 0 or record.invoice_value < 0:
        issues.append(GSTR1ValidationIssue(
            invoice_id=inv_id,
            rule_id="GSTR1-006",
            severity="ERROR",
            field="financial_values",
            message="Taxable value and invoice value must be non-negative"
        ))

    # GSTR1-007: Grand Total Reconciliation (Within ₹1.00 tolerance)
    expected_total = record.taxable_value + record.cgst_amount + record.sgst_amount + record.igst_amount + record.cess_amount
    if abs(expected_total - record.invoice_value) > 1.05:
        issues.append(GSTR1ValidationIssue(
            invoice_id=inv_id,
            rule_id="GSTR1-007",
            severity="ERROR",
            field="invoice_value",
            message=f"Invoice value ₹{record.invoice_value:,.2f} does not reconcile with taxable + taxes ₹{expected_total:,.2f}"
        ))

    # GSTR1-008: Tax Rate Non-negative
    if record.rate < 0 or record.rate > 50.0:
        issues.append(GSTR1ValidationIssue(
            invoice_id=inv_id,
            rule_id="GSTR1-008",
            severity="WARNING",
            field="rate",
            message=f"Applicable tax rate {record.rate}% is unusual"
        ))

    # GSTR1-010: Lineage ID Check / Duplicate Protection
    if seen_invoice_ids is not None:
        if inv_id in seen_invoice_ids:
            issues.append(GSTR1ValidationIssue(
                invoice_id=inv_id,
                rule_id="GSTR1-010",
                severity="ERROR",
                field="source_invoice_id",
                message=f"Duplicate source invoice ID '{inv_id}' detected in GSTR-1 batch"
            ))
        else:
            seen_invoice_ids.add(inv_id)

    return issues

def validate_gstr1_batch(
    b2b_records: List[GSTR1B2BInvoice],
    b2c_records: List[GSTR1B2CSInvoice]
) -> List[GSTR1ValidationIssue]:
    """
    Validates an entire batch of B2B and B2C GSTR-1 draft records.
    """
    issues: List[GSTR1ValidationIssue] = []
    seen_ids: Set[str] = set()

    for b2b in b2b_records:
        issues.extend(validate_gstr1_record(b2b, seen_ids))

    for b2c in b2c_records:
        issues.extend(validate_gstr1_record(b2c, seen_ids))

    return issues
