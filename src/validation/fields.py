"""
Mandatory and optional field validation rules for Canonical Invoices.
"""
import re
from datetime import datetime
from typing import List, Optional, Tuple

from src.extraction.models import CanonicalInvoice
from src.validation.models import ValidationIssue

DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def validate_mandatory_fields(invoice: CanonicalInvoice) -> List[ValidationIssue]:
    """
    Evaluates required fields on a CanonicalInvoice and returns a list of ValidationIssues.
    
    Rule IDs:
    - FIELD-001: Missing / empty invoice_number (ERROR)
    - FIELD-002: Missing / invalid invoice_date (ERROR)
    - FIELD-003: Missing / empty seller GSTIN (ERROR)
    - FIELD-004: Missing buyer GSTIN (INFO / WARNING depending on B2B vs B2C)
    - FIELD-005: Empty items list (WARNING)
    - FIELD-006: Missing tax_summary taxable_value (ERROR)
    - FIELD-007: Missing grand_total (ERROR)
    """
    issues: List[ValidationIssue] = []
    
    # 1. Invoice Number (FIELD-001)
    if not invoice.invoice_number or not invoice.invoice_number.strip():
        issues.append(ValidationIssue(
            rule_id="FIELD-001",
            severity="ERROR",
            field="invoice_number",
            message="Mandatory field 'invoice_number' is missing or empty",
            expected="Valid invoice string identifier",
            actual=invoice.invoice_number
        ))
        
    # 2. Invoice Date (FIELD-002)
    if not invoice.invoice_date or not invoice.invoice_date.strip():
        issues.append(ValidationIssue(
            rule_id="FIELD-002",
            severity="ERROR",
            field="invoice_date",
            message="Mandatory field 'invoice_date' is missing or empty",
            expected="YYYY-MM-DD",
            actual=invoice.invoice_date
        ))
    else:
        date_str = invoice.invoice_date.strip()
        if not DATE_REGEX.match(date_str):
            issues.append(ValidationIssue(
                rule_id="FIELD-002",
                severity="ERROR",
                field="invoice_date",
                message=f"Invoice date '{date_str}' is not in valid ISO YYYY-MM-DD format",
                expected="YYYY-MM-DD",
                actual=date_str
            ))
        else:
            try:
                parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
                # Basic sanity check: year between 2000 and 2099
                if parsed_date.year < 2000 or parsed_date.year > 2099:
                    issues.append(ValidationIssue(
                        rule_id="FIELD-002",
                        severity="WARNING",
                        field="invoice_date",
                        message=f"Invoice date year {parsed_date.year} is outside normal commercial bounds (2000-2099)",
                        expected="2000 <= Year <= 2099",
                        actual=date_str
                    ))
            except ValueError:
                issues.append(ValidationIssue(
                    rule_id="FIELD-002",
                    severity="ERROR",
                    field="invoice_date",
                    message=f"Invoice date '{date_str}' represents an invalid calendar date",
                    expected="Valid calendar date",
                    actual=date_str
                ))

    # 3. Seller GSTIN (FIELD-003)
    if not invoice.seller or not invoice.seller.gstin or not invoice.seller.gstin.strip():
        issues.append(ValidationIssue(
            rule_id="FIELD-003",
            severity="ERROR",
            field="seller.gstin",
            message="Mandatory seller GSTIN is missing",
            expected="15-character GSTIN",
            actual=invoice.seller.gstin if invoice.seller else None
        ))

    # 4. Buyer GSTIN (FIELD-004) - In B2C this is legitimately null
    if not invoice.buyer or not invoice.buyer.gstin or not invoice.buyer.gstin.strip():
        issues.append(ValidationIssue(
            rule_id="FIELD-004",
            severity="INFO",
            field="buyer.gstin",
            message="Buyer GSTIN is absent (treated as B2C/unregistered buyer transaction)",
            expected="15-character GSTIN for B2B or null for B2C",
            actual=invoice.buyer.gstin if invoice.buyer else None
        ))

    # 5. Items List (FIELD-005)
    if not invoice.items or len(invoice.items) == 0:
        issues.append(ValidationIssue(
            rule_id="FIELD-005",
            severity="WARNING",
            field="items",
            message="Line items list is empty",
            expected="At least 1 item",
            actual=0
        ))

    # 6. Tax Summary Taxable Value (FIELD-006)
    if invoice.tax_summary.taxable_value is None:
        issues.append(ValidationIssue(
            rule_id="FIELD-006",
            severity="ERROR",
            field="tax_summary.taxable_value",
            message="Tax summary taxable value is missing",
            expected="Numeric amount >= 0",
            actual=None
        ))

    # 7. Grand Total (FIELD-007)
    if invoice.grand_total is None:
        issues.append(ValidationIssue(
            rule_id="FIELD-007",
            severity="ERROR",
            field="grand_total",
            message="Mandatory grand total is missing",
            expected="Numeric amount >= 0",
            actual=None
        ))

    return issues
