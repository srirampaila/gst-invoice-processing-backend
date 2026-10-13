"""
Rule catalog and evaluation engine for GST & Financial Validation.
"""
from decimal import Decimal
from typing import List, Tuple, Optional, Any

from src.extraction.models import CanonicalInvoice
from src.validation.config import ValidationConfig
from src.validation.models import (
    ValidationIssue,
    ValidationCheck,
    CalculatedFinancials,
    GSTINValidationResult
)
from src.validation.gstin import validate_gstin
from src.validation.fields import validate_mandatory_fields
from src.validation.arithmetic import (
    to_decimal,
    is_close,
    calculate_item_taxable_sum,
    calculate_expected_tax,
    calculate_grand_total
)

def evaluate_all_rules(
    invoice: CanonicalInvoice,
    config: ValidationConfig
) -> Tuple[List[ValidationIssue], List[ValidationCheck], CalculatedFinancials, GSTINValidationResult, Optional[GSTINValidationResult]]:
    """
    Executes all deterministic validation checks against a CanonicalInvoice.
    
    Returns:
        (issues, checks, calculated_financials, seller_gstin_res, buyer_gstin_res)
    """
    issues: List[ValidationIssue] = []
    checks: List[ValidationCheck] = []
    
    # -------------------------------------------------------------
    # 1. Mandatory Field Checks (FIELD-001 to FIELD-007)
    # -------------------------------------------------------------
    field_issues = validate_mandatory_fields(invoice)
    issues.extend(field_issues)
    has_field_errors = any(i.severity == "ERROR" for i in field_issues)
    checks.append(ValidationCheck(
        check_name="Mandatory Fields",
        category="FIELD",
        status="FAIL" if has_field_errors else "PASS",
        message="All mandatory fields present" if not has_field_errors else "One or more mandatory fields missing"
    ))

    # -------------------------------------------------------------
    # 2. GSTIN Validation (GSTIN-001, GSTIN-002, GSTIN-003)
    # -------------------------------------------------------------
    seller_gstin_str = invoice.seller.gstin if invoice.seller else None
    seller_gstin_res = validate_gstin(seller_gstin_str, strict_checksum=config.strict_gstin_checksum)
    
    if not seller_gstin_res.valid:
        for err in seller_gstin_res.errors:
            issues.append(ValidationIssue(
                rule_id="GSTIN-001",
                severity="ERROR",
                field="seller.gstin",
                message=f"Seller GSTIN invalid: {err}",
                expected="Valid 15-character GSTIN",
                actual=seller_gstin_str
            ))
    for warn in seller_gstin_res.warnings:
        issues.append(ValidationIssue(
            rule_id="GSTIN-003",
            severity="INFO" if not config.strict_gstin_checksum else "WARNING",
            field="seller.gstin",
            message=f"Seller GSTIN note: {warn}",
            expected="Valid Checksum",
            actual=seller_gstin_str
        ))
        
    checks.append(ValidationCheck(
        check_name="Seller GSTIN Validation",
        category="GSTIN",
        status="PASS" if seller_gstin_res.valid else "FAIL",
        message=f"Seller GSTIN ({seller_gstin_res.state_name or 'Unknown State'}) valid" if seller_gstin_res.valid else "Seller GSTIN validation failed"
    ))

    # Buyer GSTIN Validation (if present)
    buyer_gstin_str = invoice.buyer.gstin if invoice.buyer else None
    buyer_gstin_res: Optional[GSTINValidationResult] = None
    if buyer_gstin_str:
        buyer_gstin_res = validate_gstin(buyer_gstin_str, strict_checksum=config.strict_gstin_checksum)
        if not buyer_gstin_res.valid:
            for err in buyer_gstin_res.errors:
                issues.append(ValidationIssue(
                    rule_id="GSTIN-002",
                    severity="ERROR",
                    field="buyer.gstin",
                    message=f"Buyer GSTIN invalid: {err}",
                    expected="Valid 15-character GSTIN",
                    actual=buyer_gstin_str
                ))
        for warn in buyer_gstin_res.warnings:
            issues.append(ValidationIssue(
                rule_id="GSTIN-003",
                severity="INFO" if not config.strict_gstin_checksum else "WARNING",
                field="buyer.gstin",
                message=f"Buyer GSTIN note: {warn}",
                expected="Valid Checksum",
                actual=buyer_gstin_str
            ))
        checks.append(ValidationCheck(
            check_name="Buyer GSTIN Validation",
            category="GSTIN",
            status="PASS" if buyer_gstin_res.valid else "FAIL",
            message="Buyer GSTIN valid" if buyer_gstin_res.valid else "Buyer GSTIN validation failed"
        ))
    else:
        checks.append(ValidationCheck(
            check_name="Buyer GSTIN Validation",
            category="GSTIN",
            status="PASS",
            message="B2C / Unregistered Buyer (GSTIN not required)"
        ))

    # -------------------------------------------------------------
    # 3. State Code & Tax Type Consistency (STATE-001, STATE-002, STATE-003)
    # -------------------------------------------------------------
    seller_state = seller_gstin_res.state_code if seller_gstin_res.valid else None
    pos = invoice.place_of_supply.strip() if invoice.place_of_supply else None
    
    if seller_state and pos:
        is_intra_state = (seller_state == pos)
        ext_cgst = to_decimal(invoice.tax_summary.cgst) or Decimal("0.00")
        ext_sgst = to_decimal(invoice.tax_summary.sgst) or Decimal("0.00")
        ext_igst = to_decimal(invoice.tax_summary.igst) or Decimal("0.00")

        if is_intra_state:
            # Intra-state: CGST and SGST apply, IGST must be 0
            if ext_igst > config.amount_tolerance:
                issues.append(ValidationIssue(
                    rule_id="STATE-002",
                    severity="WARNING",
                    field="tax_summary.igst",
                    message=f"Intra-state supply (State {seller_state}) should not have IGST amount > 0",
                    expected=0.0,
                    actual=float(ext_igst)
                ))
        else:
            # Inter-state: IGST applies, CGST & SGST must be 0
            if (ext_cgst > config.amount_tolerance) or (ext_sgst > config.amount_tolerance):
                issues.append(ValidationIssue(
                    rule_id="STATE-003",
                    severity="INFO",
                    field="tax_summary.cgst_sgst",
                    message=f"Inter-state supply (Seller: {seller_state}, POS: {pos}) with intra-state CGST/SGST breakdown",
                    expected="IGST applicable",
                    actual=f"CGST={float(ext_cgst)}, SGST={float(ext_sgst)}"
                ))
                
        checks.append(ValidationCheck(
            check_name="State & Tax Type Consistency",
            category="CROSS_FIELD",
            status="PASS",
            message=f"{'Intra-state' if is_intra_state else 'Inter-state'} supply consistency verified"
        ))
    else:
        checks.append(ValidationCheck(
            check_name="State & Tax Type Consistency",
            category="CROSS_FIELD",
            status="SKIPPED",
            message="Insufficient state code / place of supply data"
        ))

    # -------------------------------------------------------------
    # 4. Financial Calculations & Reconciliation
    # -------------------------------------------------------------
    calc_item_taxable_sum = calculate_item_taxable_sum(invoice.items) if invoice.items else None
    ext_taxable = to_decimal(invoice.tax_summary.taxable_value)
    
    # Extract taxes
    ext_cgst = to_decimal(invoice.tax_summary.cgst)
    ext_sgst = to_decimal(invoice.tax_summary.sgst)
    ext_igst = to_decimal(invoice.tax_summary.igst)
    ext_grand_total = to_decimal(invoice.grand_total)

    # Item-level tax sums fallback when summary amounts are missing
    item_cgst_sum = None
    item_sgst_sum = None
    item_igst_sum = None
    if invoice.items and len(invoice.items) > 0:
        c_tot = Decimal("0.00")
        s_tot = Decimal("0.00")
        i_tot = Decimal("0.00")
        has_item_taxes = False
        for it in invoice.items:
            c = to_decimal(it.cgst_amount)
            s = to_decimal(it.sgst_amount)
            i = to_decimal(it.igst_amount)
            if c is not None:
                c_tot += c
                has_item_taxes = True
            if s is not None:
                s_tot += s
                has_item_taxes = True
            if i is not None:
                i_tot += i
                has_item_taxes = True
        if has_item_taxes:
            item_cgst_sum = c_tot
            item_sgst_sum = s_tot
            item_igst_sum = i_tot

    effective_cgst = ext_cgst if ext_cgst is not None else item_cgst_sum
    effective_sgst = ext_sgst if ext_sgst is not None else item_sgst_sum
    effective_igst = ext_igst if ext_igst is not None else item_igst_sum

    # 4a. MATH-001: Item taxable sum vs Summary Taxable Value
    taxable_disc = None
    if calc_item_taxable_sum is not None and ext_taxable is not None:
        taxable_disc = abs(calc_item_taxable_sum - ext_taxable)
        if not is_close(calc_item_taxable_sum, ext_taxable, config.amount_tolerance):
            issues.append(ValidationIssue(
                rule_id="MATH-001",
                severity="ERROR",
                field="tax_summary.taxable_value",
                message=f"Sum of item taxable values ({float(calc_item_taxable_sum):.2f}) does not match summary taxable value ({float(ext_taxable):.2f})",
                expected=float(calc_item_taxable_sum),
                actual=float(ext_taxable)
            ))
            checks.append(ValidationCheck(
                check_name="Item Taxable Reconciliation",
                category="ARITHMETIC",
                status="FAIL",
                message=f"Discrepancy of ₹{float(taxable_disc):.2f} detected"
            ))
        else:
            checks.append(ValidationCheck(
                check_name="Item Taxable Reconciliation",
                category="ARITHMETIC",
                status="PASS",
                message="Line item taxable sum reconciles with summary"
            ))
    else:
        checks.append(ValidationCheck(
            check_name="Item Taxable Reconciliation",
            category="ARITHMETIC",
            status="SKIPPED",
            message="Missing item or summary taxable values"
        ))

    # 4b. Tax Rate Verification (TAX-CGST-001, TAX-SGST-001, TAX-IGST-001)
    cgst_disc = None
    sgst_disc = None
    igst_disc = None
    
    item_cgst_rate = None
    item_sgst_rate = None
    item_igst_rate = None
    if invoice.items and len(invoice.items) > 0:
        first_item = invoice.items[0]
        if first_item.cgst_rate is not None:
            item_cgst_rate = to_decimal(first_item.cgst_rate)
        if first_item.sgst_rate is not None:
            item_sgst_rate = to_decimal(first_item.sgst_rate)
        if first_item.igst_rate is not None:
            item_igst_rate = to_decimal(first_item.igst_rate)

    calc_cgst = None
    if ext_taxable is not None and item_cgst_rate is not None:
        calc_cgst = calculate_expected_tax(ext_taxable, item_cgst_rate)
        if effective_cgst is not None and calc_cgst is not None:
            cgst_disc = abs(calc_cgst - effective_cgst)
            if not is_close(calc_cgst, effective_cgst, config.amount_tolerance):
                issues.append(ValidationIssue(
                    rule_id="TAX-CGST-001",
                    severity="ERROR",
                    field="tax_summary.cgst",
                    message=f"CGST calculation mismatch: Expected ₹{float(calc_cgst):.2f} at {float(item_cgst_rate)}%, got ₹{float(effective_cgst):.2f}",
                    expected=float(calc_cgst),
                    actual=float(effective_cgst)
                ))

    calc_sgst = None
    if ext_taxable is not None and item_sgst_rate is not None:
        calc_sgst = calculate_expected_tax(ext_taxable, item_sgst_rate)
        if effective_sgst is not None and calc_sgst is not None:
            sgst_disc = abs(calc_sgst - effective_sgst)
            if not is_close(calc_sgst, effective_sgst, config.amount_tolerance):
                issues.append(ValidationIssue(
                    rule_id="TAX-SGST-001",
                    severity="ERROR",
                    field="tax_summary.sgst",
                    message=f"SGST calculation mismatch: Expected ₹{float(calc_sgst):.2f} at {float(item_sgst_rate)}%, got ₹{float(effective_sgst):.2f}",
                    expected=float(calc_sgst),
                    actual=float(effective_sgst)
                ))

    calc_igst = None
    if ext_taxable is not None and item_igst_rate is not None and item_igst_rate > 0:
        calc_igst = calculate_expected_tax(ext_taxable, item_igst_rate)
        if effective_igst is not None and calc_igst is not None:
            igst_disc = abs(calc_igst - effective_igst)
            if not is_close(calc_igst, effective_igst, config.amount_tolerance):
                issues.append(ValidationIssue(
                    rule_id="TAX-IGST-001",
                    severity="ERROR",
                    field="tax_summary.igst",
                    message=f"IGST calculation mismatch: Expected ₹{float(calc_igst):.2f} at {float(item_igst_rate)}%, got ₹{float(effective_igst):.2f}",
                    expected=float(calc_igst),
                    actual=float(effective_igst)
                ))

    # 4c. TOTAL-001: Grand Total Reconciliation
    # Expected Grand Total = Taxable Value + CGST + SGST + IGST
    eff_c = effective_cgst or Decimal("0.00")
    eff_s = effective_sgst or Decimal("0.00")
    eff_i = effective_igst or Decimal("0.00")
    calc_tax_sum = (eff_c + eff_s + eff_i)
    calc_grand_total = calculate_grand_total(ext_taxable, eff_c, eff_s, eff_i) if ext_taxable is not None else None
    
    gt_disc = None
    if calc_grand_total is not None and ext_grand_total is not None:
        gt_disc = abs(calc_grand_total - ext_grand_total)
        if not is_close(calc_grand_total, ext_grand_total, config.amount_tolerance):
            issues.append(ValidationIssue(
                rule_id="TOTAL-001",
                severity="ERROR",
                field="grand_total",
                message=f"Grand total does not reconcile with taxable + taxes. Expected ₹{float(calc_grand_total):.2f}, got ₹{float(ext_grand_total):.2f} (Discrepancy: ₹{float(gt_disc):.2f})",
                expected=float(calc_grand_total),
                actual=float(ext_grand_total)
            ))
            checks.append(ValidationCheck(
                check_name="Grand Total Reconciliation",
                category="ARITHMETIC",
                status="FAIL",
                message=f"Grand total mismatch of ₹{float(gt_disc):.2f}"
            ))
        else:
            checks.append(ValidationCheck(
                check_name="Grand Total Reconciliation",
                category="ARITHMETIC",
                status="PASS",
                message="Grand total matches Taxable + Taxes within tolerance"
            ))
    else:
        checks.append(ValidationCheck(
            check_name="Grand Total Reconciliation",
            category="ARITHMETIC",
            status="SKIPPED",
            message="Missing taxable value or grand total for reconciliation"
        ))

    # Package independent calculations
    calculated_financials = CalculatedFinancials(
        calculated_item_taxable_sum=float(calc_item_taxable_sum) if calc_item_taxable_sum is not None else None,
        calculated_cgst=float(calc_cgst) if calc_cgst is not None else (float(effective_cgst) if effective_cgst is not None else None),
        calculated_sgst=float(calc_sgst) if calc_sgst is not None else (float(effective_sgst) if effective_sgst is not None else None),
        calculated_igst=float(calc_igst) if calc_igst is not None else (float(effective_igst) if effective_igst is not None else None),
        calculated_tax_sum=float(calc_tax_sum) if calc_tax_sum is not None else None,
        calculated_grand_total=float(calc_grand_total) if calc_grand_total is not None else None,
        taxable_discrepancy=float(taxable_disc) if taxable_disc is not None else None,
        cgst_discrepancy=float(cgst_disc) if cgst_disc is not None else None,
        sgst_discrepancy=float(sgst_disc) if sgst_disc is not None else None,
        igst_discrepancy=float(igst_disc) if igst_disc is not None else None,
        grand_total_discrepancy=float(gt_disc) if gt_disc is not None else None
    )

    return issues, checks, calculated_financials, seller_gstin_res, buyer_gstin_res
