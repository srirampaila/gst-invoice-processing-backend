"""
Comprehensive unit and integration tests for Phase 6 Deterministic GST & Financial Validation Engine.
"""
import pytest
import json
from decimal import Decimal
from src.extraction.models import (
    CanonicalInvoice,
    Party,
    InvoiceItem,
    TaxSummary,
    ExtractionMetadata
)
from src.validation.config import ValidationConfig
from src.validation.models import (
    ValidationResult,
    ValidationIssue,
    ValidationCheck,
    CalculatedFinancials,
    GSTINValidationResult
)
from src.validation.exceptions import (
    ValidationError,
    GSTINValidationError,
    FinancialReconciliationError,
    MandatoryFieldMissingError
)
from src.validation.gstin import validate_gstin, compute_gstin_checksum, get_state_name
from src.validation.arithmetic import (
    to_decimal,
    is_close,
    calculate_item_taxable_sum,
    calculate_expected_tax,
    calculate_grand_total
)
from src.validation.fields import validate_mandatory_fields
from src.validation.service import ValidationService, validate_canonical_invoice

# -------------------------------------------------------------
# 1-4. GSTIN Validation Tests
# -------------------------------------------------------------

def test_valid_gstin_structure():
    res = validate_gstin("29ABCDE1234F1Z5")
    assert res.valid is True
    assert res.state_code == "29"
    assert res.state_name == "Karnataka"
    assert res.pan == "ABCDE1234F"
    assert res.entity_code == "1"
    assert res.checksum_char == "5"
    assert len(res.errors) == 0

def test_invalid_gstin_length():
    res_short = validate_gstin("29ABCDE1234F1Z")
    assert res_short.valid is False
    assert any("exactly 15 characters" in err for err in res_short.errors)

    res_long = validate_gstin("29ABCDE1234F1Z599")
    assert res_long.valid is False
    assert any("exactly 15 characters" in err for err in res_long.errors)

def test_invalid_gstin_structure_and_state():
    # Invalid state code (98 does not exist)
    res_state = validate_gstin("98ABCDE1234F1Z5")
    assert res_state.valid is False
    assert any("Invalid state code" in err for err in res_state.errors)

    # 14th char not Z
    res_z = validate_gstin("29ABCDE1234F1A5")
    assert res_z.valid is False
    assert any("must be 'Z'" in err for err in res_z.errors)

def test_gstin_checksum_algorithm():
    # Compute checksum for a 14 char string
    calc_sum = compute_gstin_checksum("29ABCDE1234F1Z")
    assert len(calc_sum) == 1
    assert calc_sum == "W"

    # Test strict checksum mode
    res_strict_fail = validate_gstin("29ABCDE1234F1Z5", strict_checksum=True)
    assert res_strict_fail.valid is False
    assert any("checksum mismatch" in err for err in res_strict_fail.errors)

    res_strict_pass = validate_gstin(f"29ABCDE1234F1Z{calc_sum}", strict_checksum=True)
    assert res_strict_pass.valid is True
    assert res_strict_pass.checksum_valid is True

# -------------------------------------------------------------
# 5-16. Financial, Field & Rule Tests
# -------------------------------------------------------------

def test_valid_canonical_invoice():
    invoice = CanonicalInvoice(
        invoice_number="INV-VALID-001",
        invoice_date="2026-08-15",
        seller=Party(name="Seller", gstin="29ABCDE1234F1Z5"),
        buyer=Party(name="Buyer", gstin="27XYZAB5678C1Z2"),
        place_of_supply="29",
        items=[
            InvoiceItem(
                description="Widget",
                quantity=1.0,
                unit_price=1000.0,
                taxable_value=1000.0,
                cgst_rate=9.0,
                cgst_amount=90.0,
                sgst_rate=9.0,
                sgst_amount=90.0
            )
        ],
        tax_summary=TaxSummary(
            taxable_value=1000.0,
            cgst=90.0,
            sgst=90.0,
            igst=0.0
        ),
        grand_total=1180.0
    )
    service = ValidationService()
    res = service.validate_invoice(invoice)
    assert res.status == "PASS"
    assert res.passed is True
    assert res.requires_review is False
    assert res.error_count == 0

def test_missing_required_fields():
    invoice = CanonicalInvoice(
        invoice_number=None,
        invoice_date="invalid-date",
        seller=Party(name="Seller", gstin=None),
        grand_total=None
    )
    issues = validate_mandatory_fields(invoice)
    rule_ids = [i.rule_id for i in issues]
    assert "FIELD-001" in rule_ids
    assert "FIELD-002" in rule_ids
    assert "FIELD-003" in rule_ids
    assert "FIELD-007" in rule_ids

def test_item_taxable_mismatch():
    invoice = CanonicalInvoice(
        invoice_number="INV-002",
        invoice_date="2026-08-15",
        seller=Party(name="Seller", gstin="29ABCDE1234F1Z5"),
        items=[
            InvoiceItem(description="Item 1", quantity=1.0, unit_price=400.0, taxable_value=400.0),
            InvoiceItem(description="Item 2", quantity=1.0, unit_price=400.0, taxable_value=400.0)
        ],
        tax_summary=TaxSummary(taxable_value=1000.0, cgst=90.0, sgst=90.0),
        grand_total=1180.0
    )
    res = validate_canonical_invoice(invoice)
    assert res.status == "FAIL"
    assert any(i.rule_id == "MATH-001" for i in res.issues)

def test_tax_mismatch_cgst_sgst():
    invoice = CanonicalInvoice(
        invoice_number="INV-003",
        invoice_date="2026-08-15",
        seller=Party(name="Seller", gstin="29ABCDE1234F1Z5"),
        items=[
            InvoiceItem(
                description="Item 1",
                quantity=1.0,
                unit_price=1000.0,
                taxable_value=1000.0,
                cgst_rate=9.0,
                cgst_amount=90.0,
                sgst_rate=9.0,
                sgst_amount=90.0
            )
        ],
        tax_summary=TaxSummary(taxable_value=1000.0, cgst=50.0, sgst=90.0),  # Wrong CGST
        grand_total=1140.0
    )
    res = validate_canonical_invoice(invoice)
    assert res.status == "FAIL"
    assert any(i.rule_id == "TAX-CGST-001" for i in res.issues)

def test_grand_total_mismatch_and_invalid_math_detection():
    invoice = CanonicalInvoice(
        invoice_number="INV_INV_001",
        invoice_date="2026-08-15",
        seller=Party(name="Seller", gstin="29ABCDE1234F1Z5"),
        items=[InvoiceItem(description="Item 1", quantity=1.0, unit_price=1000.0, taxable_value=1000.0)],
        tax_summary=TaxSummary(taxable_value=1000.0, cgst=90.0, sgst=90.0, igst=0.0),
        grand_total=1680.0  # Mismatched by 500
    )
    res = validate_canonical_invoice(invoice)
    assert res.status == "FAIL"
    assert res.requires_review is True
    assert any(i.rule_id == "TOTAL-001" for i in res.issues)
    assert res.calculated_values.grand_total_discrepancy == 500.0

def test_configurable_tolerance_behavior():
    # 1 INR difference
    invoice = CanonicalInvoice(
        invoice_number="INV-TOL-001",
        invoice_date="2026-08-15",
        seller=Party(name="Seller", gstin="29ABCDE1234F1Z5"),
        items=[InvoiceItem(description="Item 1", taxable_value=1000.0)],
        tax_summary=TaxSummary(taxable_value=1000.0, cgst=90.0, sgst=90.0, igst=0.0),
        grand_total=1180.80  # 80 paise difference
    )
    # Tolerance = 1.00 -> PASS
    config_1 = ValidationConfig(amount_tolerance=Decimal("1.00"))
    res_pass = validate_canonical_invoice(invoice, config=config_1)
    assert res_pass.status == "PASS"

    # Tolerance = 0.50 -> FAIL
    config_strict = ValidationConfig(amount_tolerance=Decimal("0.50"))
    res_fail = validate_canonical_invoice(invoice, config=config_strict)
    assert res_fail.status == "FAIL"
    assert any(i.rule_id == "TOTAL-001" for i in res_fail.issues)

def test_multi_item_reconciliation():
    items = [
        InvoiceItem(description=f"Item {i}", quantity=1.0, unit_price=1000.0, taxable_value=1000.0, cgst_rate=9.0, cgst_amount=90.0, sgst_rate=9.0, sgst_amount=90.0)
        for i in range(5)
    ]
    invoice = CanonicalInvoice(
        invoice_number="INV-MULTI-001",
        invoice_date="2026-08-15",
        seller=Party(name="Seller", gstin="29ABCDE1234F1Z5"),
        items=items,
        tax_summary=TaxSummary(taxable_value=5000.0, cgst=450.0, sgst=450.0, igst=0.0),
        grand_total=5900.0
    )
    res = validate_canonical_invoice(invoice)
    assert res.status == "PASS"
    assert res.calculated_values.calculated_item_taxable_sum == 5000.0
    assert res.calculated_values.calculated_grand_total == 5900.0

def test_b2c_missing_buyer_gstin_acceptance():
    invoice = CanonicalInvoice(
        invoice_number="INV-B2C-001",
        invoice_date="2026-08-15",
        seller=Party(name="Retail Store", gstin="29ABCDE1234F1Z5"),
        buyer=Party(name="Walk-in Consumer", gstin=None),
        items=[InvoiceItem(description="Retail Goods", taxable_value=500.0, cgst_amount=45.0, sgst_amount=45.0)],
        tax_summary=TaxSummary(taxable_value=500.0, cgst=45.0, sgst=45.0),
        grand_total=590.0
    )
    res = validate_canonical_invoice(invoice)
    assert res.status == "PASS"
    assert res.buyer_gstin_result is None or res.buyer_gstin_result.valid is False
    assert not any(i.severity == "ERROR" and "buyer" in i.field for i in res.issues)

def test_decimal_arithmetic_precision():
    # 0.1 + 0.2 in binary float is 0.30000000000000004
    d1 = to_decimal(0.1)
    d2 = to_decimal(0.2)
    assert d1 + d2 == Decimal("0.30")

    # Item sum calculation
    items = [{"taxable_value": 0.1}, {"taxable_value": 0.2}]
    item_sum = calculate_item_taxable_sum(items)
    assert item_sum == Decimal("0.30")

def test_validation_result_serialization():
    invoice = CanonicalInvoice(
        invoice_number="INV-SERIAL-001",
        invoice_date="2026-08-15",
        seller=Party(name="Seller", gstin="29ABCDE1234F1Z5"),
        items=[InvoiceItem(description="Item", taxable_value=100.0, cgst_amount=9.0, sgst_amount=9.0)],
        tax_summary=TaxSummary(taxable_value=100.0, cgst=9.0, sgst=9.0),
        grand_total=118.0
    )
    res = validate_canonical_invoice(invoice)
    res_dict = res.model_dump()
    res_json_str = res.model_dump_json()

    assert res_dict["status"] == "PASS"
    assert "calculated_values" in res_dict
    assert isinstance(json.loads(res_json_str), dict)
