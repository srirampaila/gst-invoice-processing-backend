"""
Comprehensive Unit and Integration Tests for Phase 8 GSTR-1 Draft Generation & Export.
"""
import pytest
import copy
import os
import tempfile
from pathlib import Path
from decimal import Decimal

from src.extraction.models import (
    CanonicalInvoice,
    Party,
    InvoiceItem,
    TaxSummary,
    ExtractionMetadata
)
from src.validation.models import (
    ValidationResult,
    ValidationIssue,
    CalculatedFinancials,
    GSTINValidationResult
)
from src.confidence.models import (
    ConfidenceResult,
    ReviewRecord,
    ConfidenceTier,
    Recommendation,
    ReviewStatus,
    ReviewerAction
)
from src.gst.gstr1_models import (
    GSTR1B2BInvoice,
    GSTR1B2CSInvoice,
    GSTR1ExclusionRecord,
    GSTR1Draft
)
from src.gst.gstr1_mapper import (
    resolve_applicable_tax_rate,
    normalize_invoice_date,
    check_invoice_eligibility,
    map_canonical_to_gstr1
)
from src.gst.gstr1_validator import (
    validate_gstr1_record,
    validate_gstr1_batch
)
from src.gst.gstr1_generator import GSTR1DraftGenerator
from src.export.gstr1_csv import (
    export_b2b_csv,
    export_b2c_csv,
    export_exclusions_csv,
    export_gstr1_combined_csv
)
from src.export.gstr1_excel import export_gstr1_excel

@pytest.fixture
def clean_b2b_invoice() -> CanonicalInvoice:
    return CanonicalInvoice(
        invoice_number="INV-2026-B2B-001",
        invoice_date="2026-04-10",
        seller=Party(
            name="Supplier Tech Pvt Ltd",
            gstin="27AABCU9603R1ZM",
            address="123 Industrial Area, Pune, Maharashtra"
        ),
        buyer=Party(
            name="Client Corp Ltd",
            gstin="27AAACB1234A1Z5",
            address="456 IT Park, Mumbai, Maharashtra"
        ),
        place_of_supply="27-Maharashtra",
        items=[
            InvoiceItem(
                description="Cloud Hosting Services",
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
        grand_total=1180.0,
        currency="INR",
        extraction_metadata=ExtractionMetadata(
            source_file="INV_CLEAN_001.png",
            extraction_method="mock",
            confidence=0.95
        )
    )

@pytest.fixture
def clean_b2c_invoice() -> CanonicalInvoice:
    return CanonicalInvoice(
        invoice_number="INV-2026-B2C-002",
        invoice_date="2026-04-12",
        seller=Party(
            name="Retail Hub Pvt Ltd",
            gstin="27AABCU9603R1ZM",
            address="123 Market Road, Pune, Maharashtra"
        ),
        buyer=Party(
            name="Consumer Individual",
            gstin=None,
            address="Pune, Maharashtra"
        ),
        place_of_supply="27-Maharashtra",
        items=[
            InvoiceItem(
                description="Consumer Electronics",
                quantity=1.0,
                unit_price=2000.0,
                taxable_value=2000.0,
                cgst_rate=9.0,
                cgst_amount=180.0,
                sgst_rate=9.0,
                sgst_amount=180.0
            )
        ],
        tax_summary=TaxSummary(
            taxable_value=2000.0,
            cgst=180.0,
            sgst=180.0,
            igst=0.0
        ),
        grand_total=2360.0,
        currency="INR"
    )

@pytest.fixture
def valid_val_res() -> ValidationResult:
    return ValidationResult(
        invoice_id="TEST-001",
        status="PASS",
        passed=True,
        requires_review=False,
        error_count=0,
        warning_count=0,
        issues=[],
        calculated_values=CalculatedFinancials(
            calculated_item_taxable_sum=1000.0,
            calculated_cgst=90.0,
            calculated_sgst=90.0,
            calculated_igst=0.0,
            calculated_tax_sum=180.0,
            calculated_grand_total=1180.0,
            grand_total_discrepancy=0.0
        )
    )

@pytest.fixture
def high_conf_res() -> ConfidenceResult:
    return ConfidenceResult(
        invoice_id="TEST-001",
        overall_score=96.0,
        confidence_tier=ConfidenceTier.HIGH.value,
        recommendation=Recommendation.AUTO_ACCEPT.value,
        requires_review=False,
        explanation="High confidence clean invoice"
    )

# 1. B2B classification
def test_b2b_classification(clean_b2b_invoice, valid_val_res, high_conf_res):
    record, exc = map_canonical_to_gstr1(clean_b2b_invoice, valid_val_res, high_conf_res, invoice_id="INV-001")
    assert exc is None
    assert isinstance(record, GSTR1B2BInvoice)
    assert record.recipient_gstin == "27AAACB1234A1Z5"

# 2. B2C classification
def test_b2c_classification(clean_b2c_invoice, valid_val_res, high_conf_res):
    val_res = copy.deepcopy(valid_val_res)
    val_res.calculated_values.calculated_grand_total = 2360.0
    record, exc = map_canonical_to_gstr1(clean_b2c_invoice, val_res, high_conf_res, invoice_id="INV-002")
    assert exc is None
    assert isinstance(record, GSTR1B2CSInvoice)

# 3. B2B GSTIN mapping
def test_b2b_gstin_mapping(clean_b2b_invoice, valid_val_res, high_conf_res):
    record, exc = map_canonical_to_gstr1(clean_b2b_invoice, valid_val_res, high_conf_res, invoice_id="INV-001")
    assert record.recipient_gstin == "27AAACB1234A1Z5"

# 4. B2C buyer GSTIN absence
def test_b2c_buyer_gstin_absence(clean_b2c_invoice, valid_val_res, high_conf_res):
    record, exc = map_canonical_to_gstr1(clean_b2c_invoice, valid_val_res, high_conf_res, invoice_id="INV-002")
    assert isinstance(record, GSTR1B2CSInvoice)
    assert not hasattr(record, "recipient_gstin")

# 5. Invoice number preservation
def test_invoice_number_preservation(clean_b2b_invoice, valid_val_res, high_conf_res):
    record, exc = map_canonical_to_gstr1(clean_b2b_invoice, valid_val_res, high_conf_res, invoice_id="INV-001")
    assert record.invoice_number == "INV-2026-B2B-001"

# 6. Invoice date mapping (YYYY-MM-DD)
def test_invoice_date_mapping(clean_b2b_invoice, valid_val_res, high_conf_res):
    inv = copy.deepcopy(clean_b2b_invoice)
    inv.invoice_date = "15/04/2026"
    record, exc = map_canonical_to_gstr1(inv, valid_val_res, high_conf_res, invoice_id="INV-001")
    assert record.invoice_date == "2026-04-15"

# 7. Taxable value mapping
def test_taxable_value_mapping(clean_b2b_invoice, valid_val_res, high_conf_res):
    record, exc = map_canonical_to_gstr1(clean_b2b_invoice, valid_val_res, high_conf_res, invoice_id="INV-001")
    assert record.taxable_value == 1000.0

# 8. CGST mapping
def test_cgst_mapping(clean_b2b_invoice, valid_val_res, high_conf_res):
    record, exc = map_canonical_to_gstr1(clean_b2b_invoice, valid_val_res, high_conf_res, invoice_id="INV-001")
    assert record.cgst_amount == 90.0

# 9. SGST mapping
def test_sgst_mapping(clean_b2b_invoice, valid_val_res, high_conf_res):
    record, exc = map_canonical_to_gstr1(clean_b2b_invoice, valid_val_res, high_conf_res, invoice_id="INV-001")
    assert record.sgst_amount == 90.0

# 10. IGST mapping
def test_igst_mapping(clean_b2b_invoice, valid_val_res, high_conf_res):
    inv = copy.deepcopy(clean_b2b_invoice)
    inv.place_of_supply = "29-Karnataka"
    inv.tax_summary.cgst = 0.0
    inv.tax_summary.sgst = 0.0
    inv.tax_summary.igst = 180.0
    record, exc = map_canonical_to_gstr1(inv, valid_val_res, high_conf_res, invoice_id="INV-001")
    assert record.igst_amount == 180.0
    assert record.cgst_amount == 0.0

# 11. Grand total mapping
def test_grand_total_mapping(clean_b2b_invoice, valid_val_res, high_conf_res):
    record, exc = map_canonical_to_gstr1(clean_b2b_invoice, valid_val_res, high_conf_res, invoice_id="INV-001")
    assert record.invoice_value == 1180.0

# 12. Place of supply
def test_place_of_supply(clean_b2b_invoice, valid_val_res, high_conf_res):
    record, exc = map_canonical_to_gstr1(clean_b2b_invoice, valid_val_res, high_conf_res, invoice_id="INV-001")
    assert record.place_of_supply == "27-Maharashtra"

# 13. Invalid invoice exclusion (TOTAL-001 failure)
def test_invalid_invoice_exclusion(clean_b2b_invoice, high_conf_res):
    val_res = ValidationResult(
        invoice_id="INV-INV-001",
        status="FAIL",
        error_count=1,
        issues=[
            ValidationIssue(rule_id="TOTAL-001", severity="ERROR", field="grand_total", message="Discrepancy ₹500")
        ]
    )
    conf_res = ConfidenceResult(
        invoice_id="INV-INV-001",
        overall_score=67.0,
        confidence_tier="LOW",
        recommendation="REVIEW_REQUIRED",
        requires_review=True,
        explanation="TOTAL-001 failed"
    )
    record, exc = map_canonical_to_gstr1(clean_b2b_invoice, val_res, conf_res, invoice_id="INV-INV-001")
    assert record is None
    assert exc is not None
    assert "TOTAL-001" in exc.reason

# 14. Low-confidence exclusion
def test_low_confidence_exclusion(clean_b2b_invoice, valid_val_res):
    conf_res = ConfidenceResult(
        invoice_id="INV-NOISY-001",
        overall_score=35.0,
        confidence_tier="LOW",
        recommendation="REJECT_OR_REPROCESS",
        requires_review=True,
        explanation="Corrupted OCR"
    )
    record, exc = map_canonical_to_gstr1(clean_b2b_invoice, valid_val_res, conf_res, invoice_id="INV-NOISY-001")
    assert record is None
    assert exc is not None

# 15. Approved invoice inclusion
def test_approved_invoice_inclusion(clean_b2b_invoice, valid_val_res):
    conf_res = ConfidenceResult(
        invoice_id="INV-REV-001",
        overall_score=70.0,
        confidence_tier="MEDIUM",
        recommendation="REVIEW_REQUIRED",
        requires_review=True,
        explanation="Medium score needing approval"
    )
    rev_rec = ReviewRecord(
        invoice_id="INV-REV-001",
        review_status=ReviewStatus.APPROVED.value,
        confidence_score=70.0,
        confidence_tier="MEDIUM",
        recommendation="REVIEW_REQUIRED",
        requires_review=True,
        reviewer_action="APPROVE"
    )
    record, exc = map_canonical_to_gstr1(clean_b2b_invoice, valid_val_res, conf_res, rev_rec, invoice_id="INV-REV-001")
    assert record is not None
    assert exc is None

# 16. HIGH + AUTO_ACCEPT inclusion
def test_high_auto_accept_inclusion(clean_b2b_invoice, valid_val_res, high_conf_res):
    record, exc = map_canonical_to_gstr1(clean_b2b_invoice, valid_val_res, high_conf_res, invoice_id="INV-001")
    assert record is not None
    assert exc is None

# 17. Missing POS handling
def test_missing_pos_handling(clean_b2b_invoice, valid_val_res, high_conf_res):
    inv = copy.deepcopy(clean_b2b_invoice)
    inv.place_of_supply = None
    record, exc = map_canonical_to_gstr1(inv, valid_val_res, high_conf_res, invoice_id="INV-001")
    assert record is None
    assert exc is not None
    assert "Missing place of supply" in exc.reason

# 18. Duplicate source invoice detection
def test_duplicate_source_invoice_detection(clean_b2b_invoice, valid_val_res, high_conf_res):
    generator = GSTR1DraftGenerator()
    entries = [
        {"invoice": clean_b2b_invoice, "validation_result": valid_val_res, "confidence_result": high_conf_res, "invoice_id": "INV_DUP_001"},
        {"invoice": clean_b2b_invoice, "validation_result": valid_val_res, "confidence_result": high_conf_res, "invoice_id": "INV_DUP_001"}
    ]
    draft = generator.generate_draft(entries)
    assert draft.included_invoice_count == 1
    assert draft.excluded_invoice_count == 1
    assert "Duplicate source invoice" in draft.exclusions[0].reason

# 19. GSTR-1 validation
def test_gstr1_validation(clean_b2b_invoice, valid_val_res, high_conf_res):
    record, _ = map_canonical_to_gstr1(clean_b2b_invoice, valid_val_res, high_conf_res, invoice_id="INV-001")
    issues = validate_gstr1_record(record)
    assert len(issues) == 0

# 20. CSV export
def test_csv_export(clean_b2b_invoice, valid_val_res, high_conf_res):
    generator = GSTR1DraftGenerator()
    entries = [{"invoice": clean_b2b_invoice, "validation_result": valid_val_res, "confidence_result": high_conf_res, "invoice_id": "INV_001"}]
    draft = generator.generate_draft(entries)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_gstr1.csv"
        export_gstr1_combined_csv(draft, csv_path)
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0

# 21. XLSX export
def test_xlsx_export(clean_b2b_invoice, valid_val_res, high_conf_res):
    generator = GSTR1DraftGenerator()
    entries = [{"invoice": clean_b2b_invoice, "validation_result": valid_val_res, "confidence_result": high_conf_res, "invoice_id": "INV_001"}]
    draft = generator.generate_draft(entries)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        xlsx_path = Path(tmpdir) / "test_gstr1.xlsx"
        export_gstr1_excel(draft, xlsx_path)
        assert xlsx_path.exists()
        assert xlsx_path.stat().st_size > 0

# 22. Lineage / source_invoice_id
def test_lineage_source_invoice_id(clean_b2b_invoice, valid_val_res, high_conf_res):
    record, _ = map_canonical_to_gstr1(clean_b2b_invoice, valid_val_res, high_conf_res, invoice_id="INV_CLEAN_001")
    assert record.source_invoice_id == "INV_CLEAN_001"

# 23. Summary calculations
def test_summary_calculations(clean_b2b_invoice, clean_b2c_invoice, valid_val_res, high_conf_res):
    generator = GSTR1DraftGenerator()
    entries = [
        {"invoice": clean_b2b_invoice, "validation_result": valid_val_res, "confidence_result": high_conf_res, "invoice_id": "INV_001"},
        {"invoice": clean_b2c_invoice, "validation_result": valid_val_res, "confidence_result": high_conf_res, "invoice_id": "INV_002"}
    ]
    draft = generator.generate_draft(entries)
    assert draft.included_invoice_count == 2
    assert draft.b2b_count == 1
    assert draft.b2c_count == 1
    assert draft.total_taxable_value == 3000.0
    assert draft.total_invoice_value == 3540.0

# 24. No silent value mutation
def test_no_silent_value_mutation(clean_b2b_invoice, valid_val_res, high_conf_res):
    inv_copy = copy.deepcopy(clean_b2b_invoice)
    map_canonical_to_gstr1(clean_b2b_invoice, valid_val_res, high_conf_res, invoice_id="INV_001")
    assert clean_b2b_invoice.model_dump() == inv_copy.model_dump()
