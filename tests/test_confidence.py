"""
Unit and Integration Tests for Phase 7 Confidence Scoring & Human Review Workflow.
"""
import pytest
import copy
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
    ValidationCheck,
    CalculatedFinancials,
    GSTINValidationResult
)
from src.validation.service import validate_canonical_invoice
from src.confidence.config import ConfidenceConfig
from src.confidence.exceptions import ConfidenceConfigError
from src.confidence.models import (
    ConfidenceResult,
    ConfidenceTier,
    Recommendation,
    ReviewStatus,
    ReviewerAction,
    ReviewRecord
)
from src.confidence.service import (
    ConfidenceService,
    evaluate_invoice_confidence
)

@pytest.fixture
def valid_invoice() -> CanonicalInvoice:
    return CanonicalInvoice(
        invoice_number="INV-2026-001",
        invoice_date="2026-04-15",
        seller=Party(
            name="Alpha Enterprises Pvt Ltd",
            gstin="27AABCU9603R1ZM",
            address="123 Industrial Area, Pune, Maharashtra"
        ),
        buyer=Party(
            name="Beta Solutions Ltd",
            gstin="27AAACB1234A1Z5",
            address="456 Tech Park, Mumbai, Maharashtra"
        ),
        place_of_supply="27-Maharashtra",
        items=[
            InvoiceItem(
                description="Consulting Services",
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
def valid_validation_result() -> ValidationResult:
    return ValidationResult(
        invoice_id="INV-2026-001",
        status="PASS",
        passed=True,
        requires_review=False,
        error_count=0,
        warning_count=0,
        issues=[],
        checks=[
            ValidationCheck(check_name="TOTAL-001", category="ARITHMETIC", status="PASS", message="Reconciled")
        ],
        calculated_values=CalculatedFinancials(
            calculated_item_taxable_sum=1000.0,
            calculated_cgst=90.0,
            calculated_sgst=90.0,
            calculated_igst=0.0,
            calculated_tax_sum=180.0,
            calculated_grand_total=1180.0,
            grand_total_discrepancy=0.0
        ),
        seller_gstin_result=GSTINValidationResult(
            gstin="27AABCU9603R1ZM",
            valid=True,
            checksum_valid=True
        ),
        buyer_gstin_result=GSTINValidationResult(
            gstin="27AAACB1234A1Z5",
            valid=True,
            checksum_valid=True
        )
    )

# 1. High-confidence valid invoice
def test_high_confidence_valid_invoice(valid_invoice, valid_validation_result):
    service = ConfidenceService()
    res = service.evaluate_confidence(
        invoice=valid_invoice,
        validation_result=valid_validation_result,
        ocr_metadata={"average_confidence": 92.0}
    )
    assert res.overall_score >= 85.0
    assert res.confidence_tier == ConfidenceTier.HIGH.value
    assert res.recommendation == Recommendation.AUTO_ACCEPT.value
    assert res.requires_review is False
    assert len(res.triggered_rules) == 0

# 2. Missing invoice number
def test_missing_invoice_number(valid_invoice, valid_validation_result):
    inv = copy.deepcopy(valid_invoice)
    inv.invoice_number = None
    service = ConfidenceService()
    res = service.evaluate_confidence(invoice=inv, validation_result=valid_validation_result)
    
    assert res.confidence_tier == ConfidenceTier.LOW.value
    assert res.requires_review is True
    assert any(r.rule_id == "OVERRIDE_MISSING_INVOICE_NUMBER" for r in res.triggered_rules)

# 3. Missing invoice date
def test_missing_invoice_date(valid_invoice, valid_validation_result):
    inv = copy.deepcopy(valid_invoice)
    inv.invoice_date = ""
    service = ConfidenceService()
    res = service.evaluate_confidence(invoice=inv, validation_result=valid_validation_result)
    
    assert res.confidence_tier == ConfidenceTier.LOW.value
    assert res.requires_review is True
    assert any(r.rule_id == "OVERRIDE_MISSING_INVOICE_DATE" for r in res.triggered_rules)

# 4. Invalid seller GSTIN
def test_invalid_seller_gstin(valid_invoice, valid_validation_result):
    inv = copy.deepcopy(valid_invoice)
    inv.seller.gstin = "INVALID_GST"
    val_res = copy.deepcopy(valid_validation_result)
    val_res.seller_gstin_result = GSTINValidationResult(gstin="INVALID_GST", valid=False)
    val_res.error_count = 1
    val_res.issues.append(ValidationIssue(
        rule_id="GSTIN-001",
        severity="ERROR",
        field="seller.gstin",
        message="Invalid GSTIN"
    ))
    service = ConfidenceService()
    res = service.evaluate_confidence(invoice=inv, validation_result=val_res)
    
    assert res.confidence_tier == ConfidenceTier.LOW.value
    assert res.requires_review is True
    assert any(r.rule_id == "OVERRIDE_INVALID_SELLER_GSTIN" for r in res.triggered_rules)

# 5. Invalid buyer GSTIN
def test_invalid_buyer_gstin(valid_invoice, valid_validation_result):
    inv = copy.deepcopy(valid_invoice)
    inv.buyer.gstin = "INVALID_BUYER"
    val_res = copy.deepcopy(valid_validation_result)
    val_res.buyer_gstin_result = GSTINValidationResult(gstin="INVALID_BUYER", valid=False)
    val_res.error_count = 1
    val_res.issues.append(ValidationIssue(
        rule_id="GSTIN-002",
        severity="ERROR",
        field="buyer.gstin",
        message="Invalid Buyer GSTIN"
    ))
    service = ConfidenceService()
    res = service.evaluate_confidence(invoice=inv, validation_result=val_res)
    
    # Financial/validation error forces review
    assert res.requires_review is True
    assert res.recommendation != Recommendation.AUTO_ACCEPT.value

# 6. Financial validation failure
def test_financial_validation_failure(valid_invoice, valid_validation_result):
    val_res = copy.deepcopy(valid_validation_result)
    val_res.status = "FAIL"
    val_res.error_count = 1
    val_res.issues.append(ValidationIssue(
        rule_id="MATH-001",
        severity="ERROR",
        field="items",
        message="Item taxable sum does not match"
    ))
    service = ConfidenceService()
    res = service.evaluate_confidence(invoice=valid_invoice, validation_result=val_res)
    
    assert res.requires_review is True
    assert res.confidence_tier != ConfidenceTier.HIGH.value
    assert res.recommendation != Recommendation.AUTO_ACCEPT.value

# 7. TOTAL-001 failure
def test_total_001_failure(valid_invoice, valid_validation_result):
    val_res = copy.deepcopy(valid_validation_result)
    val_res.status = "FAIL"
    val_res.error_count = 1
    val_res.calculated_values.grand_total_discrepancy = 500.0
    val_res.issues.append(ValidationIssue(
        rule_id="TOTAL-001",
        severity="ERROR",
        field="grand_total",
        message="Grand total discrepancy of ₹500"
    ))
    service = ConfidenceService()
    res = service.evaluate_confidence(invoice=valid_invoice, validation_result=val_res)
    
    assert res.confidence_tier == ConfidenceTier.LOW.value
    assert res.requires_review is True
    assert any(r.rule_id == "OVERRIDE_TOTAL_001_FAILED" for r in res.triggered_rules)

# 8. Poor OCR confidence
def test_poor_ocr_confidence(valid_invoice, valid_validation_result):
    service = ConfidenceService()
    res = service.evaluate_confidence(
        invoice=valid_invoice,
        validation_result=valid_validation_result,
        ocr_metadata={"average_confidence": 35.0}
    )
    assert res.confidence_tier == ConfidenceTier.LOW.value
    assert res.recommendation == Recommendation.REJECT_OR_REPROCESS.value
    assert any(r.rule_id == "OVERRIDE_SEVERE_OCR_FAILURE" for r in res.triggered_rules)

# 9. Missing line items
def test_missing_line_items(valid_invoice, valid_validation_result):
    inv = copy.deepcopy(valid_invoice)
    inv.items = []
    service = ConfidenceService()
    res = service.evaluate_confidence(invoice=inv, validation_result=valid_validation_result)
    assert res.factors["line_item_quality"].score == 30.0

# 10. Missing buyer GSTIN for B2C
def test_missing_buyer_gstin_b2c(valid_invoice, valid_validation_result):
    inv = copy.deepcopy(valid_invoice)
    inv.buyer.gstin = None
    val_res = copy.deepcopy(valid_validation_result)
    val_res.buyer_gstin_result = None
    service = ConfidenceService()
    res = service.evaluate_confidence(invoice=inv, validation_result=val_res, ocr_metadata={"average_confidence": 90.0})
    
    # B2C valid invoices should still achieve HIGH confidence if everything else reconciles
    assert res.confidence_tier == ConfidenceTier.HIGH.value
    assert res.recommendation == Recommendation.AUTO_ACCEPT.value

# 11. Multiple validation errors
def test_multiple_validation_errors(valid_invoice, valid_validation_result):
    val_res = copy.deepcopy(valid_validation_result)
    val_res.status = "FAIL"
    val_res.error_count = 3
    val_res.issues = [
        ValidationIssue(rule_id="FIELD-001", severity="ERROR", field="invoice_number", message="Missing"),
        ValidationIssue(rule_id="GSTIN-001", severity="ERROR", field="seller.gstin", message="Invalid"),
        ValidationIssue(rule_id="TOTAL-001", severity="ERROR", field="grand_total", message="Mismatch")
    ]
    service = ConfidenceService()
    res = service.evaluate_confidence(invoice=valid_invoice, validation_result=val_res)
    
    assert res.confidence_tier == ConfidenceTier.LOW.value
    assert res.recommendation == Recommendation.REJECT_OR_REPROCESS.value
    assert any(r.rule_id == "OVERRIDE_MULTIPLE_CRITICAL_ERRORS" for r in res.triggered_rules)

# 12. Confidence score boundaries
def test_confidence_score_boundaries(valid_invoice, valid_validation_result):
    service = ConfidenceService()
    res = service.evaluate_confidence(invoice=valid_invoice, validation_result=valid_validation_result)
    assert 0.0 <= res.overall_score <= 100.0
    for factor in res.factors.values():
        assert 0.0 <= factor.score <= 100.0
        assert 0.0 <= factor.weighted_score <= 100.0

# 13. HIGH cannot occur with critical financial failure
def test_high_cannot_occur_with_financial_failure(valid_invoice, valid_validation_result):
    val_res = copy.deepcopy(valid_validation_result)
    val_res.status = "FAIL"
    val_res.error_count = 1
    val_res.issues.append(ValidationIssue(
        rule_id="TOTAL-001",
        severity="ERROR",
        field="grand_total",
        message="Discrepancy ₹100"
    ))
    service = ConfidenceService()
    res = service.evaluate_confidence(invoice=valid_invoice, validation_result=val_res)
    assert res.confidence_tier != ConfidenceTier.HIGH.value
    assert res.recommendation != Recommendation.AUTO_ACCEPT.value

# 14. Review required logic
def test_review_required_logic(valid_invoice, valid_validation_result):
    val_res = copy.deepcopy(valid_validation_result)
    val_res.warning_count = 1
    val_res.issues.append(ValidationIssue(
        rule_id="STATE-002",
        severity="WARNING",
        field="tax",
        message="Intra-state IGST warning"
    ))
    service = ConfidenceService()
    res = service.evaluate_confidence(invoice=valid_invoice, validation_result=val_res)
    assert res.requires_review is True

# 15. Approve action
def test_review_approve_action(valid_invoice, valid_validation_result):
    service = ConfidenceService()
    conf_res = service.evaluate_confidence(invoice=valid_invoice, validation_result=valid_validation_result)
    record = service.create_review_record(conf_res, valid_validation_result)
    assert record.review_status == ReviewStatus.PENDING_REVIEW.value
    
    updated = service.apply_review_action(record, ReviewerAction.APPROVE, notes="Approved by Senior Auditor")
    assert updated.review_status == ReviewStatus.APPROVED.value
    assert updated.reviewer_action == "APPROVE"
    assert updated.notes == "Approved by Senior Auditor"
    assert updated.reviewed_at is not None

# 16. Reject action
def test_review_reject_action(valid_invoice, valid_validation_result):
    service = ConfidenceService()
    conf_res = service.evaluate_confidence(invoice=valid_invoice, validation_result=valid_validation_result)
    record = service.create_review_record(conf_res, valid_validation_result)
    
    updated = service.apply_review_action(record, ReviewerAction.REJECT, notes="Illegible invoice scan")
    assert updated.review_status == ReviewStatus.REJECTED.value
    assert updated.reviewer_action == "REJECT"

# 17. Reprocess action
def test_review_reprocess_action(valid_invoice, valid_validation_result):
    service = ConfidenceService()
    conf_res = service.evaluate_confidence(invoice=valid_invoice, validation_result=valid_validation_result)
    record = service.create_review_record(conf_res, valid_validation_result)
    
    updated = service.apply_review_action(record, ReviewerAction.REQUEST_REPROCESSING, notes="Needs deskew and higher DPI OCR")
    assert updated.review_status == ReviewStatus.REPROCESS_REQUIRED.value
    assert updated.reviewer_action == "REQUEST_REPROCESSING"

# 18. Original invoice remains unchanged (immutability)
def test_original_invoice_immutability(valid_invoice, valid_validation_result):
    inv_copy = copy.deepcopy(valid_invoice)
    service = ConfidenceService()
    conf_res = service.evaluate_confidence(invoice=valid_invoice, validation_result=valid_validation_result)
    record = service.create_review_record(conf_res, valid_validation_result)
    service.apply_review_action(record, ReviewerAction.APPROVE, notes="Approved")
    
    assert valid_invoice.model_dump() == inv_copy.model_dump()

# 19. Confidence result schema validation
def test_confidence_result_schema_validation(valid_invoice, valid_validation_result):
    res = evaluate_invoice_confidence(invoice=valid_invoice, validation_result=valid_validation_result)
    assert isinstance(res, ConfidenceResult)
    data = res.model_dump()
    assert "overall_score" in data
    assert "confidence_tier" in data
    assert "factors" in data
    assert "triggered_rules" in data
    assert "explanation" in data

# 20. Config validation test
def test_confidence_config_validation():
    with pytest.raises(ConfidenceConfigError):
        ConfidenceConfig(ocr_weight=0.5, field_completeness_weight=0.8)
        
    with pytest.raises(ConfidenceConfigError):
        ConfidenceConfig(high_threshold=50.0, medium_threshold=70.0)
