"""
Integration and End-to-End Tests for Phase 9 Streamlit Application & Pipeline Modules.
"""
import pytest
import os
import tempfile
from pathlib import Path

from src.preprocessing.pipeline import preprocess_image
from src.preprocessing.config import PreprocessingConfig
from src.ocr.service import extract_text
from src.ocr.config import OCRConfig
from src.extraction.service import extract_invoice_from_ocr
from src.extraction.config import ExtractionConfig
from src.validation.service import validate_canonical_invoice
from src.validation.config import ValidationConfig
from src.confidence.service import ConfidenceService
from src.confidence.config import ConfidenceConfig
from src.confidence.models import ReviewerAction, ReviewStatus, ReviewRecord, ConfidenceTier
from src.gst.gstr1_models import GSTR1Draft, GSTR1B2BInvoice, GSTR1B2CSInvoice
from src.gst.gstr1_mapper import map_canonical_to_gstr1
from src.export.gstr1_csv import export_gstr1_combined_csv
from src.export.gstr1_excel import export_gstr1_excel
from app.dashboard import load_or_create_review_record, load_batch_gstr1_draft

RAW_INVOICES_DIR = Path("data/raw/invoices")

# 1. Module Imports Verification
def test_application_module_imports():
    import app.dashboard
    assert hasattr(app.dashboard, "main")
    assert hasattr(app.dashboard, "load_or_create_review_record")
    assert hasattr(app.dashboard, "load_batch_gstr1_draft")

# 2. End-to-End Single Invoice Pipeline (Clean Invoice)
def test_e2e_clean_invoice_pipeline():
    clean_sample_path = RAW_INVOICES_DIR / "INV_CLEAN_001.png"
    assert clean_sample_path.exists(), "Sample invoice image must exist"

    with tempfile.TemporaryDirectory() as tmpdir:
        proc_path = Path(tmpdir) / "proc_clean.png"

        # Step 1: Preprocessing
        prep_config = PreprocessingConfig()
        prep_res = preprocess_image(str(clean_sample_path), str(proc_path), prep_config)
        assert prep_res["success"] is True
        assert proc_path.exists()

        # Step 2: OCR
        ocr_config = OCRConfig()
        ocr_res = extract_text(str(proc_path), ocr_config)
        assert ocr_res.success is True
        assert len(ocr_res.raw_text) > 0

        # Step 3: Extraction (Mock Mode)
        ext_config = ExtractionConfig(provider="mock")
        invoice = extract_invoice_from_ocr(
            ocr_text=ocr_res.raw_text,
            source_file="INV_CLEAN_001.png",
            config=ext_config
        )
        assert invoice.invoice_number is not None
        assert invoice.grand_total is not None

        # Step 4: Deterministic Validation
        val_config = ValidationConfig(amount_tolerance=1.0)
        val_res = validate_canonical_invoice(invoice=invoice, config=val_config, invoice_id="INV_CLEAN_001")
        assert val_res.status == "PASS"
        assert val_res.error_count == 0

        # Step 5: Confidence Scoring
        conf_config = ConfidenceConfig()
        conf_service = ConfidenceService(conf_config)
        conf_res = conf_service.evaluate_confidence(
            invoice=invoice,
            validation_result=val_res,
            ocr_metadata={"average_confidence": ocr_res.average_confidence},
            invoice_id="INV_CLEAN_001"
        )
        assert conf_res.confidence_tier == ConfidenceTier.HIGH.value
        assert conf_res.overall_score >= 85.0

        # Step 6: Review Record Creation
        rev_rec = conf_service.create_review_record(conf_res, val_res)
        assert rev_rec.review_status == ReviewStatus.PENDING_REVIEW.value

        # Step 7: GSTR-1 Mapping
        gstr1_rec, gstr1_exc = map_canonical_to_gstr1(
            invoice=invoice,
            validation_result=val_res,
            confidence_result=conf_res,
            review_record=rev_rec,
            invoice_id="INV_CLEAN_001"
        )
        assert gstr1_exc is None
        assert isinstance(gstr1_rec, GSTR1B2BInvoice)
        assert gstr1_rec.taxable_value == 1000.0
        assert gstr1_rec.invoice_value == 1180.0

# 3. End-to-End Single Invoice Pipeline (Invalid Math Invoice Exclusion)
def test_e2e_invalid_math_invoice_pipeline():
    inv_sample_path = RAW_INVOICES_DIR / "INV_INV_001.png"
    assert inv_sample_path.exists()

    with tempfile.TemporaryDirectory() as tmpdir:
        proc_path = Path(tmpdir) / "proc_inv.png"

        # Step 1: Preprocessing & OCR
        prep_res = preprocess_image(str(inv_sample_path), str(proc_path), PreprocessingConfig())
        ocr_res = extract_text(str(proc_path), OCRConfig())

        # Step 2: Extraction
        invoice = extract_invoice_from_ocr(
            ocr_text=ocr_res.raw_text,
            source_file="INV_INV_001.png",
            config=ExtractionConfig(provider="mock")
        )

        # Step 3: Validation (Must Catch TOTAL-001)
        val_res = validate_canonical_invoice(invoice=invoice, config=ValidationConfig(), invoice_id="INV_INV_001")
        assert val_res.status == "FAIL"
        assert any(i.rule_id == "TOTAL-001" for i in val_res.issues)

        # Step 4: Confidence Scoring (Must Cap to LOW Tier)
        conf_service = ConfidenceService(ConfidenceConfig())
        conf_res = conf_service.evaluate_confidence(
            invoice=invoice,
            validation_result=val_res,
            ocr_metadata={"average_confidence": ocr_res.average_confidence},
            invoice_id="INV_INV_001"
        )
        assert conf_res.confidence_tier == ConfidenceTier.LOW.value
        assert conf_res.requires_review is True

        # Step 5: GSTR-1 Mapping (Must Exclude Ineligible Invoice)
        rev_rec = conf_service.create_review_record(conf_res, val_res)
        gstr1_rec, gstr1_exc = map_canonical_to_gstr1(
            invoice=invoice,
            validation_result=val_res,
            confidence_result=conf_res,
            review_record=rev_rec,
            invoice_id="INV_INV_001"
        )
        assert gstr1_rec is None
        assert gstr1_exc is not None
        assert "TOTAL-001" in gstr1_exc.reason

# 4. GSTR-1 Batch Summary Load Test
def test_batch_gstr1_summary_loader():
    draft = load_batch_gstr1_draft()
    assert isinstance(draft, GSTR1Draft)
    assert draft.source_invoice_count == 25
    assert draft.included_invoice_count == 15
    assert draft.excluded_invoice_count == 10
    assert draft.b2b_count == 12
    assert draft.b2c_count == 3
    assert draft.total_taxable_value == 40000.0

# 5. Export Artifacts Creation Test
def test_export_artifacts_creation():
    draft = load_batch_gstr1_draft()
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_gstr1.csv"
        xlsx_path = Path(tmpdir) / "test_gstr1.xlsx"

        export_gstr1_combined_csv(draft, csv_path)
        export_gstr1_excel(draft, xlsx_path)

        assert csv_path.exists()
        assert csv_path.stat().st_size > 0
        assert xlsx_path.exists()
        assert xlsx_path.stat().st_size > 0
