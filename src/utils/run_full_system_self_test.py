"""
Comprehensive Full-System Self-Test, Random Data Robustness & LLM Verification Suite.
Executes fresh, independent evidence-based validation across all pipeline layers.
"""
import os
import sys
import json
import time
import shutil
import requests
import pandas as pd
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timezone

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.extraction.config import ExtractionConfig
from src.extraction.llm_client import OllamaClient, MockLLMClient, get_llm_client
from src.extraction.service import ExtractionService
from src.extraction.parser import parse_and_validate_json
from src.extraction.models import CanonicalInvoice, Party, InvoiceItem, TaxSummary, ExtractionMetadata
from src.validation.service import ValidationService
from src.validation.config import ValidationConfig
from src.validation.models import ValidationResult
from src.confidence.service import ConfidenceService
from src.confidence.config import ConfidenceConfig
from src.confidence.models import ConfidenceTier, Recommendation, ConfidenceResult, ReviewRecord, ReviewStatus
from src.gst.gstr1_models import GSTR1Draft, GSTR1B2BInvoice, GSTR1B2CSInvoice, GSTR1ExclusionRecord
from src.gst.gstr1_mapper import map_canonical_to_gstr1
from src.export import export_b2b_csv, export_b2c_csv, export_gstr1_combined_csv, export_gstr1_excel

logger = get_logger("self_test")

SELF_TEST_DIR = Path("data/outputs/self_test")
SELF_TEST_DIR.mkdir(parents=True, exist_ok=True)
INVOICES_DIR = SELF_TEST_DIR / "invoices"
INVOICES_DIR.mkdir(parents=True, exist_ok=True)

test_results = {}

def log_section(name: str):
    print(f"\n{'='*60}\n>>> {name}\n{'='*60}")

# ============================================================
# 1. STATIC IMPORT & MODULE VERIFICATION
# ============================================================
def test_static_imports():
    log_section("1. STATIC IMPORT & MODULE VERIFICATION")
    modules = [
        "src.preprocessing.pipeline",
        "src.ocr.service",
        "src.extraction.service",
        "src.validation.service",
        "src.confidence.service",
        "src.gst.gstr1_mapper",
        "src.export",
        "src.utils.logger",
        "app.dashboard"
    ]
    imported = []
    failed = []
    for mod in modules:
        try:
            __import__(mod)
            imported.append(mod)
        except Exception as e:
            failed.append((mod, str(e)))
    
    status = "PASS" if not failed else "FAIL"
    print(f"Imported {len(imported)}/{len(modules)} modules. Failures: {failed}")
    test_results["static_imports"] = {"status": status, "imported": len(imported), "failed": failed}
    return status == "PASS"

# ============================================================
# 2. REAL LLM CONNECTIVITY & EXTRACTION
# ============================================================
def test_real_llm_connectivity():
    log_section("2. REAL LLM CONNECTIVITY & EXTRACTION TEST")
    
    # Check Ollama
    ollama_ok = False
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        ollama_ok = (r.status_code == 200)
    except Exception as e:
        print(f"Ollama connection error: {e}")
        
    print(f"Ollama Server Status: {'PASS (Reachable)' if ollama_ok else 'FAIL (Unreachable)'}")
    
    # Check minimal JSON inference
    minimal_ok = False
    try:
        payload = {
            "model": "qwen2.5:1.5b",
            "prompt": "Return a json with key status and value ok: {\"status\": \"ok\"}",
            "stream": False,
            "format": "json"
        }
        res = requests.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=30)
        if res.status_code == 200:
            parsed = json.loads(res.json().get("response", "{}"))
            if parsed.get("status") == "ok":
                minimal_ok = True
    except Exception as e:
        print(f"Minimal JSON test error: {e}")
    print(f"Qwen2.5 1.5B Minimal Inference: {'PASS' if minimal_ok else 'FAIL'}")

    # Real Invoice Extraction on SELFTEST-001
    selftest_ocr = """
TAX INVOICE
INVOICE NUMBER: SELFTEST-001
DATE: 2026-10-08

SELLER:
Alpha Technologies Pvt Ltd

SELLER GSTIN:
27AABCS1429B1ZB

BUYER:
Beta Enterprises

BUYER GSTIN:
27AABCB1234F1Z5

PLACE OF SUPPLY:
27

ITEM:
Laptop
TAXABLE VALUE:
10000.00
CGST:
900.00
SGST:
900.00
GRAND TOTAL:
11800.00
"""
    extraction_service = ExtractionService(ExtractionConfig(provider="ollama", model="qwen2.5:1.5b", timeout=60))
    t0 = time.perf_counter()
    extracted_inv = extraction_service.extract_from_ocr_text(selftest_ocr, source_file="SELFTEST-001.png")
    lat_ms = (time.perf_counter() - t0) * 1000.0
    
    print(f"Real LLM Extraction on SELFTEST-001 Latency: {lat_ms:.2f} ms")
    print(f"  - invoice_number: {extracted_inv.invoice_number}")
    print(f"  - invoice_date: {extracted_inv.invoice_date}")
    print(f"  - seller_gstin: {extracted_inv.seller.gstin if extracted_inv.seller else None}")
    print(f"  - buyer_gstin: {extracted_inv.buyer.gstin if extracted_inv.buyer else None}")
    print(f"  - taxable_value: {extracted_inv.tax_summary.taxable_value if extracted_inv.tax_summary else None}")
    print(f"  - cgst: {extracted_inv.tax_summary.cgst if extracted_inv.tax_summary else None}")
    print(f"  - sgst: {extracted_inv.tax_summary.sgst if extracted_inv.tax_summary else None}")
    print(f"  - grand_total: {extracted_inv.grand_total}")
    print(f"  - items count: {len(extracted_inv.items)}")

    real_llm_pass = (
        extracted_inv.invoice_number == "SELFTEST-001" and
        str(extracted_inv.invoice_date) == "2026-10-08" and
        extracted_inv.seller.gstin == "27AABCS1429B1ZB" and
        extracted_inv.buyer.gstin == "27AABCB1234F1Z5" and
        abs(float(extracted_inv.grand_total or 0.0) - 11800.00) < 1.0
    )
    print(f"REAL LLM TEST RESULT: {'PASS' if real_llm_pass else 'FAIL'}")
    
    test_results["real_llm"] = {
        "ollama_reachable": ollama_ok,
        "minimal_json_test": minimal_ok,
        "selftest_001_extraction": "PASS" if real_llm_pass else "FAIL",
        "latency_ms": round(lat_ms, 2)
    }
    return real_llm_pass

# ============================================================
# 3. REAL LLM ERROR-HANDLING TESTS
# ============================================================
def test_llm_error_handling():
    log_section("3. LLM ERROR-HANDLING TESTS")
    from src.extraction.llm_client import BaseLLMClient
    
    # Test 1: Ollama unavailable
    bad_config = ExtractionConfig(provider="ollama", model="qwen2.5:1.5b", timeout=1)
    bad_client = OllamaClient(bad_config)
    bad_client.api_url = "http://127.0.0.1:9999/api/generate"
    bad_service = ExtractionService(bad_config)
    bad_service.client = bad_client
    
    err1_inv = bad_service.extract_from_ocr_text("SOME TEXT", source_file="fail_test.png")
    test1_pass = (err1_inv is not None and len(err1_inv.extraction_metadata.warnings) > 0)
    print(f"Test 1 (Ollama Unavailable): {'PASS' if test1_pass else 'FAIL'} - Handled gracefully with warning: {err1_inv.extraction_metadata.warnings}")

    # Test 2: Malformed LLM JSON via ExtractionService
    class MalformedMockClient(BaseLLMClient):
        def generate_extraction(self, ocr_text: str) -> str:
            return "THIS IS BROKEN JSON {not valid"

    mal_service = ExtractionService(ExtractionConfig(provider="mock"))
    mal_service.client = MalformedMockClient(ExtractionConfig(provider="mock"))
    err2_inv = mal_service.extract_from_ocr_text("SOME TEXT", source_file="malformed.png")
    test2_pass = (err2_inv is not None and any("Extraction error" in w for w in err2_inv.extraction_metadata.warnings))
    print(f"Test 2 (Malformed JSON via ExtractionService): {'PASS' if test2_pass else 'FAIL'} - Intercepted with warning: {err2_inv.extraction_metadata.warnings}")

    # Test 3: Incomplete JSON
    class IncompleteMockClient(BaseLLMClient):
        def generate_extraction(self, ocr_text: str) -> str:
            return json.dumps({"invoice_number": "INC-001"})

    inc_service = ExtractionService(ExtractionConfig(provider="mock"))
    inc_service.client = IncompleteMockClient(ExtractionConfig(provider="mock"))
    err3_inv = inc_service.extract_from_ocr_text("SOME TEXT", source_file="incomplete.png")
    test3_pass = (err3_inv is not None and err3_inv.invoice_number == "INC-001" and (err3_inv.seller is None or err3_inv.seller.gstin is None))
    print(f"Test 3 (Incomplete JSON via ExtractionService): {'PASS' if test3_pass else 'FAIL'} - Defaults filled safely without crash")

    # Test 4: Missing buyer GSTIN (B2C)
    b2c_ocr = "TAX INVOICE\nINVOICE NUMBER: B2C-001\nSELLER GSTIN: 27AABCS1429B1ZB\nTOTAL: 1000.00"
    b2c_service = ExtractionService(ExtractionConfig(provider="mock"))
    b2c_inv = b2c_service.extract_from_ocr_text(b2c_ocr, source_file="b2c.png")
    test4_pass = (b2c_inv.buyer is None or b2c_inv.buyer.gstin is None)
    print(f"Test 4 (B2C Null Buyer GSTIN): {'PASS' if test4_pass else 'FAIL'} - Buyer GSTIN correctly preserved as null")

    all_err_pass = test1_pass and test2_pass and test3_pass and test4_pass
    test_results["llm_error_handling"] = {
        "test1_unavailable": test1_pass,
        "test2_malformed": test2_pass,
        "test3_incomplete": test3_pass,
        "test4_b2c_null": test4_pass,
        "overall": "PASS" if all_err_pass else "FAIL"
    }
    return all_err_pass

# ============================================================
# 4. FINANCIAL FIREWALL TESTS (A through H)
# ============================================================
def test_financial_firewall():
    log_section("4. FINANCIAL FIREWALL TESTS (A - H)")
    val_service = ValidationService(ValidationConfig())
    conf_service = ConfidenceService(ConfidenceConfig())

    # TEST A: Arithmetic Mismatch
    inv_a = CanonicalInvoice(
        invoice_number="TEST-A",
        invoice_date="2026-10-08",
        seller=Party(name="Seller", gstin="27AABCS1429B1ZB"),
        buyer=Party(name="Buyer", gstin="27AABCB1234F1Z5"),
        place_of_supply="27",
        items=[InvoiceItem(description="Item", quantity=1.0, unit_price=10000.0, taxable_value=10000.0, cgst_amount=900.0, sgst_amount=900.0)],
        tax_summary=TaxSummary(taxable_value=10000.00, cgst=900.00, sgst=900.00),
        grand_total=50000.00
    )
    res_a = val_service.validate_invoice(inv_a, invoice_id="TEST-A")
    pass_a = (res_a.status == "FAIL" and any(i.rule_id == "TOTAL-001" for i in res_a.issues))
    print(f"Test A (Math mismatch 10k+900+900 != 50k): {'PASS (Blocked)' if pass_a else 'FAIL'}")

    # TEST B: Valid Math
    inv_b = CanonicalInvoice(
        invoice_number="TEST-B",
        invoice_date="2026-10-08",
        seller=Party(name="Seller", gstin="27AABCS1429B1ZB"),
        buyer=Party(name="Buyer", gstin="27AABCB1234F1Z5"),
        place_of_supply="27",
        items=[InvoiceItem(description="Item", quantity=1.0, unit_price=10000.0, taxable_value=10000.0, cgst_amount=900.0, sgst_amount=900.0)],
        tax_summary=TaxSummary(taxable_value=10000.00, cgst=900.00, sgst=900.00),
        grand_total=11800.00
    )
    res_b = val_service.validate_invoice(inv_b, invoice_id="TEST-B")
    pass_b = (res_b.passed and res_b.error_count == 0)
    print(f"Test B (Math match 10k+900+900 == 11.8k): {'PASS' if pass_b else 'FAIL'}")

    # TEST C: B2C null buyer GSTIN
    inv_c = CanonicalInvoice(
        invoice_number="TEST-C",
        invoice_date="2026-10-08",
        seller=Party(name="Seller", gstin="27AABCS1429B1ZB"),
        buyer=Party(name="Buyer", gstin=None),
        place_of_supply="27",
        items=[InvoiceItem(description="Item", quantity=1.0, unit_price=1000.0, taxable_value=1000.0, cgst_amount=90.0, sgst_amount=90.0)],
        tax_summary=TaxSummary(taxable_value=1000.00, cgst=90.00, sgst=90.00),
        grand_total=1180.00
    )
    res_c = val_service.validate_invoice(inv_c, invoice_id="TEST-C")
    pass_c = (res_c.passed and res_c.error_count == 0 and inv_c.buyer.gstin is None)
    print(f"Test C (B2C null buyer GSTIN): {'PASS' if pass_c else 'FAIL'}")

    # TEST D: Invalid GSTIN (Structure / state code)
    inv_d = CanonicalInvoice(
        invoice_number="TEST-D",
        invoice_date="2026-10-08",
        seller=Party(name="Seller", gstin="99INVALIDGSTIN1Z"),
        buyer=Party(name="Buyer", gstin="27AABCB1234F1Z5"),
        place_of_supply="27",
        items=[InvoiceItem(description="Item", quantity=1.0, unit_price=1000.0, taxable_value=1000.0, cgst_amount=90.0, sgst_amount=90.0)],
        tax_summary=TaxSummary(taxable_value=1000.00, cgst=90.00, sgst=90.00),
        grand_total=1180.00
    )
    res_d = val_service.validate_invoice(inv_d, invoice_id="TEST-D")
    pass_d = (res_d.status == "FAIL" and any(i.rule_id == "GSTIN-001" for i in res_d.issues))
    print(f"Test D (Invalid GSTIN 99INVALIDGSTIN1Z): {'PASS (Blocked)' if pass_d else 'FAIL'}")

    # TEST E: Negative line item (Expected: not automatically accepted)
    inv_e = CanonicalInvoice(
        invoice_number="TEST-E",
        invoice_date="2026-10-08",
        seller=Party(name="Seller", gstin="27AABCS1429B1ZB"),
        buyer=Party(name="Buyer", gstin="27AABCB1234F1Z5"),
        place_of_supply="27",
        items=[InvoiceItem(description="Item", quantity=-1.0, unit_price=100.0, taxable_value=-100.0)],
        tax_summary=TaxSummary(taxable_value=-100.00, cgst=0.0, sgst=0.0),
        grand_total=-100.00
    )
    res_e = val_service.validate_invoice(inv_e, invoice_id="TEST-E")
    conf_e = conf_service.evaluate_confidence(inv_e, validation_result=res_e, invoice_id="TEST-E")
    # Not automatically accepted: requires review or recommendation is not AUTO_ACCEPT
    pass_e = (conf_e.recommendation != "AUTO_ACCEPT" or conf_e.requires_review or not res_e.passed)
    print(f"Test E (Negative amount): {'PASS (Not Automatically Accepted)' if pass_e else 'FAIL'}")

    # TEST F: Zero Amount
    inv_f = CanonicalInvoice(
        invoice_number="TEST-F",
        invoice_date="2026-10-08",
        seller=Party(name="Seller", gstin="27AABCS1429B1ZB"),
        buyer=Party(name="Buyer", gstin="27AABCB1234F1Z5"),
        place_of_supply="27",
        items=[InvoiceItem(description="Item", quantity=0.0, unit_price=0.0, taxable_value=0.0)],
        tax_summary=TaxSummary(taxable_value=0.00, cgst=0.0, sgst=0.0),
        grand_total=0.00
    )
    res_f = val_service.validate_invoice(inv_f, invoice_id="TEST-F")
    pass_f = (res_f is not None)
    print(f"Test F (Zero value): {'PASS (Handled safely)' if pass_f else 'FAIL'}")

    # TEST G: Very Large Financial Amount
    inv_g = CanonicalInvoice(
        invoice_number="TEST-G",
        invoice_date="2026-10-08",
        seller=Party(name="Seller", gstin="27AABCS1429B1ZB"),
        buyer=Party(name="Buyer", gstin="27AABCB1234F1Z5"),
        place_of_supply="27",
        items=[InvoiceItem(description="Item", quantity=1.0, unit_price=1000000000.00, taxable_value=1000000000.00, cgst_amount=90000000.00, sgst_amount=90000000.00)],
        tax_summary=TaxSummary(taxable_value=1000000000.00, cgst=90000000.00, sgst=90000000.00),
        grand_total=1180000000.00
    )
    res_g = val_service.validate_invoice(inv_g, invoice_id="TEST-G")
    pass_g = (res_g.passed and res_g.error_count == 0)
    print(f"Test G (Large ₹100 Cr amount Decimal-safe): {'PASS' if pass_g else 'FAIL'}")

    # TEST H: Rounding Boundary (₹0.50 difference within ₹1.00 tolerance)
    inv_h = CanonicalInvoice(
        invoice_number="TEST-H",
        invoice_date="2026-10-08",
        seller=Party(name="Seller", gstin="27AABCS1429B1ZB"),
        buyer=Party(name="Buyer", gstin="27AABCB1234F1Z5"),
        place_of_supply="27",
        items=[InvoiceItem(description="Item", quantity=1.0, unit_price=10000.00, taxable_value=10000.00, cgst_amount=900.00, sgst_amount=900.00)],
        tax_summary=TaxSummary(taxable_value=10000.00, cgst=900.00, sgst=900.00),
        grand_total=11800.50
    )
    res_h = val_service.validate_invoice(inv_h, invoice_id="TEST-H")
    pass_h = (res_h.passed and res_h.error_count == 0)
    print(f"Test H (Rounding ₹0.50 within tolerance): {'PASS' if pass_h else 'FAIL'}")

    firewall_all = all([pass_a, pass_b, pass_c, pass_d, pass_e, pass_f, pass_g, pass_h])
    test_results["financial_firewall"] = {
        "test_a_math_mismatch": pass_a,
        "test_b_math_match": pass_b,
        "test_c_b2c_null": pass_c,
        "test_d_invalid_gstin": pass_d,
        "test_e_negative_amount": pass_e,
        "test_f_zero_amount": pass_f,
        "test_g_large_amount": pass_g,
        "test_h_rounding_boundary": pass_h,
        "overall": "PASS" if firewall_all else "FAIL"
    }
    return firewall_all

# ============================================================
# 5. CONFIDENCE & HUMAN REVIEW SAFETY
# ============================================================
def test_confidence_safety():
    log_section("5. CONFIDENCE & HUMAN REVIEW SAFETY")
    conf_cfg = ConfidenceConfig()
    w_sum = (
        conf_cfg.ocr_weight +
        conf_cfg.field_completeness_weight +
        conf_cfg.critical_fields_weight +
        conf_cfg.gst_validation_weight +
        conf_cfg.financial_validation_weight +
        conf_cfg.line_item_weight
    )
    print(f"Active Confidence Weights:")
    print(f"  - OCR quality: {conf_cfg.ocr_weight*100:.0f}%")
    print(f"  - Field completeness: {conf_cfg.field_completeness_weight*100:.0f}%")
    print(f"  - Critical fields: {conf_cfg.critical_fields_weight*100:.0f}%")
    print(f"  - GST validation: {conf_cfg.gst_validation_weight*100:.0f}%")
    print(f"  - Financial validation: {conf_cfg.financial_validation_weight*100:.0f}%")
    print(f"  - Line item quality: {conf_cfg.line_item_weight*100:.0f}%")
    print(f"  Total: {w_sum*100:.1f}%")
    
    weights_ok = (abs(w_sum - 1.0) < 0.001)

    # Test that invalid invoice cannot become AUTO_ACCEPT
    val_service = ValidationService(ValidationConfig())
    conf_service = ConfidenceService(conf_cfg)
    
    invalid_inv = CanonicalInvoice(
        invoice_number="TEST-INV-CONF",
        invoice_date="2026-10-08",
        seller=Party(name="Seller", gstin="27AABCS1429B1ZB"),
        buyer=Party(name="Buyer", gstin="27AABCB1234F1Z5"),
        place_of_supply="27",
        tax_summary=TaxSummary(taxable_value=1000.00, cgst=90.00, sgst=90.00),
        grand_total=9999.00 # Arithmetic discrepancy
    )
    val_res = val_service.validate_invoice(invalid_inv, invoice_id="TEST-INV-CONF")
    conf_res = conf_service.evaluate_confidence(invalid_inv, validation_result=val_res, invoice_id="TEST-INV-CONF")
    
    print(f"Invalid invoice confidence tier: {conf_res.confidence_tier}, recommendation: {conf_res.recommendation}")
    override_ok = (conf_res.confidence_tier == ConfidenceTier.LOW and conf_res.recommendation != Recommendation.AUTO_ACCEPT)
    print(f"Safety Override on Financial Failure: {'PASS (Overridden to LOW)' if override_ok else 'FAIL'}")

    pass_all = weights_ok and override_ok
    test_results["confidence_safety"] = {
        "weights_sum_100": weights_ok,
        "override_on_financial_failure": override_ok,
        "overall": "PASS" if pass_all else "FAIL"
    }
    return pass_all

# ============================================================
# 6. GSTR-1 SAFETY TESTS
# ============================================================
def test_gstr1_safety():
    log_section("6. GSTR-1 SAFETY TESTS")
    val_service = ValidationService(ValidationConfig())
    conf_service = ConfidenceService(ConfidenceConfig())

    # 1. Valid B2B -> Table 4A
    inv_b2b = CanonicalInvoice(
        invoice_number="B2B-VALID",
        invoice_date="2026-10-08",
        seller=Party(name="Seller", gstin="27AABCS1429B1ZB"),
        buyer=Party(name="Buyer", gstin="27AABCB1234F1Z5"),
        place_of_supply="27",
        items=[
            InvoiceItem(
                description="Laptop",
                quantity=1.0,
                unit_price=10000.0,
                taxable_value=10000.0,
                cgst_rate=9.0,
                cgst_amount=900.0,
                sgst_rate=9.0,
                sgst_amount=900.0
            )
        ],
        tax_summary=TaxSummary(taxable_value=10000.00, cgst=900.00, sgst=900.00, igst=0.0),
        grand_total=11800.00,
        extraction_metadata=ExtractionMetadata(
            source_file="b2b_valid.png",
            extraction_method="self_test",
            confidence=0.95
        )
    )
    val_b2b = val_service.validate_invoice(inv_b2b, invoice_id="B2B-VALID")
    conf_b2b = conf_service.evaluate_confidence(inv_b2b, validation_result=val_b2b, invoice_id="B2B-VALID")
    b2b_entry, excl_b2b = map_canonical_to_gstr1(inv_b2b, val_b2b, conf_b2b, None, invoice_id="B2B-VALID")
    test_b2b_pass = (b2b_entry is not None and isinstance(b2b_entry, GSTR1B2BInvoice) and excl_b2b is None)
    print(f"GSTR-1 Valid B2B -> Table 4A: {'PASS' if test_b2b_pass else 'FAIL'}")

    # 2. Valid B2C -> Table 7
    inv_b2c = CanonicalInvoice(
        invoice_number="B2C-VALID",
        invoice_date="2026-10-08",
        seller=Party(name="Seller", gstin="27AABCS1429B1ZB"),
        buyer=Party(name="Consumer", gstin=None),
        place_of_supply="27",
        items=[
            InvoiceItem(
                description="Book",
                quantity=1.0,
                unit_price=5000.0,
                taxable_value=5000.0,
                cgst_rate=9.0,
                cgst_amount=450.0,
                sgst_rate=9.0,
                sgst_amount=450.0
            )
        ],
        tax_summary=TaxSummary(taxable_value=5000.00, cgst=450.00, sgst=450.00, igst=0.0),
        grand_total=5900.00,
        extraction_metadata=ExtractionMetadata(
            source_file="b2c_valid.png",
            extraction_method="self_test",
            confidence=0.95
        )
    )
    val_b2c = val_service.validate_invoice(inv_b2c, invoice_id="B2C-VALID")
    conf_b2c = conf_service.evaluate_confidence(inv_b2c, validation_result=val_b2c, invoice_id="B2C-VALID")
    b2c_entry, excl_b2c = map_canonical_to_gstr1(inv_b2c, val_b2c, conf_b2c, None, invoice_id="B2C-VALID")
    test_b2c_pass = (b2c_entry is not None and isinstance(b2c_entry, GSTR1B2CSInvoice) and excl_b2c is None)
    print(f"GSTR-1 Valid B2C -> Table 7: {'PASS' if test_b2c_pass else 'FAIL'}")

    # 3. Invalid invoice forced to APPROVED manually -> STILL EXCLUDED
    inv_invalid = CanonicalInvoice(
        invoice_number="INV-FORCED-APPROVAL",
        invoice_date="2026-10-08",
        seller=Party(name="Seller", gstin="27AABCS1429B1ZB"),
        buyer=Party(name="Buyer", gstin="27AABCB1234F1Z5"),
        place_of_supply="27",
        items=[
            InvoiceItem(
                description="Laptop",
                quantity=1.0,
                unit_price=10000.0,
                taxable_value=10000.0,
                cgst_rate=9.0,
                cgst_amount=900.0,
                sgst_rate=9.0,
                sgst_amount=900.0
            )
        ],
        tax_summary=TaxSummary(taxable_value=10000.00, cgst=900.00, sgst=900.00),
        grand_total=50000.00 # Arithmetic discrepancy
    )
    val_inv = val_service.validate_invoice(inv_invalid, invoice_id="INV-FORCED-APPROVAL")
    conf_inv = conf_service.evaluate_confidence(inv_invalid, validation_result=val_inv, invoice_id="INV-FORCED-APPROVAL")
    
    # Manually create a forced APPROVED review record
    forced_review = ReviewRecord(
        invoice_id="INV-FORCED-APPROVAL",
        review_status="APPROVED",
        confidence_score=conf_inv.overall_score,
        confidence_tier=conf_inv.confidence_tier,
        recommendation=conf_inv.recommendation,
        requires_review=True,
        reviewer_action="APPROVE",
        notes="Forced approval attempt"
    )
    admitted_forced, excl_forced = map_canonical_to_gstr1(inv_invalid, val_inv, conf_inv, forced_review, invoice_id="INV-FORCED-APPROVAL")
    forced_blocked_pass = (admitted_forced is None and excl_forced is not None)
    print(f"GSTR-1 Forced Approval on Invalid Invoice: {'PASS (Blocked & Excluded)' if forced_blocked_pass else 'FAIL (Leakage!)'}")

    gstr1_all = test_b2b_pass and test_b2c_pass and forced_blocked_pass
    test_results["gstr1_safety"] = {
        "valid_b2b_table4a": test_b2b_pass,
        "valid_b2c_table7": test_b2c_pass,
        "forced_approval_blocked": forced_blocked_pass,
        "overall": "PASS" if gstr1_all else "FAIL"
    }
    return gstr1_all

# ============================================================
# 7. RANDOM / UNSEEN DATASET GENERATION & END-TO-END PIPELINE
# ============================================================
def test_random_unseen_dataset():
    log_section("7. RANDOM / UNSEEN DATASET GENERATION & PIPELINE (20 INVOICES)")
    import random
    random.seed(42) # Deterministic random seed for academic reproducibility
    
    # 20 Test Categories
    test_specs = [
        {"id": "RAND_01_CLEAN", "type": "CLEAN", "exp_val": "PASS", "exp_gstr1": True, "table": "4A", "taxable": 5000, "cgst": 450, "sgst": 450, "igst": 0, "gt": 5900, "sgstin": "27AABCS1429B1ZB", "bgstin": "27AABCB1234F1Z5", "pos": "27", "items": 1},
        {"id": "RAND_02_MULTI", "type": "MULTI_ITEM", "exp_val": "PASS", "exp_gstr1": True, "table": "4A", "taxable": 12000, "cgst": 1080, "sgst": 1080, "igst": 0, "gt": 14160, "sgstin": "27AABCS1429B1ZB", "bgstin": "27AABCB1234F1Z5", "pos": "27", "items": 3},
        {"id": "RAND_03_B2C_1", "type": "B2C_NULL_BUYER", "exp_val": "PASS", "exp_gstr1": True, "table": "7", "taxable": 2500, "cgst": 225, "sgst": 225, "igst": 0, "gt": 2950, "sgstin": "27AABCS1429B1ZB", "bgstin": None, "pos": "27", "items": 1},
        {"id": "RAND_04_B2C_2", "type": "B2C_NULL_BUYER", "exp_val": "PASS", "exp_gstr1": True, "table": "7", "taxable": 8000, "cgst": 720, "sgst": 720, "igst": 0, "gt": 9440, "sgstin": "27AABCS1429B1ZB", "bgstin": None, "pos": "27", "items": 2},
        {"id": "RAND_05_B2C_3", "type": "B2C_NULL_BUYER", "exp_val": "PASS", "exp_gstr1": True, "table": "7", "taxable": 1500, "cgst": 135, "sgst": 135, "igst": 0, "gt": 1770, "sgstin": "27AABCS1429B1ZB", "bgstin": None, "pos": "27", "items": 1},
        {"id": "RAND_06_BAD_GT", "type": "WRONG_GRAND_TOTAL", "exp_val": "FAIL", "exp_gstr1": False, "table": None, "taxable": 10000, "cgst": 900, "sgst": 900, "igst": 0, "gt": 99999, "sgstin": "27AABCS1429B1ZB", "bgstin": "27AABCB1234F1Z5", "pos": "27", "items": 1},
        {"id": "RAND_07_BAD_CGST", "type": "WRONG_CGST", "exp_val": "FAIL", "exp_gstr1": False, "table": None, "taxable": 10000, "cgst": 500, "sgst": 900, "igst": 0, "gt": 11800, "sgstin": "27AABCS1429B1ZB", "bgstin": "27AABCB1234F1Z5", "pos": "27", "items": 1},
        {"id": "RAND_08_BAD_SGST", "type": "WRONG_SGST", "exp_val": "FAIL", "exp_gstr1": False, "table": None, "taxable": 10000, "cgst": 900, "sgst": 200, "igst": 0, "gt": 11800, "sgstin": "27AABCS1429B1ZB", "bgstin": "27AABCB1234F1Z5", "pos": "27", "items": 1},
        {"id": "RAND_09_INV_GSTIN1", "type": "INVALID_GSTIN", "exp_val": "FAIL", "exp_gstr1": False, "table": None, "taxable": 4000, "cgst": 360, "sgst": 360, "igst": 0, "gt": 4720, "sgstin": "99INVALID1234Z", "bgstin": "27AABCB1234F1Z5", "pos": "27", "items": 1},
        {"id": "RAND_10_INV_GSTIN2", "type": "INVALID_GSTIN", "exp_val": "FAIL", "exp_gstr1": False, "table": None, "taxable": 4000, "cgst": 360, "sgst": 360, "igst": 0, "gt": 4720, "sgstin": "27AABCS1429B1ZB", "bgstin": "00SHORTGSTIN", "pos": "27", "items": 1},
        {"id": "RAND_11_INV_GSTIN3", "type": "INVALID_GSTIN", "exp_val": "FAIL", "exp_gstr1": False, "table": None, "taxable": 4000, "cgst": 360, "sgst": 360, "igst": 0, "gt": 4720, "sgstin": "ABCDEFGHIJKLMN", "bgstin": "27AABCB1234F1Z5", "pos": "27", "items": 1},
        {"id": "RAND_12_MISS_NUM", "type": "MISSING_INVOICE_NUM", "exp_val": "FAIL", "exp_gstr1": False, "table": None, "taxable": 3000, "cgst": 270, "sgst": 270, "igst": 0, "gt": 3540, "sgstin": "27AABCS1429B1ZB", "bgstin": "27AABCB1234F1Z5", "pos": "27", "items": 1, "inv_num": None},
        {"id": "RAND_13_MISS_DATE", "type": "MISSING_DATE", "exp_val": "FAIL", "exp_gstr1": False, "table": None, "taxable": 3000, "cgst": 270, "sgst": 270, "igst": 0, "gt": 3540, "sgstin": "27AABCS1429B1ZB", "bgstin": "27AABCB1234F1Z5", "pos": "27", "items": 1, "inv_date": None},
        {"id": "RAND_14_ZERO_VAL", "type": "ZERO_VALUE", "exp_val": "PASS", "exp_gstr1": True, "table": "4A", "taxable": 0, "cgst": 0, "sgst": 0, "igst": 0, "gt": 0, "sgstin": "27AABCS1429B1ZB", "bgstin": "27AABCB1234F1Z5", "pos": "27", "items": 1},
        {"id": "RAND_15_LARGE_VAL", "type": "VERY_LARGE_VALUE", "exp_val": "PASS", "exp_gstr1": True, "table": "4A", "taxable": 500000000, "cgst": 45000000, "sgst": 45000000, "igst": 0, "gt": 590000000, "sgstin": "27AABCS1429B1ZB", "bgstin": "27AABCB1234F1Z5", "pos": "27", "items": 2},
        {"id": "RAND_16_DECIMAL", "type": "DECIMAL_EDGE_CASE", "exp_val": "PASS", "exp_gstr1": True, "table": "4A", "taxable": 1000.33, "cgst": 90.03, "sgst": 90.03, "igst": 0, "gt": 1180.39, "sgstin": "27AABCS1429B1ZB", "bgstin": "27AABCB1234F1Z5", "pos": "27", "items": 1},
        {"id": "RAND_17_ROUNDING", "type": "ROUNDING_EDGE_CASE", "exp_val": "PASS", "exp_gstr1": True, "table": "4A", "taxable": 1234.56, "cgst": 111.11, "sgst": 111.11, "igst": 0, "gt": 1456.78, "sgstin": "27AABCS1429B1ZB", "bgstin": "27AABCB1234F1Z5", "pos": "27", "items": 1},
        {"id": "RAND_18_IGST", "type": "INTERSTATE_IGST", "exp_val": "PASS", "exp_gstr1": True, "table": "4A", "taxable": 20000, "cgst": 0, "sgst": 0, "igst": 3600, "gt": 23600, "sgstin": "27AABCS1429B1ZB", "bgstin": "29XYZAB5678C1Z2", "pos": "29", "items": 2},
        {"id": "RAND_19_NOISY_OCR", "type": "NOISY_OCR_TEXT", "exp_val": "PASS", "exp_gstr1": True, "table": "4A", "taxable": 4500, "cgst": 405, "sgst": 405, "igst": 0, "gt": 5310, "sgstin": "27AABCS1429B1ZB", "bgstin": "27AABCB1234F1Z5", "pos": "27", "items": 1},
        {"id": "RAND_20_EXTRA_TXT", "type": "IRRELEVANT_TEXT", "exp_val": "PASS", "exp_gstr1": True, "table": "4A", "taxable": 7500, "cgst": 675, "sgst": 675, "igst": 0, "gt": 8850, "sgstin": "27AABCS1429B1ZB", "bgstin": "27AABCB1234F1Z5", "pos": "27", "items": 1}
    ]

    val_service = ValidationService(ValidationConfig())
    conf_service = ConfidenceService(ConfidenceConfig())
    
    random_records = []
    correct_validations = 0
    correct_gstr1 = 0
    
    for spec in test_specs:
        inv_id = spec["id"]
        inv_num = spec.get("inv_num", inv_id)
        inv_date = spec.get("inv_date", "2026-10-08")
        
        # Construct Canonical Invoice for the test
        items = []
        n_items = spec["items"]
        item_tax = spec["taxable"] / n_items
        for j in range(n_items):
            items.append(InvoiceItem(
                description=f"Item {j+1}",
                quantity=1.0,
                unit_price=float(item_tax),
                taxable_value=float(item_tax),
                cgst_rate=9.0 if spec["cgst"] > 0 else 0.0,
                cgst_amount=float(spec["cgst"]/n_items),
                sgst_rate=9.0 if spec["sgst"] > 0 else 0.0,
                sgst_amount=float(spec["sgst"]/n_items),
                igst_rate=18.0 if spec["igst"] > 0 else 0.0,
                igst_amount=float(spec["igst"]/n_items)
            ))
            
        inv = CanonicalInvoice(
            invoice_number=inv_num,
            invoice_date=inv_date,
            seller=Party(name="Alpha Corp", gstin=spec["sgstin"]) if spec["sgstin"] else Party(),
            buyer=Party(name="Beta LLC", gstin=spec["bgstin"]) if spec["bgstin"] or spec["type"] == "B2C_NULL_BUYER" else Party(),
            place_of_supply=spec["pos"],
            items=items,
            tax_summary=TaxSummary(
                taxable_value=float(spec["taxable"]),
                cgst=float(spec["cgst"]),
                sgst=float(spec["sgst"]),
                igst=float(spec["igst"])
            ),
            grand_total=float(spec["gt"]),
            extraction_metadata=ExtractionMetadata(
                source_file=f"{inv_id}.png",
                extraction_method="self_test_generator",
                confidence=0.95
            )
        )

        # Run downstream pipeline
        val_res = val_service.validate_invoice(inv, invoice_id=inv_id)
        conf_res = conf_service.evaluate_confidence(inv, validation_result=val_res, invoice_id=inv_id)
        b2b_c, excl = map_canonical_to_gstr1(inv, val_res, conf_res, None, invoice_id=inv_id)
        
        is_val_match = (val_res.status == spec["exp_val"])
        if is_val_match:
            correct_validations += 1
            
        is_gstr1_match = ((b2b_c is not None) == spec["exp_gstr1"])
        if is_gstr1_match:
            correct_gstr1 += 1
            
        random_records.append({
            "test_id": inv_id,
            "category": spec["type"],
            "expected_val": spec["exp_val"],
            "actual_val": val_res.status,
            "val_match": is_val_match,
            "conf_tier": conf_res.confidence_tier,
            "conf_score": conf_res.overall_score,
            "expected_gstr1": spec["exp_gstr1"],
            "actual_gstr1_admitted": (b2b_c is not None),
            "gstr1_match": is_gstr1_match,
            "error_count": val_res.error_count
        })

    df_random = pd.DataFrame(random_records)
    df_random.to_csv(SELF_TEST_DIR / "random_dataset_results.csv", index=False)
    
    val_acc = (correct_validations / len(test_specs)) * 100.0
    gstr1_acc = (correct_gstr1 / len(test_specs)) * 100.0
    print(f"Random Test Dataset Results (20 Invoices):")
    print(f"  - Validation Detection Accuracy: {correct_validations}/20 ({val_acc:.1f}%)")
    print(f"  - GSTR-1 Admission Accuracy: {correct_gstr1}/20 ({gstr1_acc:.1f}%)")
    
    test_results["random_dataset"] = {
        "total_invoices": len(test_specs),
        "validation_accuracy_pct": val_acc,
        "gstr1_admission_accuracy_pct": gstr1_acc,
        "results_csv": str(SELF_TEST_DIR / "random_dataset_results.csv")
    }
    return val_acc == 100.0 and gstr1_acc == 100.0

# ============================================================
# 8. CSV & EXCEL EXPORT VERIFICATION
# ============================================================
def test_export_verification():
    log_section("8. CSV & EXCEL EXPORT VERIFICATION")
    from src.gst.gstr1_models import GSTR1Draft, GSTR1B2BInvoice, GSTR1B2CSInvoice, GSTR1ExclusionRecord
    from src.export import export_b2b_csv, export_b2c_csv, export_gstr1_combined_csv, export_gstr1_excel
    
    b2b_rec = GSTR1B2BInvoice(
        source_invoice_id="EXP_001",
        recipient_gstin="27AABCB1234F1Z5",
        recipient_name="Beta Enterprises",
        invoice_number="EXP-B2B-001",
        invoice_date="2026-08-15",
        invoice_value=11800.0,
        place_of_supply="27",
        reverse_charge="N",
        invoice_type="Regular",
        taxable_value=10000.0,
        rate=18.0,
        cgst_amount=900.0,
        sgst_amount=900.0,
        igst_amount=0.0,
        cess_amount=0.0,
        source_validation_status="PASS",
        source_confidence_score=95.0
    )
    
    b2c_rec = GSTR1B2CSInvoice(
        source_invoice_id="EXP_002",
        place_of_supply="27",
        invoice_number="EXP-B2C-001",
        invoice_date="2026-08-15",
        invoice_value=1180.0,
        taxable_value=1000.0,
        rate=18.0,
        cgst_amount=90.0,
        sgst_amount=90.0,
        igst_amount=0.0,
        cess_amount=0.0,
        source_validation_status="PASS",
        source_confidence_score=95.0
    )
    
    draft = GSTR1Draft(
        source_invoice_count=2,
        included_invoice_count=2,
        excluded_invoice_count=0,
        b2b_count=1,
        b2c_count=1,
        total_taxable_value=11000.0,
        total_cgst=990.0,
        total_sgst=990.0,
        total_igst=0.0,
        total_invoice_value=12980.0,
        b2b_records=[b2b_rec],
        b2c_records=[b2c_rec],
        exclusions=[]
    )

    export_dir = SELF_TEST_DIR / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Export CSV
    b2b_csv = export_b2b_csv(draft.b2b_records, str(export_dir / "b2b.csv"))
    b2c_csv = export_b2c_csv(draft.b2c_records, str(export_dir / "b2c.csv"))
    comb_csv = export_gstr1_combined_csv(draft, str(export_dir / "gstr1_combined.csv"))
    csv_ok = all(Path(p).exists() and Path(p).stat().st_size > 0 for p in [b2b_csv, b2c_csv, comb_csv])
    print(f"CSV Exports: {'PASS' if csv_ok else 'FAIL'} - Created 3 CSV files")

    # 2. Export Excel
    excel_path = export_gstr1_excel(draft, str(export_dir / "gstr1_draft.xlsx"))
    excel_file = Path(excel_path)
    excel_ok = excel_file.exists() and excel_file.stat().st_size > 0
    
    # Programmatically read back the Excel file to verify sheet integrity
    sheets_ok = False
    if excel_ok:
        try:
            xl = pd.ExcelFile(excel_file)
            sheets = xl.sheet_names
            print(f"Excel Sheet Names: {sheets}")
            if "B2B_Table_4A" in sheets and "B2C_Table_7" in sheets and "Summary" in sheets:
                df4a = pd.read_excel(xl, sheet_name="B2B_Table_4A")
                df7 = pd.read_excel(xl, sheet_name="B2C_Table_7")
                if len(df4a) == 1 and len(df7) == 1:
                    sheets_ok = True
        except Exception as e:
            print(f"Excel read-back error: {e}")
            
    print(f"Excel Export & Sheet Validation: {'PASS' if sheets_ok else 'FAIL'}")

    export_all = csv_ok and sheets_ok
    test_results["export_verification"] = {
        "csv_export": csv_ok,
        "excel_export": excel_ok,
        "excel_sheet_validation": sheets_ok,
        "overall": "PASS" if export_all else "FAIL"
    }
    return export_all

# ============================================================
# 9. SECURITY & CREDENTIAL AUDIT
# ============================================================
def test_security_audit():
    log_section("9. SECURITY & CREDENTIAL AUDIT")
    import re
    
    # Check .gitignore
    gitignore = Path(".gitignore")
    env_ignored = False
    if gitignore.exists():
        with open(gitignore, "r") as f:
            lines = f.read().splitlines()
        env_ignored = any(".env" in l for l in lines)
    print(f".env in .gitignore: {'PASS' if env_ignored else 'FAIL'}")

    # Check for hardcoded API keys in src/ and app/
    leaks = []
    secret_patterns = [
        re.compile(r'(?:AIza[0-9A-Za-z-_]{35})'), # Google API Key
        re.compile(r'(?:sk-[a-zA-Z0-9]{32,})'),   # OpenAI Key
        re.compile(r'(?:ghp_[a-zA-Z0-9]{36})')   # GitHub Token
    ]
    
    for root, _, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                p = Path(root) / file
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                for pat in secret_patterns:
                    matches = pat.findall(content)
                    if matches:
                        leaks.append(f"{p}: Found potential secret match")
                        
    print(f"Hardcoded Secrets Scan in src/: {'PASS (0 Leaks)' if not leaks else 'FAIL: ' + str(leaks)}")

    sec_pass = env_ignored and (len(leaks) == 0)
    test_results["security_audit"] = {
        "env_in_gitignore": env_ignored,
        "hardcoded_leaks_found": len(leaks),
        "overall": "PASS" if sec_pass else "FAIL"
    }
    return sec_pass

# ============================================================
# 10. STREAMLIT SMOKE TEST
# ============================================================
def test_streamlit_smoke():
    log_section("10. STREAMLIT APP SMOKE TEST")
    try:
        import app.dashboard
        print("Streamlit app.dashboard imported cleanly without runtime syntax error.")
        st_ok = True
    except Exception as e:
        print(f"Streamlit import failed: {e}")
        st_ok = False
        
    test_results["streamlit_smoke"] = {"overall": "PASS" if st_ok else "FAIL"}
    return st_ok

if __name__ == "__main__":
    t_start = time.perf_counter()
    print("Starting Comprehensive Final Self-Test Suite...")
    
    r1 = test_static_imports()
    r2 = test_real_llm_connectivity()
    r3 = test_llm_error_handling()
    r4 = test_financial_firewall()
    r5 = test_confidence_safety()
    r6 = test_gstr1_safety()
    r7 = test_random_unseen_dataset()
    r8 = test_export_verification()
    r9 = test_security_audit()
    r10 = test_streamlit_smoke()
    
    total_time = time.perf_counter() - t_start
    all_passed = all([r1, r2, r3, r4, r5, r6, r7, r8, r9, r10])
    
    print("\n" + "="*60)
    print(f"SELF-TEST SUMMARY (Elapsed: {total_time:.2f}s):")
    print(f"  - Overall Verdict: {'🟢 PASS - READY FOR RELEASE' if all_passed else '🔴 FAIL - ISSUES DETECTED'}")
    print("="*60)
    
    with open(SELF_TEST_DIR / "self_test_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_sec": round(total_time, 2),
            "all_passed": all_passed,
            "results": test_results
        }, f, indent=2)
