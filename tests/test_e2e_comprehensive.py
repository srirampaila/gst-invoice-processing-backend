"""
Comprehensive End-to-End Test Suite for GST Intelligence React + FastAPI System.
Executes Phases A through G programmatically.
"""
import io
import json
import urllib.request
import urllib.error
import pytest
import pandas as pd
import openpyxl

FASTAPI_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://localhost:5173"
OLLAMA_URL = "http://127.0.0.1:11434"

def make_request(url, method="GET", data=None, headers=None, timeout=60):
    if headers is None:
        headers = {}
    req_data = json.dumps(data).encode("utf-8") if data is not None else None
    if req_data:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        content = response.read()
        return response.status, content

# =========================================================================
# PHASE A: ENVIRONMENT & SYSTEM READINESS
# =========================================================================
def test_phase_a_backend_health():
    status, body = make_request(f"{FASTAPI_URL}/api/health")
    assert status == 200
    data = json.loads(body.decode("utf-8"))
    assert data["status"] == "READY"
    assert data["tesseract_available"] is True
    assert data["ollama_ready"] is True
    assert "qwen2.5:1.5b" in data["installed_ollama_models"]
    assert data["canonical_invoice_count"] == 25

def test_phase_a_frontend_reachable():
    status, body = make_request(f"{FRONTEND_URL}/")
    assert status == 200
    html = body.decode("utf-8")
    assert "<!doctype html>" in html.lower()
    assert "GST Intelligence" in html

def test_phase_a_ollama_tags():
    status, body = make_request(f"{OLLAMA_URL}/api/tags")
    assert status == 200
    data = json.loads(body.decode("utf-8"))
    model_names = [m["name"] for m in data.get("models", [])]
    assert any("qwen2.5:1.5b" in m for m in model_names)

# =========================================================================
# PHASE B: FRONTEND ROUTES & ASSET VERIFICATION
# =========================================================================
@pytest.mark.parametrize("route", [
    "/",
    "/studio",
    "/extraction",
    "/validation",
    "/review",
    "/gstr1",
    "/evaluation",
    "/performance",
    "/architecture"
])
def test_phase_b_routes_accessible(route):
    status, body = make_request(f"{FRONTEND_URL}{route}")
    assert status == 200
    html = body.decode("utf-8")
    assert "root" in html
    assert "/src/main.tsx" in html

# =========================================================================
# PHASE C: CLEAN INVOICE EXTRACTION & DETERMINISTIC VALIDATION
# =========================================================================
def test_phase_c_clean_invoice_mock_and_ollama():
    # 1. Test Mock Provider on INV_CLEAN_001.png
    payload_mock = {
        "file_name": "INV_CLEAN_001.png",
        "provider": "mock",
        "amount_tolerance": 1.0,
        "high_threshold": 85.0,
        "medium_threshold": 60.0
    }
    status, body = make_request(f"{FASTAPI_URL}/api/invoices/process", method="POST", data=payload_mock)
    assert status == 200
    res = json.loads(body.decode("utf-8"))
    
    inv = res["canonical_invoice"]
    val = res["validation"]
    conf = res["confidence"]
    gstr1 = res["gstr1"]

    # Strict ground truth verification
    assert inv["invoice_number"] == "INV_CLEAN_001"
    assert inv["invoice_date"] == "2026-08-15"
    assert inv["seller"]["gstin"] == "29ABCDE1234F1Z5"
    assert inv["buyer"]["gstin"] == "27XYZAB5678C1Z2"
    assert inv["tax_summary"]["taxable_value"] == 1000.0
    assert inv["tax_summary"]["cgst"] == 90.0
    assert inv["tax_summary"]["sgst"] == 90.0
    assert inv["grand_total"] == 1180.0

    # Validation & Confidence checks
    assert val["status"] == "PASS"
    assert val["error_count"] == 0
    assert conf["confidence_tier"] == "HIGH"
    assert conf["recommendation"] == "AUTO_ACCEPT"
    assert gstr1["is_admitted"] is True
    assert gstr1["table_type"] == "Table 4A (B2B)"

    # 2. Test Real Local Ollama Qwen2.5 1.5B Extraction
    payload_ollama = {
        "file_name": "INV_CLEAN_001.png",
        "provider": "ollama",
        "amount_tolerance": 1.0
    }
    status_ol, body_ol = make_request(f"{FASTAPI_URL}/api/invoices/process", method="POST", data=payload_ollama)
    assert status_ol == 200
    res_ol = json.loads(body_ol.decode("utf-8"))
    assert res_ol["provider_used"] == "ollama"
    assert res_ol["model_used"] == "qwen2.5:1.5b"
    assert res_ol["canonical_invoice"]["invoice_number"] is not None

# =========================================================================
# PHASE D: INVALID INVOICE & SAFETY GATING (TOTAL-001)
# =========================================================================
def test_phase_d_invalid_invoice_safety_gating():
    payload = {
        "file_name": "INV_INV_001.png",
        "provider": "mock",
        "amount_tolerance": 1.0
    }
    status, body = make_request(f"{FASTAPI_URL}/api/invoices/process", method="POST", data=payload)
    assert status == 200
    res = json.loads(body.decode("utf-8"))

    val = res["validation"]
    c_vals = val["calculated_values"]
    conf = res["confidence"]
    gstr1 = res["gstr1"]

    # Verify TOTAL-001 discrepancy is caught
    assert val["status"] == "FAIL"
    assert val["error_count"] >= 1
    assert any(issue["rule_id"] == "TOTAL-001" for issue in val["issues"])
    assert c_vals["grand_total_discrepancy"] == 500.0  # Declared 1680 vs Recomputed 1180

    # Verify GSTR-1 exclusion
    assert gstr1["is_admitted"] is False
    assert gstr1["exclusion"] is not None
    assert "TOTAL-001" in gstr1["exclusion"]["reason"]

# =========================================================================
# PHASE E: HUMAN REVIEW ACTION & INVARIANT PERSISTENCE
# =========================================================================
def test_phase_e_human_review_and_invariant_bypass_prevention():
    # 1. Apply approval on invalid invoice INV_INV_001
    rev_payload = {
        "action": "APPROVE",
        "notes": "Attempting auditor override on invalid arithmetic invoice"
    }
    status, body = make_request(f"{FASTAPI_URL}/api/reviews/INV_INV_001/action", method="POST", data=rev_payload)
    assert status == 200
    data = json.loads(body.decode("utf-8"))

    assert data["review_status"] == "APPROVED"
    assert data["reviewer_action"] == "APPROVE"
    # CRITICAL INVARIANT: GSTR-1 must REMAIN FALSE despite human approval because TOTAL-001 is a fatal math error
    assert data["gstr1_is_admitted"] is False
    assert "TOTAL-001" in data["gstr1_exclusion_reason"]

    # 2. Verify Review State Persistence
    status_get, body_get = make_request(f"{FASTAPI_URL}/api/reviews/INV_INV_001")
    assert status_get == 200
    persisted = json.loads(body_get.decode("utf-8"))
    assert persisted["review_status"] == "APPROVED"
    assert persisted["notes"] == "Attempting auditor override on invalid arithmetic invoice"

# =========================================================================
# PHASE F: GSTR-1 DRAFT & FILE EXPORT DELIVERIES
# =========================================================================
def test_phase_f_gstr1_draft_and_exports():
    # 1. Verify Draft Payload
    status, body = make_request(f"{FASTAPI_URL}/api/gstr1/draft")
    assert status == 200
    draft = json.loads(body.decode("utf-8"))
    assert draft["source_invoice_count"] == 25
    assert draft["included_invoice_count"] == 15
    assert draft["excluded_invoice_count"] == 10
    assert draft["b2b_count"] == 12
    assert draft["b2c_count"] == 3
    assert draft["total_taxable_value"] == 40000.0
    assert draft["total_invoice_value"] == 47200.0

    # 2. Download and verify Combined CSV
    status_csv, body_csv = make_request(f"{FASTAPI_URL}/api/gstr1/exports/combined_csv")
    assert status_csv == 200
    df_csv = pd.read_csv(io.BytesIO(body_csv))
    assert "Invoice Number" in df_csv.columns or "invoice_number" in df_csv.columns
    assert len(df_csv) == 15  # 12 B2B + 3 B2C

    # 3. Download and verify Table 4A CSV
    status_b2b, body_b2b = make_request(f"{FASTAPI_URL}/api/gstr1/exports/b2b_csv")
    assert status_b2b == 200
    df_b2b = pd.read_csv(io.BytesIO(body_b2b))
    assert len(df_b2b) == 12

    # 4. Download and verify Table 7 B2CS CSV
    status_b2c, body_b2c = make_request(f"{FASTAPI_URL}/api/gstr1/exports/b2c_csv")
    assert status_b2c == 200
    df_b2c = pd.read_csv(io.BytesIO(body_b2c))
    assert len(df_b2c) == 3

    # 5. Download and verify Exclusions CSV
    status_exc, body_exc = make_request(f"{FASTAPI_URL}/api/gstr1/exports/exclusions_csv")
    assert status_exc == 200
    df_exc = pd.read_csv(io.BytesIO(body_exc))
    assert len(df_exc) == 10

    # 6. Download and verify Multi-Sheet Excel (.xlsx)
    status_xlsx, body_xlsx = make_request(f"{FASTAPI_URL}/api/gstr1/exports/excel")
    assert status_xlsx == 200
    wb = openpyxl.load_workbook(io.BytesIO(body_xlsx))
    sheet_names = wb.sheetnames
    assert "Summary" in sheet_names or "GSTR-1 Summary" in sheet_names or len(sheet_names) >= 4
