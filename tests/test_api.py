"""
Integration tests for FastAPI bridge endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "docs_url" in data

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "READY"
    assert "tesseract_available" in data
    assert "canonical_invoice_count" in data
    assert data["canonical_invoice_count"] >= 25

def test_providers_endpoint():
    response = client.get("/api/providers")
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    prov_ids = [p["id"] for p in data["providers"]]
    assert "mock" in prov_ids
    assert "ollama" in prov_ids

def test_sample_invoices_endpoint():
    response = client.get("/api/invoices/samples")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) >= 25
    first = samples[0]
    assert "file_name" in first
    assert "category" in first
    assert "image_url" in first

def test_process_clean_invoice():
    req_body = {
        "file_name": "INV_CLEAN_001.png",
        "provider": "mock",
        "amount_tolerance": 1.0,
        "high_threshold": 85.0,
        "medium_threshold": 60.0
    }
    response = client.post("/api/invoices/process", json=req_body)
    assert response.status_code == 200
    res = response.json()
    assert res["invoice_id"] == "INV_CLEAN_001"
    assert res["validation"]["status"] == "PASS"
    assert res["confidence"]["confidence_tier"] == "HIGH"
    assert res["gstr1"]["is_admitted"] is True
    assert res["gstr1"]["table_type"] == "Table 4A (B2B)"

def test_process_invalid_invoice():
    req_body = {
        "file_name": "INV_INV_001.png",
        "provider": "mock",
        "amount_tolerance": 1.0
    }
    response = client.post("/api/invoices/process", json=req_body)
    assert response.status_code == 200
    res = response.json()
    assert res["invoice_id"] == "INV_INV_001"
    assert res["validation"]["status"] == "FAIL"
    assert res["gstr1"]["is_admitted"] is False
    assert res["gstr1"]["exclusion"] is not None

def test_gstr1_draft_endpoint():
    response = client.get("/api/gstr1/draft")
    assert response.status_code == 200
    data = response.json()
    assert data["source_invoice_count"] == 25
    assert data["included_invoice_count"] == 15
    assert data["excluded_invoice_count"] == 10
    assert len(data["b2b_records"]) == 12
    assert len(data["b2c_records"]) == 3

def test_evaluation_metrics_endpoint():
    response = client.get("/api/evaluation/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "ocr_evaluation" in data
    assert "validation_evaluation" in data

def test_performance_metrics_endpoint():
    response = client.get("/api/evaluation/performance")
    assert response.status_code == 200
    data = response.json()
    assert "summary_kpis" in data
    assert "stage_latency" in data

def test_architecture_summary_endpoint():
    response = client.get("/api/architecture/summary")
    assert response.status_code == 200
    data = response.json()
    assert "principles" in data
    assert "stages" in data
    assert "viva_qa" in data

def test_missing_invoice_404():
    req_body = {
        "file_name": "NON_EXISTENT_INVOICE_999.png",
        "provider": "mock"
    }
    response = client.post("/api/invoices/process", json=req_body)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

def test_cors_options_request():
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }
    response = client.options("/api/invoices/process", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert response.headers.get("access-control-allow-credentials") == "true"

def test_environment_validator():
    from src.utils.config_validator import validate_environment
    res = validate_environment()
    assert "status" in res
    assert "diagnostics" in res
    assert "warnings" in res

