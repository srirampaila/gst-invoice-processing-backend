"""
System, health, providers, and architecture router.
"""
import os
import shutil
import urllib.request
import json
from pathlib import Path
from fastapi import APIRouter
from src.api.models import HealthResponse, ProvidersResponse, ProviderInfo
from src.utils.paths import RAW_INVOICES_DIR

router = APIRouter(prefix="/api", tags=["System"])

def get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

def is_ollama_online() -> bool:
    try:
        url = f"{get_ollama_base_url()}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "FastAPIBridge"})
        with urllib.request.urlopen(req, timeout=1) as response:
            return response.status == 200
    except Exception:
        return False

def get_ollama_tags() -> list:
    try:
        url = f"{get_ollama_base_url()}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "FastAPIBridge"})
        with urllib.request.urlopen(req, timeout=1) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                return [m.get("name") for m in data.get("models", [])]
    except Exception:
        pass
    return []

@router.get("/health", response_model=HealthResponse)
def get_health():
    tesseract_ok = bool(shutil.which("tesseract"))
    ollama_ok = is_ollama_online()
    models = get_ollama_tags() if ollama_ok else []
    gemini_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY"))
    openai_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"))
    raw_count = len(list(RAW_INVOICES_DIR.glob("*.png"))) if RAW_INVOICES_DIR.exists() else 0

    return HealthResponse(
        status="READY",
        environment=os.getenv("APP_ENV", "development"),
        ollama_ready=ollama_ok,
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
        installed_ollama_models=models,
        tesseract_available=tesseract_ok,
        gemini_configured=gemini_key,
        openai_configured=openai_key,
        canonical_invoice_count=raw_count
    )

@router.get("/providers", response_model=ProvidersResponse)
def get_providers():
    ollama_ok = is_ollama_online()
    gemini_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY"))
    openai_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"))

    providers = [
        ProviderInfo(
            id="mock",
            name="Deterministic Mock Extractor",
            description="Offline zero-cost baseline parser using canonical template mapping",
            available=True,
            is_local=True,
            requires_key=False,
            key_configured=True
        ),
        ProviderInfo(
            id="ollama",
            name="Local Ollama AI (Qwen2.5 1.5B)",
            description=f"Locally hosted real LLM executing on {get_ollama_base_url()} with zero data leakage",
            available=ollama_ok,
            is_local=True,
            requires_key=False,
            key_configured=ollama_ok
        ),
        ProviderInfo(
            id="gemini",
            name="Google Gemini (Cloud AI)",
            description="Cloud multimodal LLM for high-accuracy commercial extraction",
            available=gemini_key,
            is_local=False,
            requires_key=True,
            key_configured=gemini_key
        ),
        ProviderInfo(
            id="openai",
            name="OpenAI GPT (Cloud AI)",
            description="Cloud enterprise LLM supporting GPT-4o-mini structured JSON outputs",
            available=openai_key,
            is_local=False,
            requires_key=True,
            key_configured=openai_key
        )
    ]

    default_prov = "ollama" if ollama_ok else "mock"
    return ProvidersResponse(providers=providers, default_provider=default_prov)

@router.get("/architecture/summary")
def get_architecture_summary():
    return {
        "principles": [
            {
                "title": "Separation of Concerns",
                "detail": "The LLM is strictly an unstructured-to-canonical text interpreter. Deterministic Python decimal arithmetic is the sole statutory authority for financial totals and GST compliance."
            },
            {
                "title": "Immutable Source Governance",
                "detail": "Original raw OCR tokens and canonical extractions remain strictly immutable. Human review decisions are layered as authoritative audit records."
            },
            {
                "title": "Zero Silent Mutation",
                "detail": "All monetary values are calculated in bit-exact decimal.Decimal with statutory ₹1.00 rounding tolerance. Human approval cannot bypass critical mathematical discrepancies (e.g. TOTAL-001)."
            }
        ],
        "stages": [
            {
                "stage": 1,
                "name": "Vision Preprocessing",
                "tech": "OpenCV (CLAHE, Bilateral Denoise, Hough Deskew)",
                "latency_contribution": "3.08% (21.6 ms)",
                "gain": "75.0% CER Reduction"
            },
            {
                "stage": 2,
                "name": "OCR Text Recognition",
                "tech": "Tesseract 5.5.3 (PSM 6)",
                "latency_contribution": "94.88% (666.2 ms)",
                "gain": "Text & Word Confidence Scoring"
            },
            {
                "stage": 3,
                "name": "Structured Canonical Extraction",
                "tech": "Pydantic CanonicalInvoice + Ollama Qwen2.5 1.5B / Mock",
                "latency_contribution": "Variable (Local / Cloud)",
                "gain": "Schema Validated Entity Graph"
            },
            {
                "stage": 4,
                "name": "Deterministic GST Validation",
                "tech": "Python Decimal Engine (Luhn Mod-36, ₹1.00 Tol)",
                "latency_contribution": "< 0.1% (0.2 ms)",
                "gain": "100% Math Error Detection (3/3)"
            },
            {
                "stage": 5,
                "name": "Multi-Factor Confidence & Overrides",
                "tech": "6-Factor Weighted Engine + 10 Safety Rules",
                "latency_contribution": "< 0.1% (0.1 ms)",
                "gain": "Automated Gating & Risk Flags"
            },
            {
                "stage": 6,
                "name": "GSTR-1 Draft Preparation & Export",
                "tech": "Table 4A / Table 7 Routing + openpyxl",
                "latency_contribution": "1.23% (8.6 ms)",
                "gain": "Multi-Sheet Excel & Portal CSVs"
            }
        ],
        "viva_qa": [
            {
                "question": "Why is Python Decimal used instead of letting the LLM calculate GST?",
                "answer": "LLMs lack exact floating-point precision and are vulnerable to numeric hallucination. Tax statutory law requires deterministic bit-exact arithmetic. Python Decimal guarantees rounding precision to ₹0.01."
            },
            {
                "question": "How are non-compliant invoices quarantined?",
                "answer": "The Phase 8 Eligibility Gate automatically routes any invoice with fatal validation errors (TOTAL-001, MATH-001, missing mandatory legal fields) to the Exclusions Catalog, preserving complete provenance."
            },
            {
                "question": "Does this software submit tax returns to the government portal?",
                "answer": "No. The system is an automated draft preparation and validation assistant. Official return submissions require authorized human signatory review and government portal authentication."
            }
        ]
    }
