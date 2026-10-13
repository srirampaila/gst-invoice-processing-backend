"""
Environment variable validation and deployment diagnostics.
Reports missing or misconfigured settings without exposing secrets.
"""
import os
import shutil
from decimal import Decimal, InvalidOperation
from typing import Dict, Any, List
from src.utils.logger import get_logger

logger = get_logger("deployment.config")

def validate_environment() -> Dict[str, Any]:
    """
    Validates environment variables for backend deployment.
    Logs diagnostics safely without leaking credentials.
    Returns a status dictionary.
    """
    warnings: List[str] = []
    diagnostics: Dict[str, Any] = {}

    app_env = os.getenv("APP_ENV", "development").lower()
    diagnostics["app_env"] = app_env

    # 1. OCR Binary Availability
    tesseract_bin = shutil.which("tesseract")
    diagnostics["tesseract_available"] = bool(tesseract_bin)
    if not tesseract_bin:
        warnings.append("Tesseract OCR binary not found in system PATH. Image-based OCR extraction will fail.")

    # 2. LLM Provider Configuration
    provider = os.getenv("LLM_PROVIDER", "").lower().strip()
    if not provider:
        if os.getenv("GEMINI_API_KEY"):
            provider = "gemini"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "mock"
    diagnostics["resolved_provider"] = provider

    gemini_key_set = bool(os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY"))
    openai_key_set = bool(os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"))
    diagnostics["gemini_key_configured"] = gemini_key_set
    diagnostics["openai_key_configured"] = openai_key_set

    if provider == "gemini" and not gemini_key_set:
        warnings.append("LLM_PROVIDER is configured as 'gemini' but neither GEMINI_API_KEY nor LLM_API_KEY is set.")
    elif provider == "openai" and not openai_key_set:
        warnings.append("LLM_PROVIDER is configured as 'openai' but neither OPENAI_API_KEY nor LLM_API_KEY is set.")
    elif provider in ("ollama", "local"):
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        diagnostics["ollama_base_url"] = ollama_url

    # 3. Numeric Tolerances Validation
    env_tol = os.getenv("VALIDATION_AMOUNT_TOLERANCE")
    if env_tol:
        try:
            val = Decimal(env_tol)
            if val < 0:
                warnings.append(f"VALIDATION_AMOUNT_TOLERANCE is negative ({val}); falling back to default 1.00.")
        except InvalidOperation:
            warnings.append(f"VALIDATION_AMOUNT_TOLERANCE '{env_tol}' is not a valid number; falling back to default 1.00.")

    env_pct = os.getenv("VALIDATION_PERCENTAGE_TOLERANCE")
    if env_pct:
        try:
            val_pct = Decimal(env_pct)
            if val_pct < 0:
                warnings.append(f"VALIDATION_PERCENTAGE_TOLERANCE is negative ({val_pct}); falling back to default 0.10.")
        except InvalidOperation:
            warnings.append(f"VALIDATION_PERCENTAGE_TOLERANCE '{env_pct}' is not a valid number; falling back to default 0.10.")

    # 4. CORS Origins Configuration
    raw_cors = os.getenv("CORS_ORIGINS", "")
    if raw_cors.strip():
        origins = [o.strip() for o in raw_cors.split(",") if o.strip()]
        diagnostics["cors_origins_count"] = len(origins)
    else:
        diagnostics["cors_origins_count"] = "default_local"

    # Log summary safely
    logger.info(
        f"Deployment Configuration: APP_ENV={app_env}, Provider={provider}, "
        f"Tesseract={'OK' if tesseract_bin else 'MISSING'}, Warnings={len(warnings)}"
    )
    for w in warnings:
        logger.warning(f"Config Warning: {w}")

    return {
        "status": "HEALTHY" if not warnings else "WARNINGS_PRESENT",
        "warnings": warnings,
        "diagnostics": diagnostics,
    }
