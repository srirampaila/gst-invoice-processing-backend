"""
Centralized path definitions and storage directory management.
Supports optional DATA_DIR environment variable for persistent volume mounts.
"""
import os
from pathlib import Path

# Base data directory: defaults to 'data' in repository root, or configurable via DATA_DIR
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

RAW_INVOICES_DIR = DATA_DIR / "raw" / "invoices"
TEMP_PROCESSED_DIR = DATA_DIR / "processed" / "temp"

# Output directories
OUTPUTS_DIR = DATA_DIR / "outputs"
REVIEWS_DIR = OUTPUTS_DIR / "confidence" / "reviews"
CONFIDENCE_RESULTS_DIR = OUTPUTS_DIR / "confidence" / "results"
VALIDATION_RESULTS_DIR = OUTPUTS_DIR / "validation" / "results"
EXTRACTION_JSON_DIR = OUTPUTS_DIR / "extraction" / "json"
GSTR1_DIR = OUTPUTS_DIR / "gstr1"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
PERFORMANCE_DIR = OUTPUTS_DIR / "performance"

ALL_RUNTIME_DIRS = [
    RAW_INVOICES_DIR,
    TEMP_PROCESSED_DIR,
    REVIEWS_DIR,
    CONFIDENCE_RESULTS_DIR,
    VALIDATION_RESULTS_DIR,
    EXTRACTION_JSON_DIR,
    GSTR1_DIR,
    EVALUATION_DIR,
    PERFORMANCE_DIR,
]

def ensure_data_directories() -> None:
    """Ensure all required runtime data and output directories exist."""
    for d in ALL_RUNTIME_DIRS:
        d.mkdir(parents=True, exist_ok=True)
