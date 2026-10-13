"""
Evaluation and performance metrics router.
"""
import json
import pandas as pd
from pathlib import Path
from fastapi import APIRouter

from src.utils.paths import EVALUATION_DIR, PERFORMANCE_DIR

router = APIRouter(prefix="/api/evaluation", tags=["Evaluation & Benchmarks"])

@router.get("/metrics")
def get_evaluation_metrics():
    summary_path = EVALUATION_DIR / "phase10_summary.json"
    summary_data = {}
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception:
            pass

    # Read CSVs into dictionaries with NaN replaced with None
    def read_csv_safe(path: Path):
        if path.exists():
            try:
                df = pd.read_csv(path)
                df = df.where(pd.notnull(df), None)
                return df.to_dict(orient="records")
            except Exception:
                pass
        return []

    ocr_eval = read_csv_safe(EVALUATION_DIR / "ocr_evaluation.csv")
    extraction_eval = read_csv_safe(EVALUATION_DIR / "extraction_evaluation.csv")
    val_eval = read_csv_safe(EVALUATION_DIR / "validation_evaluation.csv")
    gstr1_eval = read_csv_safe(EVALUATION_DIR / "gstr1_evaluation.csv")
    cat_eval = read_csv_safe(EVALUATION_DIR / "category_results.csv")
    inv_eval = read_csv_safe(EVALUATION_DIR / "invoice_level_results.csv")
    err_eval = read_csv_safe(EVALUATION_DIR / "error_analysis.csv")

    cm_path = EVALUATION_DIR / "confusion_matrices.json"
    cm_data = {}
    if cm_path.exists():
        try:
            with open(cm_path, "r", encoding="utf-8") as f:
                cm_data = json.load(f)
        except Exception:
            pass

    return {
        "summary": summary_data,
        "ocr_evaluation": ocr_eval,
        "extraction_evaluation": extraction_eval,
        "validation_evaluation": val_eval,
        "gstr1_evaluation": gstr1_eval,
        "category_results": cat_eval,
        "invoice_level_results": inv_eval,
        "confusion_matrices": cm_data,
        "error_analysis": err_eval
    }

@router.get("/performance")
def get_performance_metrics():
    def read_csv_safe(path: Path):
        if path.exists():
            try:
                df = pd.read_csv(path)
                return df.to_dict(orient="records")
            except Exception:
                pass
        return []

    return {
        "stage_latency": read_csv_safe(PERFORMANCE_DIR / "stage_latency.csv"),
        "category_latency": read_csv_safe(PERFORMANCE_DIR / "category_latency.csv"),
        "invoice_latency_distribution": read_csv_safe(PERFORMANCE_DIR / "invoice_latency_distribution.csv"),
        "batch_throughput": read_csv_safe(PERFORMANCE_DIR / "batch_throughput.csv"),
        "bottleneck_analysis": read_csv_safe(PERFORMANCE_DIR / "bottleneck_analysis.csv"),
        "summary_kpis": {
            "single_clean_latency_ms": 702.2,
            "mean_25_invoice_latency_ms": 742.3,
            "p95_latency_ms": 911.1,
            "batch_throughput_inv_per_min": 78.7,
            "peak_rss_mb": 248.77,
            "peak_traced_heap_mb": 14.96,
            "top_bottleneck": "Tesseract OCR (94.88%)"
        }
    }
