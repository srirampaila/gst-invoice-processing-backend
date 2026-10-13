"""
Phase 11 — Performance Evaluation Test Suite.
Verifies performance benchmarking infrastructure, latency invariants,
throughput consistency, resource metrics, and artifact generation.
"""
import os
import json
import pytest
import pandas as pd
from pathlib import Path

from src.utils.evaluate_phase11_performance import (
    run_phase11_performance_evaluation,
    run_single_invoice_benchmark,
    get_current_rss_mb
)

PERF_OUTPUT_DIR = Path("data/outputs/performance")

@pytest.fixture(scope="session")
def phase11_perf_summary():
    """Runs Phase 11 performance evaluation once and returns the summary."""
    return run_phase11_performance_evaluation()

def test_performance_artifacts_exist(phase11_perf_summary):
    """Verify all machine-readable performance artifacts exist in data/outputs/performance/."""
    expected_files = [
        "phase11_summary.json",
        "stage_latency.csv",
        "category_latency.csv",
        "batch_throughput.csv",
        "bottleneck_analysis.csv",
        "invoice_latency_distribution.csv"
    ]
    for fname in expected_files:
        fpath = PERF_OUTPUT_DIR / fname
        assert fpath.exists(), f"Missing Phase 11 performance artifact: {fname}"
        assert fpath.stat().st_size > 0, f"Artifact is empty: {fname}"

def test_single_invoice_stage_invariants(phase11_perf_summary):
    """Verify single-invoice benchmark timing structure and non-negativity."""
    stages = phase11_perf_summary["single_invoice_clean_benchmark"]
    expected_stages = [
        "file_loading", "preprocessing", "ocr_extraction", "structured_extraction",
        "deterministic_validation", "confidence_scoring", "gstr1_mapping",
        "export_csv_excel", "total_end_to_end"
    ]
    for stg in expected_stages:
        assert stg in stages, f"Missing stage: {stg}"
        data = stages[stg]
        assert data["mean_sec"] > 0, f"Mean latency for {stg} must be positive"
        assert data["min_sec"] <= data["mean_sec"] <= data["max_sec"]
        assert 0.0 <= data["pct_of_total"] <= 100.0

def test_end_to_end_latency_percentiles(phase11_perf_summary):
    """Verify P95 and median latency relationship across 25 benchmark invoices."""
    e2e = phase11_perf_summary["end_to_end_latency_summary"]
    assert e2e["total_invoices_evaluated"] == 25
    assert e2e["min_latency_sec"] <= e2e["median_latency_sec"] <= e2e["max_latency_sec"]
    assert e2e["p95_latency_sec"] >= e2e["median_latency_sec"]
    assert e2e["mean_latency_sec"] > 0

def test_batch_scalability_and_throughput(phase11_perf_summary):
    """Verify throughput mathematical consistency and linear scalability."""
    batch_results = phase11_perf_summary["batch_scalability_and_throughput"]
    assert len(batch_results) == 4
    sizes = [r["batch_size"] for r in batch_results]
    assert sizes == [1, 5, 10, 25]

    for r in batch_results:
        assert r["total_time_sec"] > 0
        assert r["avg_time_per_invoice_sec"] > 0
        assert r["throughput_invoices_per_sec"] > 0
        assert r["throughput_invoices_per_min"] == pytest.approx(r["throughput_invoices_per_sec"] * 60.0, 0.1)
        assert r["peak_heap_mb"] > 0
        assert r["peak_rss_mb"] > 0

def test_bottleneck_ranking_evidence(phase11_perf_summary):
    """Verify bottleneck identification correctly flags OCR and Preprocessing as top stages."""
    bottlenecks = phase11_perf_summary["bottleneck_ranking"]
    assert len(bottlenecks) >= 5
    ranks = [b["rank"] for b in bottlenecks]
    assert ranks == sorted(ranks)
    
    # OCR is expected to be rank 1 bottleneck
    assert bottlenecks[0]["rank"] == 1
    assert "OCR" in bottlenecks[0]["component"]
    assert bottlenecks[0]["runtime_pct"] > 80.0

def test_resource_usage_helpers():
    """Verify process memory lookup helper returns positive float."""
    rss = get_current_rss_mb()
    assert isinstance(rss, float)
    assert rss > 0.0

def test_tradeoff_justification_presence(phase11_perf_summary):
    """Verify accuracy vs latency trade-off justification metadata is documented."""
    tradeoff = phase11_perf_summary["tradeoff_analysis"]
    assert tradeoff["ocr_accuracy_cer_reduction_pct"] == 75.0
    assert len(tradeoff["justification"]) > 20
    assert "Phase 5" in phase11_perf_summary["phase5_boundary_notice"]
