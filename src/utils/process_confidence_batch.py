"""
Batch Processing and Evaluation Script for Phase 7 Confidence Scoring & Human Review Workflow.
Evaluates all 25 Canonical Invoices and deterministic ValidationResults to produce:
1. Individual ConfidenceResult JSONs
2. Default PENDING_REVIEW ReviewRecords
3. Comprehensive evaluation CSV report
4. Aggregate ConfidenceSummary JSON
"""
import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.extraction.models import CanonicalInvoice
from src.validation.models import ValidationResult
from src.confidence.config import ConfidenceConfig
from src.confidence.models import (
    ConfidenceResult,
    ConfidenceSummary,
    ReviewRecord,
    ConfidenceTier,
    Recommendation
)
from src.confidence.service import ConfidenceService
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXTRACTION_JSON_DIR = "data/outputs/extraction/json"
VALIDATION_RESULTS_DIR = "data/outputs/validation/results"
OCR_METADATA_DIR = "data/outputs/ocr/metadata"
OUT_BASE_DIR = "data/outputs/confidence"
OUT_RESULTS_DIR = os.path.join(OUT_BASE_DIR, "results")
OUT_REVIEWS_DIR = os.path.join(OUT_BASE_DIR, "reviews")
OUT_CSV_PATH = os.path.join(OUT_BASE_DIR, "evaluation_results.csv")
OUT_SUMMARY_PATH = os.path.join(OUT_BASE_DIR, "summary.json")

os.makedirs(OUT_RESULTS_DIR, exist_ok=True)
os.makedirs(OUT_REVIEWS_DIR, exist_ok=True)

def run_confidence_batch() -> ConfidenceSummary:
    config = ConfidenceConfig()
    service = ConfidenceService(config)

    json_files = sorted([f for f in os.listdir(EXTRACTION_JSON_DIR) if f.endswith(".json")])
    if not json_files:
        print(f"No extraction JSON files found in {EXTRACTION_JSON_DIR}")
        return ConfidenceSummary()

    print("=" * 75)
    print("STARTING DETERMINISTIC CONFIDENCE SCORING & REVIEW BATCH (PHASE 7)")
    print(f"Weights: OCR={config.ocr_weight} | FieldComp={config.field_completeness_weight} | Crit={config.critical_fields_weight} | GST={config.gst_validation_weight} | Fin={config.financial_validation_weight} | Items={config.line_item_weight}")
    print(f"Thresholds: High={config.high_threshold} | Medium={config.medium_threshold}")
    print("=" * 75)

    confidence_results: List[ConfidenceResult] = []
    review_records: List[ReviewRecord] = []
    csv_rows = []

    high_count = 0
    medium_count = 0
    low_count = 0

    auto_accept_count = 0
    review_required_count = 0
    reprocess_reject_count = 0

    scores = []
    category_metrics: Dict[str, Dict[str, Any]] = {}

    for idx, fname in enumerate(json_files, 1):
        inv_id = fname.replace(".json", "")
        ext_path = os.path.join(EXTRACTION_JSON_DIR, fname)
        val_path = os.path.join(VALIDATION_RESULTS_DIR, fname)
        ocr_path = os.path.join(OCR_METADATA_DIR, f"{inv_id}_preprocessed.json")
        if not os.path.exists(ocr_path):
            ocr_path = os.path.join(OCR_METADATA_DIR, f"{inv_id}_original.json")

        # Load Canonical Invoice
        with open(ext_path, "r", encoding="utf-8") as f:
            ext_data = json.load(f)
        invoice = CanonicalInvoice.model_validate(ext_data)

        # Load Validation Result
        val_res = None
        if os.path.exists(val_path):
            with open(val_path, "r", encoding="utf-8") as f:
                val_data = json.load(f)
            val_res = ValidationResult.model_validate(val_data)

        # Load OCR Metadata
        ocr_meta = None
        if os.path.exists(ocr_path):
            with open(ocr_path, "r", encoding="utf-8") as f:
                ocr_meta = json.load(f)

        # Run Confidence Evaluation
        conf_res = service.evaluate_confidence(
            invoice=invoice,
            validation_result=val_res,
            ocr_metadata=ocr_meta,
            invoice_id=inv_id
        )
        confidence_results.append(conf_res)
        scores.append(conf_res.overall_score)

        # Create Initial Review Record
        rev_record = service.create_review_record(
            confidence_result=conf_res,
            validation_result=val_res
        )
        review_records.append(rev_record)

        # Save Individual ConfidenceResult JSON
        res_json_path = os.path.join(OUT_RESULTS_DIR, f"{inv_id}.json")
        with open(res_json_path, "w", encoding="utf-8") as f:
            json.dump(conf_res.model_dump(), f, indent=2)

        # Save ReviewRecord JSON
        rev_json_path = os.path.join(OUT_REVIEWS_DIR, f"{inv_id}.json")
        with open(rev_json_path, "w", encoding="utf-8") as f:
            json.dump(rev_record.model_dump(), f, indent=2)

        # Track Tier counts
        if conf_res.confidence_tier == ConfidenceTier.HIGH.value:
            high_count += 1
        elif conf_res.confidence_tier == ConfidenceTier.MEDIUM.value:
            medium_count += 1
        else:
            low_count += 1

        # Track Recommendation counts
        if conf_res.recommendation == Recommendation.AUTO_ACCEPT.value:
            auto_accept_count += 1
        elif conf_res.recommendation == Recommendation.REVIEW_REQUIRED.value:
            review_required_count += 1
        else:
            reprocess_reject_count += 1

        # Determine Category
        category = inv_id.split("_")[1] if "_" in inv_id else "UNKNOWN"
        if category not in category_metrics:
            category_metrics[category] = {"count": 0, "scores": [], "tiers": {}, "recs": {}}
        category_metrics[category]["count"] += 1
        category_metrics[category]["scores"].append(conf_res.overall_score)
        category_metrics[category]["tiers"][conf_res.confidence_tier] = category_metrics[category]["tiers"].get(conf_res.confidence_tier, 0) + 1
        category_metrics[category]["recs"][conf_res.recommendation] = category_metrics[category]["recs"].get(conf_res.recommendation, 0) + 1

        # Print log row
        status_sym = "🟢" if conf_res.confidence_tier == "HIGH" else ("🟡" if conf_res.confidence_tier == "MEDIUM" else "🔴")
        print(f"[{idx:02d}/25] {inv_id:<15} -> {status_sym} {conf_res.confidence_tier:<6} ({conf_res.overall_score:5.1f}/100) | Rec: {conf_res.recommendation:<20} | Overrides: {len(conf_res.triggered_rules)}")

        # Build CSV row
        factors = conf_res.factors
        csv_rows.append({
            "invoice_id": inv_id,
            "overall_score": conf_res.overall_score,
            "confidence_tier": conf_res.confidence_tier,
            "recommendation": conf_res.recommendation,
            "requires_review": conf_res.requires_review,
            "validation_status": val_res.status if val_res else "N/A",
            "error_count": val_res.error_count if val_res else 0,
            "warning_count": val_res.warning_count if val_res else 0,
            "ocr_score": factors.get("ocr_quality").score if "ocr_quality" in factors else 0.0,
            "field_comp_score": factors.get("field_completeness").score if "field_completeness" in factors else 0.0,
            "critical_fields_score": factors.get("critical_fields").score if "critical_fields" in factors else 0.0,
            "gst_score": factors.get("gst_validation").score if "gst_validation" in factors else 0.0,
            "fin_score": factors.get("financial_validation").score if "financial_validation" in factors else 0.0,
            "items_score": factors.get("line_item_quality").score if "line_item_quality" in factors else 0.0,
            "triggered_rules": "; ".join(r.rule_id for r in conf_res.triggered_rules) if conf_res.triggered_rules else "NONE",
            "explanation": conf_res.explanation
        })

    # Write CSV report
    fieldnames = [
        "invoice_id", "overall_score", "confidence_tier", "recommendation", "requires_review",
        "validation_status", "error_count", "warning_count", "ocr_score", "field_comp_score",
        "critical_fields_score", "gst_score", "fin_score", "items_score", "triggered_rules", "explanation"
    ]
    with open(OUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    total_invoices = len(json_files)
    avg_score = round(sum(scores) / total_invoices, 2) if total_invoices > 0 else 0.0
    min_score = round(min(scores), 2) if scores else 0.0
    max_score = round(max(scores), 2) if scores else 0.0

    # Build Summary Model
    summary = ConfidenceSummary(
        total_invoices=total_invoices,
        high_count=high_count,
        medium_count=medium_count,
        low_count=low_count,
        auto_accept_count=auto_accept_count,
        review_required_count=review_required_count,
        reprocess_reject_count=reprocess_reject_count,
        average_confidence_score=avg_score,
        min_confidence_score=min_score,
        max_confidence_score=max_score,
        validation_error_correlation={
            "clean_invoices": category_metrics.get("CLEAN", {}),
            "duplicate_invoices": category_metrics.get("DUP", {}),
            "invalid_math_invoices": category_metrics.get("INV", {}),
            "missing_fields_invoices": category_metrics.get("MISS", {}),
            "multi_item_invoices": category_metrics.get("MULTI", {}),
            "noisy_invoices": category_metrics.get("NOISY", {}),
            "rotated_invoices": category_metrics.get("ROT", {})
        }
    )

    with open(OUT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary.model_dump(), f, indent=2)

    print("\n" + "=" * 75)
    print("PHASE 7 CONFIDENCE EVALUATION SUMMARY")
    print("=" * 75)
    print(f"Total Invoices Evaluated:       {total_invoices}")
    print(f"HIGH Tier:                      {high_count} ({round(high_count/total_invoices*100, 1)}%)")
    print(f"MEDIUM Tier:                    {medium_count} ({round(medium_count/total_invoices*100, 1)}%)")
    print(f"LOW Tier:                       {low_count} ({round(low_count/total_invoices*100, 1)}%)")
    print(f"AUTO_ACCEPT Decisions:          {auto_accept_count} ({round(auto_accept_count/total_invoices*100, 1)}%)")
    print(f"REVIEW_REQUIRED Decisions:      {review_required_count} ({round(review_required_count/total_invoices*100, 1)}%)")
    print(f"REJECT_OR_REPROCESS Decisions:  {reprocess_reject_count} ({round(reprocess_reject_count/total_invoices*100, 1)}%)")
    print(f"Score Stats (Avg / Min / Max):  {avg_score} / {min_score} / {max_score}")
    print(f"Outputs Saved To:               {OUT_BASE_DIR}")
    print("=" * 75 + "\n")

    return summary

if __name__ == "__main__":
    run_confidence_batch()
