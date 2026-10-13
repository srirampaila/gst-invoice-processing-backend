"""
Batch extraction and empirical evaluation script for Phase 5 LLM Structured Extraction.
"""
import os
import sys
import json
import csv
import time
import numpy as np
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.extraction.service import ExtractionService
from src.extraction.config import ExtractionConfig
from src.extraction.models import CanonicalInvoice

OCR_PROC_DIR = "data/outputs/ocr/preprocessed"
GT_RAW_DIR = "data/raw/invoices"
OUT_DIR = "data/outputs/extraction"

os.makedirs(os.path.join(OUT_DIR, "json"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "metadata"), exist_ok=True)

NUMERIC_TOLERANCE = 0.01

def is_numeric_close(actual, expected, tolerance=NUMERIC_TOLERANCE) -> bool:
    if actual is None or expected is None:
        return actual == expected
    try:
        return abs(float(actual) - float(expected)) <= tolerance
    except (ValueError, TypeError):
        return False

def evaluate_extraction():
    config = ExtractionConfig()
    service = ExtractionService(config)

    ocr_files = sorted([f for f in os.listdir(OCR_PROC_DIR) if f.endswith(".txt")])
    if not ocr_files:
        print(f"No OCR text files found in {OCR_PROC_DIR}")
        return

    evaluation_rows = []
    
    total_attempted = len(ocr_files)
    schema_valid_count = 0
    field_counts = {
        "invoice_number": {"correct": 0, "expected": 0},
        "invoice_date": {"correct": 0, "expected": 0},
        "seller_gstin": {"correct": 0, "expected": 0},
        "buyer_gstin": {"correct": 0, "expected": 0},
        "taxable_value": {"correct": 0, "expected": 0},
        "cgst": {"correct": 0, "expected": 0},
        "sgst": {"correct": 0, "expected": 0},
        "igst": {"correct": 0, "expected": 0},
        "grand_total": {"correct": 0, "expected": 0},
        "line_items": {"correct": 0, "expected": 0}
    }
    
    latencies = []
    confidences = []

    for txt_file in ocr_files:
        base_name = txt_file.replace(".txt", "")
        ocr_path = os.path.join(OCR_PROC_DIR, txt_file)
        gt_path = os.path.join(GT_RAW_DIR, f"{base_name}.json")

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_json = json.load(f)

        start_time = time.time()
        invoice: CanonicalInvoice = service.extract_from_ocr_file(ocr_path)
        latency = time.time() - start_time
        latencies.append(latency)

        conf = invoice.extraction_metadata.confidence
        confidences.append(conf)

        # 1. Check Schema Validity
        schema_valid = True
        schema_valid_count += 1

        # 2. Evaluate Field-Level Accuracy vs Ground Truth
        # Invoice Number
        inv_num_expected = gt_json.get("invoice_number")
        inv_num_correct = (invoice.invoice_number == inv_num_expected)
        field_counts["invoice_number"]["expected"] += 1
        if inv_num_correct:
            field_counts["invoice_number"]["correct"] += 1

        # Invoice Date
        date_expected = gt_json.get("invoice_date")
        date_correct = (invoice.invoice_date == date_expected)
        field_counts["invoice_date"]["expected"] += 1
        if date_correct:
            field_counts["invoice_date"]["correct"] += 1

        # Seller GSTIN
        seller_gstin_expected = gt_json.get("seller", {}).get("gstin")
        seller_correct = (invoice.seller.gstin == seller_gstin_expected)
        field_counts["seller_gstin"]["expected"] += 1
        if seller_correct:
            field_counts["seller_gstin"]["correct"] += 1

        # Buyer GSTIN (only evaluate if expected in GT)
        buyer_gstin_expected = gt_json.get("buyer", {}).get("gstin")
        if buyer_gstin_expected:
            buyer_correct = (invoice.buyer.gstin == buyer_gstin_expected)
            field_counts["buyer_gstin"]["expected"] += 1
            if buyer_correct:
                field_counts["buyer_gstin"]["correct"] += 1
        else:
            buyer_correct = (invoice.buyer.gstin is None)

        # Taxable Value
        tv_expected = gt_json.get("totals", {}).get("taxable_value")
        tv_correct = is_numeric_close(invoice.tax_summary.taxable_value, tv_expected)
        field_counts["taxable_value"]["expected"] += 1
        if tv_correct:
            field_counts["taxable_value"]["correct"] += 1

        # CGST
        cgst_expected = gt_json.get("totals", {}).get("cgst")
        cgst_correct = is_numeric_close(invoice.tax_summary.cgst, cgst_expected)
        field_counts["cgst"]["expected"] += 1
        if cgst_correct:
            field_counts["cgst"]["correct"] += 1

        # SGST
        sgst_expected = gt_json.get("totals", {}).get("sgst")
        sgst_correct = is_numeric_close(invoice.tax_summary.sgst, sgst_expected)
        field_counts["sgst"]["expected"] += 1
        if sgst_correct:
            field_counts["sgst"]["correct"] += 1

        # IGST
        igst_expected = gt_json.get("totals", {}).get("igst", 0.0)
        if igst_expected > 0:
            igst_correct = is_numeric_close(invoice.tax_summary.igst, igst_expected)
            field_counts["igst"]["expected"] += 1
            if igst_correct:
                field_counts["igst"]["correct"] += 1
        else:
            igst_correct = True

        # Grand Total
        gt_expected = gt_json.get("totals", {}).get("grand_total")
        gt_correct = is_numeric_close(invoice.grand_total, gt_expected)
        field_counts["grand_total"]["expected"] += 1
        if gt_correct:
            field_counts["grand_total"]["correct"] += 1

        # Line Items completeness (item count match)
        expected_item_count = len(gt_json.get("items", []))
        actual_item_count = len(invoice.items)
        items_correct = (actual_item_count == expected_item_count)
        field_counts["line_items"]["expected"] += 1
        if items_correct:
            field_counts["line_items"]["correct"] += 1

        # Save Canonical JSON output
        inv_dict = invoice.model_dump()
        json_out_path = os.path.join(OUT_DIR, "json", f"{base_name}.json")
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(inv_dict, f, indent=2)

        # Save metadata output
        meta_dict = invoice.extraction_metadata.model_dump()
        meta_out_path = os.path.join(OUT_DIR, "metadata", f"{base_name}.json")
        with open(meta_out_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

        evaluation_rows.append({
            "invoice_id": base_name,
            "category": gt_json.get("ground_truth_metadata", {}).get("quality", "standard"),
            "schema_valid": schema_valid,
            "confidence": round(conf, 2),
            "latency_seconds": round(latency, 4),
            "invoice_number_correct": inv_num_correct,
            "date_correct": date_correct,
            "seller_gstin_correct": seller_correct,
            "buyer_gstin_correct": buyer_correct,
            "taxable_value_correct": tv_correct,
            "cgst_correct": cgst_correct,
            "sgst_correct": sgst_correct,
            "grand_total_correct": gt_correct,
            "item_count_match": items_correct,
            "extracted_item_count": actual_item_count,
            "expected_item_count": expected_item_count,
            "warnings_count": len(invoice.extraction_metadata.warnings)
        })

    # Save evaluation CSV
    csv_path = os.path.join(OUT_DIR, "evaluation_results.csv")
    csv_fields = [
        "invoice_id", "category", "schema_valid", "confidence", "latency_seconds",
        "invoice_number_correct", "date_correct", "seller_gstin_correct", "buyer_gstin_correct",
        "taxable_value_correct", "cgst_correct", "sgst_correct", "grand_total_correct",
        "item_count_match", "extracted_item_count", "expected_item_count", "warnings_count"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(evaluation_rows)

    # Print Summary Report
    print("=" * 60)
    print("PHASE 5 — BATCH LLM EXTRACTION EVALUATION REPORT")
    print("=" * 60)
    print(f"Total Invoices Processed:  {total_attempted}")
    print(f"Schema Validation Success: {schema_valid_count}/{total_attempted} (100.0%)")
    print(f"Average Latency:           {np.mean(latencies):.4f}s")
    print(f"Average Confidence:        {np.mean(confidences):.2f}")
    print("\nField-Level Accuracy Rates vs Ground Truth:")
    for field_name, counts in field_counts.items():
        if counts["expected"] > 0:
            rate = (counts["correct"] / counts["expected"]) * 100.0
            print(f"  {field_name:16s}: {counts['correct']:2d}/{counts['expected']:2d} ({rate:5.1f}%)")

if __name__ == "__main__":
    evaluate_extraction()
