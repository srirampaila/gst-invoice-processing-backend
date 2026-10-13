"""
Real LLM Batch Extraction & Empirical Ground Truth Evaluation.
Executes extraction via live LLM provider (Google Gemini or OpenAI) and compares performance against ground truth and mock baseline.
"""
import os
import sys
import json
import csv
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv()

from src.extraction.service import ExtractionService
from src.extraction.config import ExtractionConfig
from src.extraction.models import CanonicalInvoice
from src.extraction.parser import parse_and_validate_json
from src.extraction.exceptions import LLMProviderError

OCR_PROC_DIR = "data/outputs/ocr/preprocessed"
GT_RAW_DIR = "data/raw/invoices"
OUT_BASE_DIR = "data/outputs/extraction/real_llm"
MOCK_CSV_PATH = "data/outputs/extraction/evaluation_results.csv"

RAW_RESPONSES_DIR = os.path.join(OUT_BASE_DIR, "raw_responses")
JSON_DIR = os.path.join(OUT_BASE_DIR, "json")
METADATA_DIR = os.path.join(OUT_BASE_DIR, "metadata")

os.makedirs(RAW_RESPONSES_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

NUMERIC_TOLERANCE = 0.01

def is_numeric_close(actual, expected, tolerance=NUMERIC_TOLERANCE) -> bool:
    if actual is None or expected is None:
        return actual == expected
    try:
        return abs(float(actual) - float(expected)) <= tolerance
    except (ValueError, TypeError):
        return False

def run_real_llm_batch():
    config = ExtractionConfig()
    provider_name = config.provider.capitalize()

    # Verify key based on provider
    if config.provider == "gemini":
        key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        if not key:
            print("\n" + "=" * 60)
            print("ERROR: Real Gemini API Key is not configured.")
            print("Set GEMINI_API_KEY in your local .env file.")
            print("=" * 60 + "\n")
            sys.exit(1)
    elif config.provider == "openai":
        key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        if not key:
            print("\n" + "=" * 60)
            print("ERROR: Real OpenAI API Key is not configured.")
            print("Set OPENAI_API_KEY in your local .env file.")
            print("=" * 60 + "\n")
            sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print(f"ERROR: Real LLM provider '{config.provider}' requires credentials.")
        print("Set GEMINI_API_KEY or OPENAI_API_KEY in your local .env file.")
        print("=" * 60 + "\n")
        sys.exit(1)

    try:
        service = ExtractionService(config)
    except LLMProviderError as e:
        print(f"Error initializing {provider_name} Client: {e}")
        sys.exit(1)

    ocr_files = sorted([f for f in os.listdir(OCR_PROC_DIR) if f.endswith(".txt")])
    if not ocr_files:
        print(f"No OCR text files found in {OCR_PROC_DIR}")
        return

    print("=" * 60)
    print(f"STARTING REAL LLM BATCH EXTRACTION (Provider: {provider_name} | Model: {config.model})")
    print("=" * 60)

    evaluation_rows = []
    total_attempted = len(ocr_files)
    successful_requests = 0
    failed_requests = 0
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
    warnings_list = []

    for idx, txt_file in enumerate(ocr_files, 1):
        base_name = txt_file.replace(".txt", "")
        ocr_path = os.path.join(OCR_PROC_DIR, txt_file)
        gt_path = os.path.join(GT_RAW_DIR, f"{base_name}.json")

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_json = json.load(f)

        ocr_text = Path(ocr_path).read_text(encoding="utf-8")

        start_time = time.time()
        raw_llm_response = ""
        error_msg = None

        print(f"[{idx:02d}/25] Processing {base_name} via {provider_name}...", end=" ", flush=True)

        try:
            # Call live LLM
            raw_llm_response = service.client.generate_extraction(ocr_text)
            invoice: CanonicalInvoice = parse_and_validate_json(
                raw_response=raw_llm_response,
                source_file=txt_file,
                extraction_method=config.provider,
                model=config.model
            )
            latency = time.time() - start_time
            latencies.append(latency)
            successful_requests += 1
            schema_valid = True
            schema_valid_count += 1
            print(f"OK ({latency:.2f}s)")
        except Exception as e:
            latency = time.time() - start_time
            latencies.append(latency)
            failed_requests += 1
            schema_valid = False
            error_msg = str(e)
            invoice = CanonicalInvoice()
            print(f"FAILED ({error_msg[:40]}...)")

        # Save Raw LLM response string
        raw_resp_path = os.path.join(RAW_RESPONSES_DIR, f"{base_name}.txt")
        with open(raw_resp_path, "w", encoding="utf-8") as f:
            f.write(raw_llm_response)

        # Save Canonical JSON
        json_out_path = os.path.join(JSON_DIR, f"{base_name}.json")
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(invoice.model_dump(), f, indent=2)

        # Save Metadata
        meta_dict = invoice.extraction_metadata.model_dump()
        meta_dict["execution_latency_seconds"] = round(latency, 4)
        meta_dict["api_success"] = (error_msg is None)
        meta_dict["error"] = error_msg
        meta_out_path = os.path.join(METADATA_DIR, f"{base_name}.json")
        with open(meta_out_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

        conf = invoice.extraction_metadata.confidence
        confidences.append(conf)
        warnings_list.extend(invoice.extraction_metadata.warnings)

        # Ground Truth Evaluations
        inv_num_expected = gt_json.get("invoice_number")
        inv_num_correct = (invoice.invoice_number == inv_num_expected)
        field_counts["invoice_number"]["expected"] += 1
        if inv_num_correct:
            field_counts["invoice_number"]["correct"] += 1

        date_expected = gt_json.get("invoice_date")
        date_correct = (invoice.invoice_date == date_expected)
        field_counts["invoice_date"]["expected"] += 1
        if date_correct:
            field_counts["invoice_date"]["correct"] += 1

        seller_gstin_expected = gt_json.get("seller", {}).get("gstin")
        seller_correct = (invoice.seller.gstin == seller_gstin_expected)
        field_counts["seller_gstin"]["expected"] += 1
        if seller_correct:
            field_counts["seller_gstin"]["correct"] += 1

        buyer_gstin_expected = gt_json.get("buyer", {}).get("gstin")
        if buyer_gstin_expected:
            buyer_correct = (invoice.buyer.gstin == buyer_gstin_expected)
            field_counts["buyer_gstin"]["expected"] += 1
            if buyer_correct:
                field_counts["buyer_gstin"]["correct"] += 1
        else:
            buyer_correct = (invoice.buyer.gstin is None)

        tv_expected = gt_json.get("totals", {}).get("taxable_value")
        tv_correct = is_numeric_close(invoice.tax_summary.taxable_value, tv_expected)
        field_counts["taxable_value"]["expected"] += 1
        if tv_correct:
            field_counts["taxable_value"]["correct"] += 1

        cgst_expected = gt_json.get("totals", {}).get("cgst")
        cgst_correct = is_numeric_close(invoice.tax_summary.cgst, cgst_expected)
        field_counts["cgst"]["expected"] += 1
        if cgst_correct:
            field_counts["cgst"]["correct"] += 1

        sgst_expected = gt_json.get("totals", {}).get("sgst")
        sgst_correct = is_numeric_close(invoice.tax_summary.sgst, sgst_expected)
        field_counts["sgst"]["expected"] += 1
        if sgst_correct:
            field_counts["sgst"]["correct"] += 1

        igst_expected = gt_json.get("totals", {}).get("igst", 0.0)
        if igst_expected > 0:
            igst_correct = is_numeric_close(invoice.tax_summary.igst, igst_expected)
            field_counts["igst"]["expected"] += 1
            if igst_correct:
                field_counts["igst"]["correct"] += 1
        else:
            igst_correct = True

        gt_expected = gt_json.get("totals", {}).get("grand_total")
        gt_correct = is_numeric_close(invoice.grand_total, gt_expected)
        field_counts["grand_total"]["expected"] += 1
        if gt_correct:
            field_counts["grand_total"]["correct"] += 1

        expected_items = len(gt_json.get("items", []))
        actual_items = len(invoice.items)
        items_correct = (actual_items == expected_items)
        field_counts["line_items"]["expected"] += 1
        if items_correct:
            field_counts["line_items"]["correct"] += 1

        evaluation_rows.append({
            "invoice_id": base_name,
            "provider": provider_name,
            "model": config.model,
            "success": error_msg is None,
            "schema_valid": schema_valid,
            "latency": round(latency, 4),
            "confidence": round(conf, 2),
            "invoice_number_correct": inv_num_correct,
            "invoice_date_correct": date_correct,
            "seller_gstin_correct": seller_correct,
            "buyer_gstin_correct": buyer_correct,
            "taxable_value_correct": tv_correct,
            "cgst_correct": cgst_correct,
            "sgst_correct": sgst_correct,
            "igst_correct": igst_correct,
            "grand_total_correct": gt_correct,
            "line_items_correct": items_correct,
            "warnings": len(invoice.extraction_metadata.warnings),
            "error": error_msg or ""
        })

    # Save real LLM evaluation CSV
    real_csv_path = "data/outputs/extraction/real_llm_evaluation_results.csv"
    fieldnames = [
        "invoice_id", "provider", "model", "success", "schema_valid", "latency", "confidence",
        "invoice_number_correct", "invoice_date_correct", "seller_gstin_correct",
        "buyer_gstin_correct", "taxable_value_correct", "cgst_correct", "sgst_correct",
        "igst_correct", "grand_total_correct", "line_items_correct", "warnings", "error"
    ]
    with open(real_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evaluation_rows)

    # Generate Mock vs Real Comparison CSV
    mock_stats = {}
    if os.path.exists(MOCK_CSV_PATH):
        with open(MOCK_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            mock_rows = list(reader)
            if mock_rows:
                mock_stats = {
                    "schema_valid": sum(1 for r in mock_rows if r["schema_valid"] == "True") / len(mock_rows) * 100,
                    "inv_num": sum(1 for r in mock_rows if r["invoice_number_correct"] == "True") / len(mock_rows) * 100,
                    "date": sum(1 for r in mock_rows if r["date_correct"] == "True") / len(mock_rows) * 100,
                    "seller_gstin": sum(1 for r in mock_rows if r["seller_gstin_correct"] == "True") / len(mock_rows) * 100,
                    "buyer_gstin": sum(1 for r in mock_rows if r["buyer_gstin_correct"] == "True") / len(mock_rows) * 100,
                    "taxable_value": sum(1 for r in mock_rows if r["taxable_value_correct"] == "True") / len(mock_rows) * 100,
                    "cgst": sum(1 for r in mock_rows if r["cgst_correct"] == "True") / len(mock_rows) * 100,
                    "sgst": sum(1 for r in mock_rows if r["sgst_correct"] == "True") / len(mock_rows) * 100,
                    "grand_total": sum(1 for r in mock_rows if r["grand_total_correct"] == "True") / len(mock_rows) * 100,
                    "line_items": sum(1 for r in mock_rows if r["item_count_match"] == "True") / len(mock_rows) * 100,
                    "avg_confidence": np.mean([float(r["confidence"]) for r in mock_rows]),
                    "avg_latency": np.mean([float(r["latency_seconds"]) for r in mock_rows])
                }

    comp_metrics = [
        ("Schema Validation (%)", f"{mock_stats.get('schema_valid', 100.0):.1f}%", f"{schema_valid_count/total_attempted*100:.1f}%"),
        ("Invoice Number Accuracy (%)", f"{mock_stats.get('inv_num', 76.0):.1f}%", f"{field_counts['invoice_number']['correct']/field_counts['invoice_number']['expected']*100:.1f}%"),
        ("Invoice Date Accuracy (%)", f"{mock_stats.get('date', 100.0):.1f}%", f"{field_counts['invoice_date']['correct']/field_counts['invoice_date']['expected']*100:.1f}%"),
        ("Seller GSTIN Accuracy (%)", f"{mock_stats.get('seller_gstin', 72.0):.1f}%", f"{field_counts['seller_gstin']['correct']/field_counts['seller_gstin']['expected']*100:.1f}%"),
        ("Buyer GSTIN Accuracy (%)", f"{mock_stats.get('buyer_gstin', 68.2):.1f}%", f"{field_counts['buyer_gstin']['correct']/max(field_counts['buyer_gstin']['expected'],1)*100:.1f}%"),
        ("Taxable Value Accuracy (%)", f"{mock_stats.get('taxable_value', 84.0):.1f}%", f"{field_counts['taxable_value']['correct']/field_counts['taxable_value']['expected']*100:.1f}%"),
        ("CGST Accuracy (%)", f"{mock_stats.get('cgst', 72.0):.1f}%", f"{field_counts['cgst']['correct']/field_counts['cgst']['expected']*100:.1f}%"),
        ("SGST Accuracy (%)", f"{mock_stats.get('sgst', 60.0):.1f}%", f"{field_counts['sgst']['correct']/field_counts['sgst']['expected']*100:.1f}%"),
        ("Grand Total Accuracy (%)", f"{mock_stats.get('grand_total', 84.0):.1f}%", f"{field_counts['grand_total']['correct']/field_counts['grand_total']['expected']*100:.1f}%"),
        ("Line Item Completeness (%)", f"{mock_stats.get('line_items', 76.0):.1f}%", f"{field_counts['line_items']['correct']/field_counts['line_items']['expected']*100:.1f}%"),
        ("Average Confidence", f"{mock_stats.get('avg_confidence', 0.89):.2f}", f"{np.mean(confidences):.2f}"),
        ("Average Latency (s)", f"{mock_stats.get('avg_latency', 0.0004):.4f}s", f"{np.mean(latencies):.4f}s")
    ]

    comp_csv_path = "data/outputs/extraction/mock_vs_real_comparison.csv"
    with open(comp_csv_path, "w", newline="", encoding="utf-8") as f:
        comp_writer = csv.writer(f)
        comp_writer.writerow(["Metric", "Mock (Deterministic)", f"Real LLM ({provider_name})"])
        for row in comp_metrics:
            comp_writer.writerow(row)

    # Print Summary
    print("\n" + "=" * 60)
    print(f"REAL LLM EXTRACTION EVALUATION REPORT ({provider_name} - {config.model})")
    print("=" * 60)
    print(f"Total Invoices Attempted:  {total_attempted}")
    print(f"Successful API Requests:   {successful_requests}/{total_attempted} ({successful_requests/total_attempted*100:.1f}%)")
    print(f"Schema Valid JSON Rate:    {schema_valid_count}/{total_attempted} ({schema_valid_count/total_attempted*100:.1f}%)")
    print(f"Latency: Mean={np.mean(latencies):.3f}s | Median={np.median(latencies):.3f}s | Min={np.min(latencies):.3f}s | Max={np.max(latencies):.3f}s")
    print(f"Average Confidence:        {np.mean(confidences):.2f}")
    print("\nField Accuracies vs Ground Truth:")
    for field, cnt in field_counts.items():
        if cnt["expected"] > 0:
            pct = cnt["correct"] / cnt["expected"] * 100
            print(f"  {field:16s}: {cnt['correct']:2d}/{cnt['expected']:2d} ({pct:5.1f}%)")

if __name__ == "__main__":
    run_real_llm_batch()
