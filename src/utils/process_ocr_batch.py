import os
import json
import time
import csv
import numpy as np
import Levenshtein
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ocr.config import OCRConfig
from src.ocr.service import extract_text
from src.ocr.exceptions import OCREngineNotAvailableError
from src.ocr.engine import TesseractEngine

RAW_DIR = "data/raw/invoices"
PROC_DIR = "data/processed/final"
OUT_DIR = "data/outputs/ocr"

os.makedirs(os.path.join(OUT_DIR, "original"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "preprocessed"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "metadata"), exist_ok=True)

def generate_expected_text(gt_json):
    # Generates a deterministic expected text from structured ground truth
    text = f"INVOICE NUMBER: {gt_json['invoice_number']}\n"
    text += f"DATE: {gt_json['invoice_date']}\n"
    text += f"SELLER GSTIN: {gt_json['seller'].get('gstin', '')}\n"
    
    buyer_gstin = gt_json['buyer'].get('gstin')
    if buyer_gstin:
        text += f"BUYER GSTIN: {buyer_gstin}\n"
        
    for i, item in enumerate(gt_json.get('items', [])):
        text += f"ITEM {i}: {item['taxable_value']} (Taxable), CGST: {item['cgst']}, SGST: {item['sgst']}\n"
        
    totals = gt_json.get('totals', {})
    text += f"TOTAL TAXABLE: {totals.get('taxable_value')}\n"
    text += f"CGST: {totals.get('cgst')}\n"
    text += f"SGST: {totals.get('sgst')}\n"
    if totals.get('igst', 0) > 0:
        text += f"IGST: {totals.get('igst')}\n"
    text += f"GRAND TOTAL: {totals.get('grand_total')}\n"
    return text.strip()

def calculate_metrics(expected, actual):
    expected_clean = " ".join(expected.split())
    actual_clean = " ".join(actual.split())
    
    if not expected_clean:
        return 0.0, 0.0
        
    cer = Levenshtein.distance(expected_clean, actual_clean) / max(len(expected_clean), 1)
    
    exp_words = expected_clean.split()
    act_words = actual_clean.split()
    wer = Levenshtein.distance(exp_words, act_words) / max(len(exp_words), 1)
    
    return float(cer), float(wer)

def field_presence(actual, gt_json):
    # Evaluates presence only for fields expected in ground truth
    fields = {
        "invoice_number": gt_json["invoice_number"] in actual,
        "invoice_date": gt_json["invoice_date"] in actual,
        "seller_gstin": (gt_json["seller"].get("gstin", "") in actual) if gt_json["seller"].get("gstin") else None,
        "buyer_gstin": (gt_json["buyer"].get("gstin", "") in actual) if gt_json["buyer"].get("gstin") else None,
        "taxable_value": str(gt_json["totals"].get("taxable_value")) in actual,
        "cgst": str(gt_json["totals"].get("cgst")) in actual,
        "sgst": str(gt_json["totals"].get("sgst")) in actual,
        "igst": (str(gt_json["totals"].get("igst")) in actual) if (gt_json["totals"].get("igst", 0) > 0) else None,
        "grand_total": str(gt_json["totals"].get("grand_total")) in actual
    }
    return fields

def compute_stats(values):
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "total": 0.0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "total": float(np.sum(values))
    }

def process_batch():
    config = OCRConfig()
    try:
        TesseractEngine(config)
    except OCREngineNotAvailableError as e:
        print(f"Error: {e}")
        return None
        
    images = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".png")])
    
    metrics = {
        "original": {"cer": [], "wer": [], "conf": [], "fields": [], "times": [], "success": 0, "attempted": 0},
        "preprocessed": {"cer": [], "wer": [], "conf": [], "fields": [], "times": [], "success": 0, "attempted": 0}
    }
    
    per_invoice_rows = []
    
    for img_file in images:
        base_name = img_file.replace(".png", "")
        json_path = os.path.join(RAW_DIR, f"{base_name}.json")
        
        with open(json_path, 'r') as f:
            gt_json = json.load(f)
            
        expected_text = generate_expected_text(gt_json)
        category = gt_json.get("ground_truth_metadata", {}).get("quality", "standard")
        
        row_data = {
            "invoice_id": gt_json["invoice_id"],
            "category": category
        }
        
        for stage, folder in [("original", RAW_DIR), ("preprocessed", PROC_DIR)]:
            img_path = os.path.join(folder, img_file)
            metrics[stage]["attempted"] += 1
            res = extract_text(img_path, config)
            
            row_data[f"{stage}_success"] = res.success
            row_data[f"{stage}_processing_time"] = round(res.processing_time, 4)
            row_data[f"{stage}_confidence"] = round(res.average_confidence, 2) if res.average_confidence is not None else 0.0
            
            if res.success:
                metrics[stage]["success"] += 1
                metrics[stage]["times"].append(res.processing_time)
                metrics[stage]["conf"].append(res.average_confidence if res.average_confidence is not None else 0.0)
                
                cer, wer = calculate_metrics(expected_text, res.raw_text)
                metrics[stage]["cer"].append(cer)
                metrics[stage]["wer"].append(wer)
                
                row_data[f"{stage}_cer"] = round(cer, 4)
                row_data[f"{stage}_wer"] = round(wer, 4)
                
                metrics[stage]["fields"].append(field_presence(res.raw_text, gt_json))
                
                # Save raw OCR text
                with open(os.path.join(OUT_DIR, stage, f"{base_name}.txt"), 'w') as f:
                    f.write(res.raw_text)
            else:
                row_data[f"{stage}_cer"] = None
                row_data[f"{stage}_wer"] = None
                    
            # Save metadata
            meta = {
                "invoice_id": gt_json["invoice_id"],
                "source_file": img_file,
                "source_type": "synthetic",
                "preprocessing_status": "raw" if stage == "original" else "preprocessed",
                "ocr_engine": "Tesseract 5.5.3",
                "language": config.language,
                "psm": config.psm,
                "oem": config.oem,
                "processing_time_seconds": round(res.processing_time, 4),
                "average_confidence": round(res.average_confidence, 2) if res.average_confidence is not None else None,
                "word_count": res.word_count,
                "character_count": res.character_count,
                "success": res.success,
                "error": res.error
            }
            with open(os.path.join(OUT_DIR, "metadata", f"{base_name}_{stage}.json"), 'w') as f:
                json.dump(meta, f, indent=2)
                
        per_invoice_rows.append(row_data)

    # Save per-invoice results CSV
    csv_path = os.path.join(OUT_DIR, "evaluation_results.csv")
    csv_fields = [
        "invoice_id", "category",
        "original_success", "preprocessed_success",
        "original_cer", "preprocessed_cer",
        "original_wer", "preprocessed_wer",
        "original_confidence", "preprocessed_confidence",
        "original_processing_time", "preprocessed_processing_time"
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(per_invoice_rows)

    # Compute Statistics
    stats = {}
    for stage in ["original", "preprocessed"]:
        m = metrics[stage]
        stats[stage] = {
            "attempted": m["attempted"],
            "success": m["success"],
            "cer": compute_stats(m["cer"]),
            "wer": compute_stats(m["wer"]),
            "conf": compute_stats(m["conf"]),
            "time": compute_stats(m["times"]),
            "fields": {}
        }
        
        # Field presence rates
        all_field_keys = ["invoice_number", "invoice_date", "seller_gstin", "buyer_gstin", "taxable_value", "cgst", "sgst", "igst", "grand_total"]
        for k in all_field_keys:
            valid_entries = [f[k] for f in m["fields"] if f[k] is not None]
            if valid_entries:
                present_count = sum(1 for v in valid_entries if v is True)
                stats[stage]["fields"][k] = {
                    "present": present_count,
                    "expected": len(valid_entries),
                    "rate": (present_count / len(valid_entries)) * 100.0
                }
            else:
                stats[stage]["fields"][k] = {
                    "present": 0,
                    "expected": 0,
                    "rate": 100.0
                }

    # Print Full Report
    print("="*60)
    print("PHASE 4 — BATCH OCR & EMPIRICAL EVALUATION REPORT")
    print("="*60)
    for stage in ["original", "preprocessed"]:
        s = stats[stage]
        print(f"\nSTAGE: {stage.upper()}")
        print(f"  Attempted: {s['attempted']}, Successful: {s['success']}")
        print(f"  Processing Time: Mean={s['time']['mean']:.3f}s | Median={s['time']['median']:.3f}s | Min={s['time']['min']:.3f}s | Max={s['time']['max']:.3f}s | Total={s['time']['total']:.3f}s")
        print(f"  Confidence:      Mean={s['conf']['mean']:.2f}% | Median={s['conf']['median']:.2f}% | Min={s['conf']['min']:.2f}% | Max={s['conf']['max']:.2f}%")
        print(f"  CER:             Mean={s['cer']['mean']:.4f} | Median={s['cer']['median']:.4f} | Min={s['cer']['min']:.4f} | Max={s['cer']['max']:.4f}")
        print(f"  WER:             Mean={s['wer']['mean']:.4f} | Median={s['wer']['median']:.4f} | Min={s['wer']['min']:.4f} | Max={s['wer']['max']:.4f}")
        print("  Field Presence Rates:")
        for k, f_stat in s["fields"].items():
            print(f"    {k:15s}: {f_stat['present']}/{f_stat['expected']} ({f_stat['rate']:.1f}%)")

    print("\n" + "="*60)
    print("ORIGINAL VS PREPROCESSED DELTA")
    print("="*60)
    delta_cer = stats["preprocessed"]["cer"]["mean"] - stats["original"]["cer"]["mean"]
    delta_wer = stats["preprocessed"]["wer"]["mean"] - stats["original"]["wer"]["mean"]
    delta_conf = stats["preprocessed"]["conf"]["mean"] - stats["original"]["conf"]["mean"]
    delta_time = stats["preprocessed"]["time"]["mean"] - stats["original"]["time"]["mean"]
    print(f"Mean CER Change:        {delta_cer:+.4f} ({'Improved' if delta_cer < 0 else ('Worsened' if delta_cer > 0 else 'Unchanged')})")
    print(f"Mean WER Change:        {delta_wer:+.4f} ({'Improved' if delta_wer < 0 else ('Worsened' if delta_wer > 0 else 'Unchanged')})")
    print(f"Mean Confidence Change: {delta_conf:+.2f}% ({'Improved' if delta_conf > 0 else ('Worsened' if delta_conf < 0 else 'Unchanged')})")
    print(f"Mean Time Change:       {delta_time:+.3f}s")

    return stats

if __name__ == "__main__":
    process_batch()
