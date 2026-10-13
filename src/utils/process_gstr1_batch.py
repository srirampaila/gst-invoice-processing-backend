"""
Batch Processing and Export Script for Phase 8 GSTR-1 Draft Generation.
Processes all 25 Canonical Invoices, executes the eligibility gate, maps admitted invoices to
GSTR-1 Tables 4A & 7, validates the draft records, and exports CSV & multi-sheet Excel reports.
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.extraction.models import CanonicalInvoice
from src.validation.models import ValidationResult
from src.confidence.models import ConfidenceResult, ReviewRecord
from src.gst.gstr1_models import GSTR1Draft
from src.gst.gstr1_generator import GSTR1DraftGenerator
from src.export.gstr1_csv import (
    export_gstr1_combined_csv,
    export_b2b_csv,
    export_b2c_csv,
    export_exclusions_csv,
    export_validation_report_csv
)
from src.export.gstr1_excel import export_gstr1_excel
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXTRACTION_JSON_DIR = "data/outputs/extraction/json"
VALIDATION_RESULTS_DIR = "data/outputs/validation/results"
CONFIDENCE_RESULTS_DIR = "data/outputs/confidence/results"
CONFIDENCE_REVIEWS_DIR = "data/outputs/confidence/reviews"

OUT_BASE_DIR = Path("data/outputs/gstr1")
OUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV_COMBINED = OUT_BASE_DIR / "gstr1_draft.csv"
OUT_CSV_B2B = OUT_BASE_DIR / "b2b_table_4a.csv"
OUT_CSV_B2C = OUT_BASE_DIR / "b2c_table_7.csv"
OUT_CSV_EXCLUSIONS = OUT_BASE_DIR / "exclusions.csv"
OUT_CSV_VALIDATION = OUT_BASE_DIR / "validation_report.csv"
OUT_EXCEL_PATH = OUT_BASE_DIR / "gstr1_draft.xlsx"
OUT_SUMMARY_PATH = OUT_BASE_DIR / "summary.json"

def run_gstr1_batch() -> GSTR1Draft:
    generator = GSTR1DraftGenerator()

    json_files = sorted([f for f in os.listdir(EXTRACTION_JSON_DIR) if f.endswith(".json")])
    if not json_files:
        print(f"No extraction JSON files found in {EXTRACTION_JSON_DIR}")
        return GSTR1Draft()

    print("=" * 75)
    print("STARTING GSTR-1 DRAFT DATA GENERATION & EXPORT BATCH (PHASE 8)")
    print("Status: GSTR-1 DRAFT — NOT FILED (Draft Preparation Only)")
    print("=" * 75)

    invoice_entries = []

    for idx, fname in enumerate(json_files, 1):
        inv_id = fname.replace(".json", "")
        ext_path = os.path.join(EXTRACTION_JSON_DIR, fname)
        val_path = os.path.join(VALIDATION_RESULTS_DIR, fname)
        conf_path = os.path.join(CONFIDENCE_RESULTS_DIR, fname)
        rev_path = os.path.join(CONFIDENCE_REVIEWS_DIR, fname)

        # 1. Load Canonical Invoice
        with open(ext_path, "r", encoding="utf-8") as f:
            invoice_data = json.load(f)
        invoice = CanonicalInvoice.model_validate(invoice_data)

        # 2. Load Validation Result
        val_res = None
        if os.path.exists(val_path):
            with open(val_path, "r", encoding="utf-8") as f:
                val_data = json.load(f)
            val_res = ValidationResult.model_validate(val_data)

        # 3. Load Confidence Result
        conf_res = None
        if os.path.exists(conf_path):
            with open(conf_path, "r", encoding="utf-8") as f:
                conf_data = json.load(f)
            conf_res = ConfidenceResult.model_validate(conf_data)

        # 4. Load Review Record
        rev_rec = None
        if os.path.exists(rev_path):
            with open(rev_path, "r", encoding="utf-8") as f:
                rev_data = json.load(f)
            rev_rec = ReviewRecord.model_validate(rev_data)

        invoice_entries.append({
            "invoice": invoice,
            "validation_result": val_res,
            "confidence_result": conf_res,
            "review_record": rev_rec,
            "invoice_id": inv_id
        })

    # Generate Draft
    draft = generator.generate_draft(invoice_entries)

    # Log Individual Invoice Status
    for idx, entry in enumerate(invoice_entries, 1):
        inv_id = entry["invoice_id"]
        # Check if included in B2B, B2C or excluded
        in_b2b = any(r.source_invoice_id == inv_id for r in draft.b2b_records)
        in_b2c = any(r.source_invoice_id == inv_id for r in draft.b2c_records)
        exc = next((e for e in draft.exclusions if e.invoice_id == inv_id), None)

        if in_b2b:
            b2b_rec = next(r for r in draft.b2b_records if r.source_invoice_id == inv_id)
            print(f"[{idx:02d}/25] {inv_id:<15} -> 🟢 INCLUDED [Table 4A B2B] (Total: ₹{b2b_rec.invoice_value:,.2f}, Taxable: ₹{b2b_rec.taxable_value:,.2f}, Rate: {b2b_rec.rate}%)")
        elif in_b2c:
            b2c_rec = next(r for r in draft.b2c_records if r.source_invoice_id == inv_id)
            print(f"[{idx:02d}/25] {inv_id:<15} -> 🟢 INCLUDED [Table 7  B2C] (Total: ₹{b2c_rec.invoice_value:,.2f}, Taxable: ₹{b2c_rec.taxable_value:,.2f}, Rate: {b2c_rec.rate}%)")
        else:
            reason_snippet = exc.reason[:60] + "..." if exc and len(exc.reason) > 60 else (exc.reason if exc else "Excluded")
            print(f"[{idx:02d}/25] {inv_id:<15} -> 🔴 EXCLUDED  [Ineligible] ({reason_snippet})")

    # Export all CSV and Excel artifacts
    export_gstr1_combined_csv(draft, OUT_CSV_COMBINED)
    export_b2b_csv(draft.b2b_records, OUT_CSV_B2B)
    export_b2c_csv(draft.b2c_records, OUT_CSV_B2C)
    export_exclusions_csv(draft.exclusions, OUT_CSV_EXCLUSIONS)
    export_validation_report_csv(draft.validation_issues, OUT_CSV_VALIDATION)
    export_gstr1_excel(draft, OUT_EXCEL_PATH)

    # Save summary JSON
    with open(OUT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(draft.model_dump(), f, indent=2)

    print("\n" + "=" * 75)
    print("PHASE 8 GSTR-1 DRAFT GENERATION SUMMARY")
    print("=" * 75)
    print(f"Total Invoices Evaluated:       {draft.source_invoice_count}")
    print(f"Invoices Admitted into Draft:   {draft.included_invoice_count} ({round(draft.included_invoice_count/draft.source_invoice_count*100, 1)}%)")
    print(f"  ├─ B2B Records (Table 4A):    {draft.b2b_count}")
    print(f"  └─ B2C Records (Table 7):     {draft.b2c_count}")
    print(f"Invoices Excluded:              {draft.excluded_invoice_count} ({round(draft.excluded_invoice_count/draft.source_invoice_count*100, 1)}%)")
    print(f"GSTR-1 Draft Validation Issues: {len(draft.validation_issues)}")
    print("-" * 75)
    print(f"Total Taxable Value:            ₹{draft.total_taxable_value:,.2f}")
    print(f"Total CGST (Central Tax):       ₹{draft.total_cgst:,.2f}")
    print(f"Total SGST (State/UT Tax):      ₹{draft.total_sgst:,.2f}")
    print(f"Total IGST (Integrated Tax):    ₹{draft.total_igst:,.2f}")
    print(f"Total Grand Invoice Value:      ₹{draft.total_invoice_value:,.2f}")
    print("-" * 75)
    print(f"Exported Artifacts in:          {OUT_BASE_DIR}")
    print(f"  ├─ {OUT_CSV_COMBINED.name}")
    print(f"  ├─ {OUT_CSV_B2B.name}")
    print(f"  ├─ {OUT_CSV_B2C.name}")
    print(f"  ├─ {OUT_CSV_EXCLUSIONS.name}")
    print(f"  ├─ {OUT_CSV_VALIDATION.name}")
    print(f"  ├─ {OUT_EXCEL_PATH.name}")
    print(f"  └─ {OUT_SUMMARY_PATH.name}")
    print("=" * 75 + "\n")

    return draft

if __name__ == "__main__":
    run_gstr1_batch()
