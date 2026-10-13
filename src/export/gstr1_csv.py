"""
CSV Export Generators for GSTR-1 Draft Tables and Reports.
"""
import csv
from pathlib import Path
from typing import List, Union
from src.gst.gstr1_models import (
    GSTR1Draft,
    GSTR1B2BInvoice,
    GSTR1B2CSInvoice,
    GSTR1ExclusionRecord,
    GSTR1ValidationIssue
)

def export_b2b_csv(records: List[GSTR1B2BInvoice], file_path: Union[str, Path]) -> Path:
    """Exports GSTR-1 Table 4A (B2B) records to CSV."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_invoice_id",
        "recipient_gstin",
        "recipient_name",
        "invoice_number",
        "invoice_date",
        "invoice_value",
        "place_of_supply",
        "reverse_charge",
        "invoice_type",
        "taxable_value",
        "rate",
        "cgst_amount",
        "sgst_amount",
        "igst_amount",
        "cess_amount",
        "source_validation_status",
        "source_confidence_score"
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r.model_dump())

    return path

def export_b2c_csv(records: List[GSTR1B2CSInvoice], file_path: Union[str, Path]) -> Path:
    """Exports GSTR-1 Table 7 (B2C Small) records to CSV."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_invoice_id",
        "place_of_supply",
        "invoice_number",
        "invoice_date",
        "invoice_value",
        "taxable_value",
        "rate",
        "cgst_amount",
        "sgst_amount",
        "igst_amount",
        "cess_amount",
        "source_validation_status",
        "source_confidence_score"
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r.model_dump())

    return path

def export_exclusions_csv(exclusions: List[GSTR1ExclusionRecord], file_path: Union[str, Path]) -> Path:
    """Exports excluded invoices catalog to CSV."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "invoice_id",
        "reason",
        "confidence_score",
        "confidence_tier",
        "recommendation",
        "review_status",
        "validation_status",
        "error_count",
        "warning_count",
        "triggered_rules",
        "timestamp"
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for exc in exclusions:
            data = exc.model_dump()
            data["triggered_rules"] = "; ".join(data.get("triggered_rules", []))
            writer.writerow(data)

    return path

def export_validation_report_csv(issues: List[GSTR1ValidationIssue], file_path: Union[str, Path]) -> Path:
    """Exports GSTR-1 draft validation findings to CSV."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["invoice_id", "rule_id", "severity", "field", "message"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for issue in issues:
            writer.writerow(issue.model_dump())

    return path

def export_gstr1_combined_csv(draft: GSTR1Draft, file_path: Union[str, Path]) -> Path:
    """Exports all included GSTR-1 records into a unified combined CSV."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "section",
        "source_invoice_id",
        "recipient_gstin",
        "recipient_name",
        "invoice_number",
        "invoice_date",
        "invoice_value",
        "place_of_supply",
        "taxable_value",
        "rate",
        "cgst_amount",
        "sgst_amount",
        "igst_amount",
        "cess_amount"
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for b2b in draft.b2b_records:
            row = b2b.model_dump()
            row["section"] = "Table 4A (B2B)"
            writer.writerow({k: row.get(k, "") for k in fieldnames})

        for b2c in draft.b2c_records:
            row = b2c.model_dump()
            row["section"] = "Table 7 (B2C)"
            row["recipient_gstin"] = ""
            row["recipient_name"] = ""
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    return path
