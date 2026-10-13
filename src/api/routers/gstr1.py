"""
GSTR-1 Draft and Exports router.
"""
import json
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.gst.gstr1_models import GSTR1Draft
from src.utils.process_gstr1_batch import run_gstr1_batch
from src.api.models import BatchGSTR1RunResponse

from src.utils.paths import GSTR1_DIR

router = APIRouter(prefix="/api/gstr1", tags=["GSTR-1 Draft"])

GSTR1_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV_COMBINED = GSTR1_DIR / "gstr1_draft.csv"
OUT_CSV_B2B = GSTR1_DIR / "b2b_table_4a.csv"
OUT_CSV_B2C = GSTR1_DIR / "b2c_table_7.csv"
OUT_CSV_EXCLUSIONS = GSTR1_DIR / "exclusions.csv"
OUT_EXCEL_PATH = GSTR1_DIR / "gstr1_draft.xlsx"
OUT_SUMMARY_PATH = GSTR1_DIR / "summary.json"

def get_or_create_gstr1_draft() -> GSTR1Draft:
    if OUT_SUMMARY_PATH.exists():
        try:
            with open(OUT_SUMMARY_PATH, "r", encoding="utf-8") as f:
                return GSTR1Draft.model_validate(json.load(f))
        except Exception:
            pass
    return run_gstr1_batch()

@router.get("/draft")
def get_gstr1_draft():
    draft = get_or_create_gstr1_draft()
    return draft.model_dump()

@router.post("/batch/run", response_model=BatchGSTR1RunResponse)
def execute_gstr1_batch():
    try:
        draft = run_gstr1_batch()
        return BatchGSTR1RunResponse(
            status="success",
            message="GSTR-1 batch generation and multi-format export completed successfully.",
            source_invoice_count=draft.source_invoice_count,
            included_invoice_count=draft.included_invoice_count,
            excluded_invoice_count=draft.excluded_invoice_count,
            b2b_count=draft.b2b_count,
            b2c_count=draft.b2c_count,
            total_taxable_value=float(draft.total_taxable_value),
            total_invoice_value=float(draft.total_invoice_value)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GSTR-1 batch execution error: {str(e)}")

@router.get("/exports/{export_type}")
def download_gstr1_export(export_type: str):
    mapping = {
        "combined_csv": (OUT_CSV_COMBINED, "gstr1_draft.csv", "text/csv"),
        "b2b_csv": (OUT_CSV_B2B, "b2b_table_4a.csv", "text/csv"),
        "b2c_csv": (OUT_CSV_B2C, "b2c_table_7.csv", "text/csv"),
        "exclusions_csv": (OUT_CSV_EXCLUSIONS, "exclusions.csv", "text/csv"),
        "excel": (OUT_EXCEL_PATH, "gstr1_draft.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    }

    if export_type not in mapping:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid export type '{export_type}'. Allowed: {list(mapping.keys())}"
        )

    file_path, filename, media_type = mapping[export_type]
    
    # Generate if not on disk
    if not file_path.exists():
        run_gstr1_batch()

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Export file '{filename}' could not be located.")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type
    )
