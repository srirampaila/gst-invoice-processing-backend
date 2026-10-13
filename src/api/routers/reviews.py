"""
Human review and audit trail router.
"""
import json
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException

from src.api.models import ReviewActionRequest
from src.confidence.service import ConfidenceService
from src.confidence.config import ConfidenceConfig
from src.confidence.models import ReviewRecord, ReviewerAction, ReviewStatus
from src.validation.service import validate_canonical_invoice
from src.validation.config import ValidationConfig
from src.extraction.service import extract_invoice_from_ocr
from src.extraction.config import ExtractionConfig
from src.gst.gstr1_mapper import map_canonical_to_gstr1
from src.utils.paths import (
    REVIEWS_DIR,
    CONFIDENCE_RESULTS_DIR,
    VALIDATION_RESULTS_DIR,
    EXTRACTION_JSON_DIR
)

router = APIRouter(prefix="/api/reviews", tags=["Human Review"])

REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

@router.get("/all")
def get_all_reviews():
    records = []
    if REVIEWS_DIR.exists():
        for p in sorted(REVIEWS_DIR.glob("*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    records.append(json.load(f))
            except Exception:
                pass
    return records

@router.get("/{invoice_id}")
def get_review_record(invoice_id: str):
    rev_file = REVIEWS_DIR / f"{invoice_id}.json"
    if not rev_file.exists():
        raise HTTPException(status_code=404, detail=f"Review record for invoice '{invoice_id}' not found.")
    
    with open(rev_file, "r", encoding="utf-8") as f:
        return json.load(f)

@router.post("/{invoice_id}/action")
def apply_review_action(invoice_id: str, req: ReviewActionRequest):
    rev_file = REVIEWS_DIR / f"{invoice_id}.json"
    if not rev_file.exists():
        raise HTTPException(status_code=404, detail=f"Review record for invoice '{invoice_id}' not found.")
    
    with open(rev_file, "r", encoding="utf-8") as f:
        record_data = json.load(f)
    
    record = ReviewRecord.model_validate(record_data)
    
    # Map action string to Enum
    action_str = req.action.upper().strip()
    if action_str == "APPROVE":
        enum_action = ReviewerAction.APPROVE
    elif action_str == "REJECT":
        enum_action = ReviewerAction.REJECT
    elif action_str in ("REQUEST_REPROCESSING", "REPROCESS"):
        enum_action = ReviewerAction.REQUEST_REPROCESSING
    else:
        raise HTTPException(status_code=400, detail=f"Invalid review action '{req.action}'. Expected APPROVE, REJECT, or REQUEST_REPROCESSING.")

    conf_service = ConfidenceService(ConfidenceConfig())
    updated_record = conf_service.apply_review_action(
        record=record,
        action=enum_action,
        notes=req.notes
    )

    # Save to disk
    with open(rev_file, "w", encoding="utf-8") as f:
        json.dump(updated_record.model_dump(), f, indent=2)

    # Re-evaluate GSTR-1 admission if extraction/validation files exist
    extr_file = EXTRACTION_JSON_DIR / f"{invoice_id}.json"
    val_file = VALIDATION_RESULTS_DIR / f"{invoice_id}.json"
    conf_file = CONFIDENCE_RESULTS_DIR / f"{invoice_id}.json"

    gstr1_status = None
    gstr1_is_admitted = False
    gstr1_exclusion_reason = None

    if extr_file.exists() and val_file.exists() and conf_file.exists():
        try:
            from src.extraction.models import CanonicalInvoice
            from src.validation.models import ValidationResult
            from src.confidence.models import ConfidenceResult

            with open(extr_file, "r", encoding="utf-8") as f:
                inv = CanonicalInvoice.model_validate(json.load(f))
            with open(val_file, "r", encoding="utf-8") as f:
                val_res = ValidationResult.model_validate(json.load(f))
            with open(conf_file, "r", encoding="utf-8") as f:
                conf_res = ConfidenceResult.model_validate(json.load(f))

            rec, exc = map_canonical_to_gstr1(
                invoice=inv,
                validation_result=val_res,
                confidence_result=conf_res,
                review_record=updated_record,
                invoice_id=invoice_id
            )
            gstr1_is_admitted = (rec is not None)
            if exc:
                gstr1_exclusion_reason = exc.reason
        except Exception:
            pass

    return {
        "status": "success",
        "invoice_id": invoice_id,
        "review_status": updated_record.review_status,
        "reviewer_action": updated_record.reviewer_action,
        "reviewed_at": updated_record.reviewed_at,
        "notes": updated_record.notes,
        "record": updated_record.model_dump(),
        "gstr1_is_admitted": gstr1_is_admitted,
        "gstr1_exclusion_reason": gstr1_exclusion_reason
    }
