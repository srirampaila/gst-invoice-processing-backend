"""
Invoice management and pipeline execution router.
"""
import os
import time
import json
import shutil
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse

from src.api.models import SampleInvoiceInfo, ProcessInvoiceRequest
from src.preprocessing.pipeline import preprocess_image
from src.preprocessing.config import PreprocessingConfig
from src.ocr.service import extract_text
from src.ocr.config import OCRConfig
from src.extraction.service import extract_invoice_from_ocr
from src.extraction.config import ExtractionConfig
from src.validation.service import validate_canonical_invoice
from src.validation.config import ValidationConfig
from src.confidence.service import ConfidenceService
from src.confidence.config import ConfidenceConfig
from src.confidence.models import ReviewRecord
from src.gst.gstr1_mapper import map_canonical_to_gstr1
from src.gst.gstr1_models import GSTR1B2BInvoice, GSTR1B2CSInvoice
from src.utils.paths import RAW_INVOICES_DIR, TEMP_PROCESSED_DIR, REVIEWS_DIR

router = APIRouter(prefix="/api/invoices", tags=["Invoices"])

TEMP_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

def resolve_category(filename: str) -> str:
    name = filename.upper()
    if "CLEAN" in name:
        return "Clean Single-Page"
    elif "MULTI" in name:
        return "Multi-Item Detailed"
    elif "B2C" in name:
        return "B2C Consumer Retail"
    elif "NOISE" in name:
        return "Noisy / Degraded"
    elif "ROT" in name:
        return "Rotated / Skewed"
    elif "INV" in name:
        return "Arithmetic Error Test"
    elif "DUP" in name:
        return "Duplicate Sequence"
    return "Standard Invoice"

def load_or_create_review_record(conf_service, conf_res, val_res, invoice_id: str) -> ReviewRecord:
    rev_file = REVIEWS_DIR / f"{invoice_id}.json"
    if rev_file.exists():
        try:
            with open(rev_file, "r", encoding="utf-8") as f:
                return ReviewRecord.model_validate(json.load(f))
        except Exception:
            pass
    record = conf_service.create_review_record(conf_res, val_res)
    with open(rev_file, "w", encoding="utf-8") as f:
        json.dump(record.model_dump(), f, indent=2)
    return record

@router.get("/samples", response_model=List[SampleInvoiceInfo])
def get_sample_invoices():
    if not RAW_INVOICES_DIR.exists():
        return []
    
    samples = []
    for p in sorted(RAW_INVOICES_DIR.glob("*.png")):
        inv_id = p.stem
        rev_file = REVIEWS_DIR / f"{inv_id}.json"
        has_rev = rev_file.exists()
        rev_status = None
        if has_rev:
            try:
                with open(rev_file, "r", encoding="utf-8") as f:
                    rev_status = json.load(f).get("review_status")
            except Exception:
                pass

        samples.append(
            SampleInvoiceInfo(
                file_name=p.name,
                invoice_id=inv_id,
                category=resolve_category(p.name),
                file_size_kb=round(p.stat().st_size / 1024.0, 1),
                has_review=has_rev,
                review_status=rev_status,
                image_url=f"/api/invoices/image/{p.name}"
            )
        )
    return samples

@router.get("/image/{filename}")
def get_invoice_image(filename: str):
    # Search in raw or temp
    raw_path = RAW_INVOICES_DIR / filename
    temp_path = TEMP_PROCESSED_DIR / filename
    if raw_path.exists():
        return FileResponse(raw_path)
    elif temp_path.exists():
        return FileResponse(temp_path)
    raise HTTPException(status_code=404, detail=f"Image {filename} not found")

@router.post("/upload")
async def upload_invoice(file: UploadFile = File(...)):
    valid_exts = [".png", ".jpg", ".jpeg", ".pdf"]
    ext = Path(file.filename).suffix.lower()
    if ext not in valid_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: PNG, JPG, JPEG, PDF"
        )
    
    # Save to temp
    dest_path = TEMP_PROCESSED_DIR / file.filename
    content = await file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    return {
        "status": "success",
        "file_name": file.filename,
        "invoice_id": Path(file.filename).stem,
        "file_size_kb": round(len(content) / 1024.0, 1),
        "image_url": f"/api/invoices/image/{file.filename}"
    }

@router.post("/process")
def process_invoice(req: ProcessInvoiceRequest):
    # Locate source image
    raw_path = RAW_INVOICES_DIR / req.file_name
    temp_path = TEMP_PROCESSED_DIR / req.file_name
    
    if raw_path.exists():
        src_path = str(raw_path)
    elif temp_path.exists():
        src_path = str(temp_path)
    else:
        raise HTTPException(status_code=404, detail=f"Invoice file '{req.file_name}' not found.")

    inv_id = Path(req.file_name).stem
    t0 = time.perf_counter()

    # Stage 1: Preprocessing
    proc_image_path = str(TEMP_PROCESSED_DIR / f"proc_{req.file_name}")
    prep_config = PreprocessingConfig()
    try:
        prep_res = preprocess_image(src_path, proc_image_path, prep_config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image Preprocessing Error: {str(e)}")

    # Stage 2: OCR Engine
    ocr_config = OCRConfig()
    try:
        ocr_res = extract_text(proc_image_path, ocr_config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR Engine Error: {str(e)}")

    # Stage 3: Structured Extraction
    ext_config = ExtractionConfig(provider=req.provider)
    try:
        invoice = extract_invoice_from_ocr(
            ocr_text=ocr_res.raw_text,
            source_file=req.file_name,
            config=ext_config
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Structured Extraction Error: {str(e)}")

    # Stage 4: Deterministic GST Validation
    val_config = ValidationConfig(amount_tolerance=req.amount_tolerance)
    val_res = validate_canonical_invoice(
        invoice=invoice,
        config=val_config,
        invoice_id=req.file_name
    )

    # Stage 5: Multi-Factor Confidence Scoring
    conf_config = ConfidenceConfig(
        high_threshold=req.high_threshold,
        medium_threshold=req.medium_threshold
    )
    conf_service = ConfidenceService(conf_config)
    conf_res = conf_service.evaluate_confidence(
        invoice=invoice,
        validation_result=val_res,
        ocr_metadata={"average_confidence": ocr_res.average_confidence},
        invoice_id=req.file_name
    )

    # Stage 6: GSTR-1 Mapping & Eligibility Gate
    review_record = load_or_create_review_record(conf_service, conf_res, val_res, inv_id)
    gstr1_record, gstr1_exclusion = map_canonical_to_gstr1(
        invoice=invoice,
        validation_result=val_res,
        confidence_result=conf_res,
        review_record=review_record,
        invoice_id=inv_id
    )

    t_total = time.perf_counter() - t0

    return {
        "invoice_id": inv_id,
        "file_name": req.file_name,
        "total_latency_seconds": round(t_total, 4),
        "provider_used": invoice.extraction_metadata.extraction_method,
        "model_used": invoice.extraction_metadata.model,
        "image_url": f"/api/invoices/image/{req.file_name}",
        "processed_image_url": f"/api/invoices/image/proc_{req.file_name}",
        "preprocessing": {
            "status": "success",
            "applied_steps": ["grayscale", "bilateral_filter", "clahe", "resize_2x", "deskew"],
            "processing_time": prep_res.get("processing_time", 0.0) if isinstance(prep_res, dict) else 0.0
        },
        "ocr": {
            "average_confidence": ocr_res.average_confidence,
            "word_count": ocr_res.word_count,
            "character_count": ocr_res.character_count,
            "processing_time": ocr_res.processing_time,
            "raw_text": ocr_res.raw_text
        },
        "canonical_invoice": invoice.model_dump(),
        "validation": val_res.model_dump(),
        "confidence": conf_res.model_dump(),
        "review_record": review_record.model_dump(),
        "gstr1": {
            "is_admitted": gstr1_record is not None,
            "table_type": "Table 4A (B2B)" if isinstance(gstr1_record, GSTR1B2BInvoice) else ("Table 7 (B2CS)" if isinstance(gstr1_record, GSTR1B2CSInvoice) else None),
            "record": gstr1_record.model_dump() if gstr1_record else None,
            "exclusion": gstr1_exclusion.model_dump() if gstr1_exclusion else None
        }
    }
