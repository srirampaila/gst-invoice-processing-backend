"""
Parser, validation, and confidence heuristics for LLM structured extraction responses.
"""
import re
import json
from typing import Dict, Any, Tuple, List
from pydantic import ValidationError
from src.extraction.models import CanonicalInvoice, ExtractionMetadata
from src.extraction.exceptions import LLMResponseParsingError, SchemaValidationError

def clean_json_string(text: str) -> str:
    """Strips markdown code fences and extraneous text from JSON response."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        # Remove closing fence
        text = re.sub(r"\n?```$", "", text)
    return text.strip()

def compute_extraction_confidence(data: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Computes an explainable application-level extraction confidence score (0.0 to 1.0)
    and collects extraction warnings based on field completeness and structure.
    """
    score = 1.0
    warnings = []

    # 1. Critical Identifier Checks
    if not data.get("invoice_number"):
        score -= 0.20
        warnings.append("Missing invoice_number")

    date_val = data.get("invoice_date")
    if not date_val:
        score -= 0.15
        warnings.append("Missing invoice_date")
    elif not re.match(r"^\d{4}-\d{2}-\d{2}$", str(date_val)):
        score -= 0.05
        warnings.append(f"Non-standard invoice_date format: {date_val}")

    # 2. Seller Information
    seller = data.get("seller") or {}
    seller_gstin = seller.get("gstin")
    if not seller_gstin:
        score -= 0.25
        warnings.append("Missing seller GSTIN")
    elif len(str(seller_gstin)) != 15:
        score -= 0.10
        warnings.append(f"Invalid seller GSTIN length ({len(str(seller_gstin))} chars): {seller_gstin}")

    # 3. Buyer Information (Optional in B2C, but noted)
    buyer = data.get("buyer") or {}
    buyer_gstin = buyer.get("gstin")
    if buyer_gstin and len(str(buyer_gstin)) != 15:
        score -= 0.05
        warnings.append(f"Invalid buyer GSTIN length: {buyer_gstin}")

    # 4. Financial Totals
    tax_summary = data.get("tax_summary") or {}
    if data.get("grand_total") is None:
        score -= 0.20
        warnings.append("Missing grand_total")

    if tax_summary.get("taxable_value") is None:
        score -= 0.10
        warnings.append("Missing taxable_value in tax_summary")

    # 5. Line Items
    items = data.get("items") or []
    if not items:
        score -= 0.15
        warnings.append("No line items extracted")
    else:
        for idx, item in enumerate(items):
            if item.get("taxable_value") is None and item.get("unit_price") is None:
                score -= 0.05
                warnings.append(f"Item {idx} missing taxable_value and unit_price")

    # Ensure score stays bounded [0.0, 1.0]
    final_score = max(0.0, min(1.0, round(score, 2)))
    return final_score, warnings

def parse_and_validate_json(
    raw_response: str,
    source_file: str = "unknown",
    extraction_method: str = "llm",
    model: str = "gpt-4o-mini"
) -> CanonicalInvoice:
    """
    Parses LLM response string into canonical JSON and validates schema via Pydantic.
    """
    cleaned = clean_json_string(raw_response)
    
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Fallback regex search for embedded JSON object
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
            except Exception:
                raise LLMResponseParsingError(f"Failed to parse LLM response as JSON: {e}")
        else:
            raise LLMResponseParsingError(f"Failed to parse LLM response as JSON: {e}")

    if not isinstance(data, dict):
        raise SchemaValidationError("LLM response root must be a JSON object dictionary.")

    # Calculate explainable confidence heuristic
    confidence, warnings = compute_extraction_confidence(data)

    # Attach ExtractionMetadata
    data["extraction_metadata"] = ExtractionMetadata(
        source_file=source_file,
        extraction_method=extraction_method,
        model=model,
        confidence=confidence,
        warnings=warnings
    ).model_dump()

    try:
        canonical_obj = CanonicalInvoice.model_validate(data)
        return canonical_obj
    except ValidationError as e:
        raise SchemaValidationError(f"Canonical schema validation failed: {e}")
