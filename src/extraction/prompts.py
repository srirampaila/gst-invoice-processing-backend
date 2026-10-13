"""
Prompt definitions for Phase 5 LLM Structured Extraction.
"""

EXTRACTION_SYSTEM_PROMPT = """You are an expert Document AI invoice extraction engine.
Your task is to parse raw, imperfect Optical Character Recognition (OCR) text from an Indian GST invoice into a strictly valid, structured JSON document adhering to the canonical schema.

RULES & CONSTRAINTS:
1. The input is raw OCR text which may contain noise, missing spaces (e.g., 'SELLERGSTIN:', 'CGST:90.0', 'SGST: 90.0'), merged words, or minor OCR misspellings.
2. Extract only information factually present in the OCR text. NEVER hallucinate, extrapolate, or invent missing values.
3. If a field is not present or cannot be determined with certainty, set its value to null (or [] for empty item lists).
4. Preserve numeric values (quantities, prices, tax amounts, grand totals) as standard numeric floats/integers without currency symbols.
5. Preserve invoice identifiers (e.g. 'INV_CLEAN_001') exactly as written.
6. Normalize all dates to standard ISO 8601 format (YYYY-MM-DD).
7. Extract Indian GSTINs (15 alphanumeric characters). The first 2 digits represent the state code.
8. Line items: Extract all distinct items with their description, quantity, unit_price, taxable_value, and individual tax breakdown (cgst_rate, cgst_amount, sgst_rate, sgst_amount, igst_rate, igst_amount). If rate/amount is not separately indicated per line, leave null.
9. Summary totals: Extract total taxable_value, cgst, sgst, igst, and grand_total.
10. Output MUST be strictly valid JSON without any markdown formatting, code fences, commentary, or trailing text.

CANONICAL JSON SCHEMA:
{
  "invoice_number": string or null,
  "invoice_date": string (YYYY-MM-DD) or null,
  "seller": {
    "name": string or null,
    "gstin": string or null,
    "address": string or null
  },
  "buyer": {
    "name": string or null,
    "gstin": string or null,
    "address": string or null
  },
  "place_of_supply": string (2-digit state code) or null,
  "items": [
    {
      "description": string or null,
      "quantity": number or null,
      "unit_price": number or null,
      "taxable_value": number or null,
      "cgst_rate": number or null,
      "cgst_amount": number or null,
      "sgst_rate": number or null,
      "sgst_amount": number or null,
      "igst_rate": number or null,
      "igst_amount": number or null
    }
  ],
  "tax_summary": {
    "taxable_value": number or null,
    "cgst": number or null,
    "sgst": number or null,
    "igst": number or null
  },
  "grand_total": number or null,
  "currency": "INR"
}
"""

EXTRACTION_USER_PROMPT_TEMPLATE = """Extract structured invoice data from the following raw OCR text:

--- RAW OCR TEXT START ---
{ocr_text}
--- RAW OCR TEXT END ---

Output only the raw JSON object adhering to the canonical schema:"""
