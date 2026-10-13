"""
Deterministic Financial Arithmetic and Precision Calculations using Python Decimal.
Avoids binary floating-point rounding inaccuracies in financial validation.
"""
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Optional, List, Any

QUANTIZE_PRECISION = Decimal("0.01")

def to_decimal(val: Any) -> Optional[Decimal]:
    """Safely converts floats, ints, strings, or existing Decimals to Decimal."""
    if val is None:
        return None
    if isinstance(val, Decimal):
        return val
    try:
        # Convert float via string to prevent float representation artifacts
        return Decimal(str(val)).quantize(QUANTIZE_PRECISION, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError, TypeError):
        return None

def is_close(
    val1: Optional[Decimal],
    val2: Optional[Decimal],
    tolerance: Decimal = Decimal("1.00")
) -> bool:
    """Checks if two decimal values are within the configured tolerance."""
    if val1 is None or val2 is None:
        return val1 == val2
    return abs(val1 - val2) <= tolerance

def calculate_item_taxable_sum(items: List[Any]) -> Decimal:
    """Sums the taxable value across all line items."""
    total = Decimal("0.00")
    for item in items:
        # Support dict or Pydantic model
        taxable = getattr(item, "taxable_value", None) if hasattr(item, "taxable_value") else item.get("taxable_value")
        d_val = to_decimal(taxable)
        if d_val is not None:
            total += d_val
    return total.quantize(QUANTIZE_PRECISION, rounding=ROUND_HALF_UP)

def calculate_expected_tax(taxable: Optional[Decimal], rate: Optional[Decimal]) -> Optional[Decimal]:
    """Calculates expected tax = (taxable * rate / 100)."""
    if taxable is None or rate is None:
        return None
    tax = (taxable * rate) / Decimal("100.00")
    return tax.quantize(QUANTIZE_PRECISION, rounding=ROUND_HALF_UP)

def calculate_grand_total(
    taxable: Optional[Decimal],
    cgst: Optional[Decimal],
    sgst: Optional[Decimal],
    igst: Optional[Decimal]
) -> Optional[Decimal]:
    """Calculates expected grand total = taxable + CGST + SGST + IGST."""
    if taxable is None:
        return None
    total = taxable
    if cgst is not None:
        total += cgst
    if sgst is not None:
        total += sgst
    if igst is not None:
        total += igst
    return total.quantize(QUANTIZE_PRECISION, rounding=ROUND_HALF_UP)
