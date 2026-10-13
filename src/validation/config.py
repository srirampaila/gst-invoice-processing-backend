"""
Configuration settings for the Deterministic GST & Financial Validation Engine.
"""
import os
from decimal import Decimal
from typing import Optional

class ValidationConfig:
    """Configurable tolerances and validation strictness settings."""
    
    def __init__(
        self,
        amount_tolerance: Optional[Decimal] = None,
        percentage_tolerance: Optional[Decimal] = None,
        strict_gstin_checksum: Optional[bool] = None,
        currency: str = "INR"
    ):
        # 1. Financial amount tolerance (default: ₹1.00)
        env_tol = os.getenv("VALIDATION_AMOUNT_TOLERANCE")
        if amount_tolerance is not None:
            self.amount_tolerance = Decimal(str(amount_tolerance))
        elif env_tol:
            self.amount_tolerance = Decimal(env_tol)
        else:
            self.amount_tolerance = Decimal("1.00")
            
        # 2. Percentage tolerance for tax rate calculations (default: 0.1%)
        env_pct_tol = os.getenv("VALIDATION_PERCENTAGE_TOLERANCE")
        if percentage_tolerance is not None:
            self.percentage_tolerance = Decimal(str(percentage_tolerance))
        elif env_pct_tol:
            self.percentage_tolerance = Decimal(env_pct_tol)
        else:
            self.percentage_tolerance = Decimal("0.10")

        # 3. GSTIN Checksum strictness
        env_strict = os.getenv("VALIDATION_STRICT_GSTIN_CHECKSUM")
        if strict_gstin_checksum is not None:
            self.strict_gstin_checksum = strict_gstin_checksum
        elif env_strict is not None:
            self.strict_gstin_checksum = env_strict.lower() in ("true", "1", "yes")
        else:
            self.strict_gstin_checksum = False

        self.currency = currency
