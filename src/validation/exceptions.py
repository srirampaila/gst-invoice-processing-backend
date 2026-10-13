"""
Exceptions for Deterministic GST & Financial Validation.
"""

class ValidationError(Exception):
    """Base exception for validation engine errors."""
    pass

class GSTINValidationError(ValidationError):
    """Raised when GSTIN validation encounters fatal formatting or calculation issues."""
    pass

class FinancialReconciliationError(ValidationError):
    """Raised when mathematical totals fail reconciliation checks."""
    pass

class MandatoryFieldMissingError(ValidationError):
    """Raised when critical required invoice fields are absent."""
    pass
