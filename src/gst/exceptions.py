"""
GSTR-1 Draft Generation Engine Exceptions.
"""

class GSTR1Error(Exception):
    """Base exception for GSTR-1 draft generation and mapping."""
    pass

class GSTR1EligibilityError(GSTR1Error):
    """Raised when an ineligible invoice attempts to enter GSTR-1 generation."""
    pass

class GSTR1MappingError(GSTR1Error):
    """Raised when canonical invoice data cannot be mapped to GSTR-1 schema."""
    pass

class GSTR1ValidationError(GSTR1Error):
    """Raised when GSTR-1 draft validation fails."""
    pass

class GSTR1ExportError(GSTR1Error):
    """Raised when GSTR-1 export generation fails."""
    pass
