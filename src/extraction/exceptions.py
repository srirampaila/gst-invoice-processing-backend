"""
Exceptions for Phase 5 LLM Structured Extraction.
"""

class ExtractionError(Exception):
    """Base exception for all extraction layer errors."""
    pass

class LLMProviderError(ExtractionError):
    """Raised when an LLM provider request fails or is misconfigured."""
    pass

class LLMResponseParsingError(ExtractionError):
    """Raised when the LLM response cannot be parsed into valid JSON."""
    pass

class SchemaValidationError(ExtractionError):
    """Raised when extracted JSON fails canonical schema validation."""
    pass
