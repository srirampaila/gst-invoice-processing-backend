"""
Confidence Engine Exceptions.
"""

class ConfidenceError(Exception):
    """Base exception for confidence scoring and review workflow."""
    pass

class ConfidenceConfigError(ConfidenceError):
    """Raised when confidence configuration is invalid (e.g. weights do not sum to 1.0)."""
    pass

class InvalidScoreError(ConfidenceError):
    """Raised when an invalid score calculation occurs."""
    pass
