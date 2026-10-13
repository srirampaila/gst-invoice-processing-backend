"""
Pydantic data models for Deterministic GST & Financial Validation.
"""
from typing import List, Optional, Dict, Any
from decimal import Decimal
from datetime import datetime, timezone
from pydantic import BaseModel, Field

class ValidationIssue(BaseModel):
    """Structured description of an individual validation finding or failure."""
    rule_id: str
    severity: str = Field(description="ERROR, WARNING, or INFO")
    field: str
    message: str
    expected: Optional[Any] = None
    actual: Optional[Any] = None

class GSTINValidationResult(BaseModel):
    """Validation analysis for an Indian GSTIN."""
    gstin: Optional[str] = None
    valid: bool = False
    state_code: Optional[str] = None
    state_name: Optional[str] = None
    pan: Optional[str] = None
    entity_code: Optional[str] = None
    checksum_char: Optional[str] = None
    calculated_checksum: Optional[str] = None
    checksum_valid: bool = False
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

class CalculatedFinancials(BaseModel):
    """Deterministic independently verified financial values."""
    calculated_item_taxable_sum: Optional[float] = None
    calculated_cgst: Optional[float] = None
    calculated_sgst: Optional[float] = None
    calculated_igst: Optional[float] = None
    calculated_tax_sum: Optional[float] = None
    calculated_grand_total: Optional[float] = None
    taxable_discrepancy: Optional[float] = None
    cgst_discrepancy: Optional[float] = None
    sgst_discrepancy: Optional[float] = None
    igst_discrepancy: Optional[float] = None
    grand_total_discrepancy: Optional[float] = None

class ValidationCheck(BaseModel):
    """Summary of a specific validation check executed."""
    check_name: str
    category: str = Field(description="FIELD, GSTIN, ARITHMETIC, TAX, CROSS_FIELD")
    status: str = Field(description="PASS, WARNING, FAIL, SKIPPED")
    message: str

class ValidationResult(BaseModel):
    """Complete validation assessment for an invoice."""
    invoice_id: str
    status: str = Field(description="PASS, WARNING, or FAIL")
    passed: bool = True
    requires_review: bool = False
    error_count: int = 0
    warning_count: int = 0
    issues: List[ValidationIssue] = Field(default_factory=list)
    checks: List[ValidationCheck] = Field(default_factory=list)
    calculated_values: CalculatedFinancials = Field(default_factory=CalculatedFinancials)
    seller_gstin_result: Optional[GSTINValidationResult] = None
    buyer_gstin_result: Optional[GSTINValidationResult] = None
    validation_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ValidationSummary(BaseModel):
    """Batch-level aggregate validation metrics."""
    total_invoices: int = 0
    pass_count: int = 0
    warning_count: int = 0
    fail_count: int = 0
    pass_rate: float = 0.0
    invalid_invoice_detection_rate: float = 0.0
    expected_invalid_count: int = 0
    detected_invalid_count: int = 0
    rule_metrics: Dict[str, Dict[str, int]] = Field(default_factory=dict)
