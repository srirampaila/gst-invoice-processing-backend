"""
Pydantic Data Models for GSTR-1 Draft Preparation and Export.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

class GSTR1Table(str, Enum):
    B2B_4A = "B2B_Table_4A"
    B2CS_7 = "B2C_Table_7"

class GSTR1B2BInvoice(BaseModel):
    """
    Standardized schema for GSTR-1 Table 4A:
    Taxable outward supplies made to registered persons (including UIN holders).
    """
    model_config = ConfigDict(extra="ignore")
    source_invoice_id: str = Field(description="Lineage ID tracing back to source extracted invoice")
    recipient_gstin: str = Field(description="15-character GSTIN of the buyer/recipient")
    recipient_name: Optional[str] = Field(default=None, description="Trade or legal name of buyer")
    invoice_number: str = Field(description="Exact invoice number")
    invoice_date: str = Field(description="Invoice date in YYYY-MM-DD format")
    invoice_value: float = Field(description="Grand total invoice value (INR)")
    place_of_supply: str = Field(description="State code & name (e.g. 27-Maharashtra)")
    reverse_charge: str = Field(default="N", description="Applicability of reverse charge ('Y' or 'N')")
    invoice_type: str = Field(default="Regular", description="Invoice type (Regular, SEZ, Deemed)")
    taxable_value: float = Field(description="Total taxable value of goods/services")
    rate: float = Field(description="Applicable unified GST tax rate percentage (e.g. 18.0)")
    cgst_amount: float = Field(default=0.0, description="Central GST amount")
    sgst_amount: float = Field(default=0.0, description="State/UT GST amount")
    igst_amount: float = Field(default=0.0, description="Integrated GST amount")
    cess_amount: float = Field(default=0.0, description="Cess amount")
    source_validation_status: str = Field(default="PASS", description="Phase 6 validation status")
    source_confidence_score: float = Field(default=100.0, description="Phase 7 confidence score")

class GSTR1B2CSInvoice(BaseModel):
    """
    Standardized schema for GSTR-1 Table 7:
    Taxable outward supplies to unregistered persons (B2C Small).
    """
    model_config = ConfigDict(extra="ignore")
    source_invoice_id: str = Field(description="Lineage ID tracing back to source extracted invoice")
    place_of_supply: str = Field(description="State code & name (e.g. 27-Maharashtra)")
    invoice_number: Optional[str] = Field(default=None, description="Invoice number")
    invoice_date: Optional[str] = Field(default=None, description="Invoice date in YYYY-MM-DD format")
    invoice_value: float = Field(description="Grand total invoice value (INR)")
    taxable_value: float = Field(description="Total taxable value")
    rate: float = Field(description="Applicable GST tax rate percentage (e.g. 18.0)")
    cgst_amount: float = Field(default=0.0, description="Central GST amount")
    sgst_amount: float = Field(default=0.0, description="State/UT GST amount")
    igst_amount: float = Field(default=0.0, description="Integrated GST amount")
    cess_amount: float = Field(default=0.0, description="Cess amount")
    source_validation_status: str = Field(default="PASS", description="Phase 6 validation status")
    source_confidence_score: float = Field(default=100.0, description="Phase 7 confidence score")

class GSTR1ExclusionRecord(BaseModel):
    """
    Record of an invoice excluded from GSTR-1 draft with diagnostic reasons.
    """
    model_config = ConfigDict(extra="ignore")
    invoice_id: str
    reason: str = Field(description="Primary justification for exclusion from GSTR-1")
    confidence_score: float
    confidence_tier: str
    recommendation: str
    review_status: str
    validation_status: str
    error_count: int
    warning_count: int
    triggered_rules: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class GSTR1ValidationIssue(BaseModel):
    """
    Validation finding produced during GSTR-1 draft verification.
    """
    model_config = ConfigDict(extra="ignore")
    invoice_id: str
    rule_id: str
    severity: str = Field(description="ERROR or WARNING")
    field: str
    message: str

class GSTR1Draft(BaseModel):
    """
    Top-level composite model containing full GSTR-1 draft datasets,
    exclusions, validations, and aggregate totals.
    """
    model_config = ConfigDict(extra="ignore")
    draft_status: str = "GSTR-1 DRAFT — NOT FILED"
    legal_disclaimer: str = (
        "This system prepares GSTR-1 draft data only. "
        "It does not file returns with the GST portal."
    )
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_invoice_count: int = 0
    included_invoice_count: int = 0
    excluded_invoice_count: int = 0
    b2b_count: int = 0
    b2c_count: int = 0
    total_taxable_value: float = 0.0
    total_cgst: float = 0.0
    total_sgst: float = 0.0
    total_igst: float = 0.0
    total_invoice_value: float = 0.0
    b2b_records: List[GSTR1B2BInvoice] = Field(default_factory=list)
    b2c_records: List[GSTR1B2CSInvoice] = Field(default_factory=list)
    exclusions: List[GSTR1ExclusionRecord] = Field(default_factory=list)
    validation_issues: List[GSTR1ValidationIssue] = Field(default_factory=list)
