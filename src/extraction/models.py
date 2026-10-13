"""
Canonical Invoice Data Models for Phase 5 LLM Structured Extraction.
"""
from typing import List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict

class Party(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: Optional[str] = None
    gstin: Optional[str] = None
    address: Optional[str] = None

class InvoiceItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    taxable_value: Optional[float] = None
    cgst_rate: Optional[float] = None
    cgst_amount: Optional[float] = None
    sgst_rate: Optional[float] = None
    sgst_amount: Optional[float] = None
    igst_rate: Optional[float] = None
    igst_amount: Optional[float] = None

class TaxSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")
    taxable_value: Optional[float] = None
    cgst: Optional[float] = None
    sgst: Optional[float] = None
    igst: Optional[float] = None

class ExtractionMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")
    source_file: Optional[str] = None
    extraction_method: str = "llm"
    model: Optional[str] = None
    extraction_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: List[str] = Field(default_factory=list)

class CanonicalInvoice(BaseModel):
    model_config = ConfigDict(extra="ignore")
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    seller: Party = Field(default_factory=Party)
    buyer: Party = Field(default_factory=Party)
    place_of_supply: Optional[str] = None
    items: List[InvoiceItem] = Field(default_factory=list)
    tax_summary: TaxSummary = Field(default_factory=TaxSummary)
    grand_total: Optional[float] = None
    currency: Optional[str] = "INR"
    extraction_metadata: ExtractionMetadata = Field(default_factory=ExtractionMetadata)
