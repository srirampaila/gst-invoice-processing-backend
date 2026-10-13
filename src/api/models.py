"""
Pydantic API request and response models for FastAPI bridge.
"""
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    environment: str
    ollama_ready: bool
    ollama_model: str
    installed_ollama_models: List[str]
    tesseract_available: bool
    gemini_configured: bool
    openai_configured: bool
    canonical_invoice_count: int

class ProviderInfo(BaseModel):
    id: str
    name: str
    description: str
    available: bool
    is_local: bool
    requires_key: bool
    key_configured: bool

class ProvidersResponse(BaseModel):
    providers: List[ProviderInfo]
    default_provider: str

class SampleInvoiceInfo(BaseModel):
    file_name: str
    invoice_id: str
    category: str
    file_size_kb: float
    has_review: bool
    review_status: Optional[str] = None
    image_url: str

class ProcessInvoiceRequest(BaseModel):
    file_name: str
    provider: str = "mock"
    amount_tolerance: float = 1.0
    high_threshold: float = 85.0
    medium_threshold: float = 60.0

class ReviewActionRequest(BaseModel):
    action: str = Field(..., description="APPROVE, REJECT, or REQUEST_REPROCESSING")
    notes: Optional[str] = None

class BatchGSTR1RunResponse(BaseModel):
    status: str
    message: str
    source_invoice_count: int
    included_invoice_count: int
    excluded_invoice_count: int
    b2b_count: int
    b2c_count: int
    total_taxable_value: float
    total_invoice_value: float
