"""
Service orchestrator for Deterministic GST & Financial Validation.
"""
import json
from pathlib import Path
from typing import Optional, Union, Dict, Any

from src.extraction.models import CanonicalInvoice
from src.validation.config import ValidationConfig
from src.validation.models import ValidationResult
from src.validation.rules import evaluate_all_rules
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ValidationService:
    """Orchestrates deterministic validation across canonical invoices."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()

    def validate_invoice(
        self,
        invoice: CanonicalInvoice,
        invoice_id: Optional[str] = None
    ) -> ValidationResult:
        """
        Runs comprehensive deterministic GST and arithmetic checks
        against a CanonicalInvoice instance.
        """
        inv_id = invoice_id or invoice.invoice_number or "UNKNOWN_INVOICE"
        
        issues, checks, calculated_financials, seller_gstin_res, buyer_gstin_res = evaluate_all_rules(
            invoice=invoice,
            config=self.config
        )
        
        error_count = sum(1 for i in issues if i.severity == "ERROR")
        warning_count = sum(1 for i in issues if i.severity == "WARNING")
        
        if error_count > 0:
            status = "FAIL"
            passed = False
            requires_review = True
        elif warning_count > 0:
            status = "WARNING"
            passed = True
            requires_review = True
        else:
            status = "PASS"
            passed = True
            requires_review = False
            
        logger.info(f"Validation completed for {inv_id}: status={status}, errors={error_count}, warnings={warning_count}")

        return ValidationResult(
            invoice_id=inv_id,
            status=status,
            passed=passed,
            requires_review=requires_review,
            error_count=error_count,
            warning_count=warning_count,
            issues=issues,
            checks=checks,
            calculated_values=calculated_financials,
            seller_gstin_result=seller_gstin_res,
            buyer_gstin_result=buyer_gstin_res
        )

    def validate_invoice_json(
        self,
        json_data: Union[str, Dict[str, Any]],
        invoice_id: Optional[str] = None
    ) -> ValidationResult:
        """Parses a JSON string or dict into CanonicalInvoice and validates it."""
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
            
        invoice = CanonicalInvoice.model_validate(data)
        return self.validate_invoice(invoice, invoice_id=invoice_id)

    def validate_invoice_file(
        self,
        file_path: Union[str, Path]
    ) -> ValidationResult:
        """Reads a CanonicalInvoice JSON file from disk and validates it."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        invoice_id = path.stem
        return self.validate_invoice_json(data, invoice_id=invoice_id)

def validate_canonical_invoice(
    invoice: CanonicalInvoice,
    config: Optional[ValidationConfig] = None,
    invoice_id: Optional[str] = None
) -> ValidationResult:
    """Convenience helper to validate a single CanonicalInvoice."""
    service = ValidationService(config)
    return service.validate_invoice(invoice, invoice_id=invoice_id)
