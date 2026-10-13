"""
Generator Orchestrator for Phase 8 GSTR-1 Draft Preparation.
"""
from typing import List, Tuple, Optional, Dict, Any, Union
from datetime import datetime, timezone

from src.extraction.models import CanonicalInvoice
from src.validation.models import ValidationResult
from src.confidence.models import ConfidenceResult, ReviewRecord
from src.gst.gstr1_models import (
    GSTR1Draft,
    GSTR1B2BInvoice,
    GSTR1B2CSInvoice,
    GSTR1ExclusionRecord
)
from src.gst.gstr1_mapper import map_canonical_to_gstr1
from src.gst.gstr1_validator import validate_gstr1_batch
from src.utils.logger import get_logger

logger = get_logger(__name__)

class GSTR1DraftGenerator:
    """Orchestrates GSTR-1 draft creation from validated invoice records."""

    def __init__(self):
        pass

    def generate_draft(
        self,
        invoice_entries: List[Dict[str, Any]]
    ) -> GSTR1Draft:
        """
        Processes a list of invoice dictionaries containing:
        - 'invoice': CanonicalInvoice
        - 'validation_result': Optional[ValidationResult]
        - 'confidence_result': Optional[ConfidenceResult]
        - 'review_record': Optional[ReviewRecord]
        - 'invoice_id': str
        """
        b2b_records: List[GSTR1B2BInvoice] = []
        b2c_records: List[GSTR1B2CSInvoice] = []
        exclusions: List[GSTR1ExclusionRecord] = []

        total_source_count = len(invoice_entries)
        seen_source_ids = set()

        for entry in invoice_entries:
            invoice: CanonicalInvoice = entry["invoice"]
            val_res: Optional[ValidationResult] = entry.get("validation_result")
            conf_res: Optional[ConfidenceResult] = entry.get("confidence_result")
            rev_rec: Optional[ReviewRecord] = entry.get("review_record")
            inv_id: str = entry.get("invoice_id") or invoice.invoice_number or "UNKNOWN"

            # Duplicate protection across incoming stream
            if inv_id in seen_source_ids:
                exclusions.append(GSTR1ExclusionRecord(
                    invoice_id=inv_id,
                    reason=f"Duplicate source invoice '{inv_id}' encountered in batch stream",
                    confidence_score=conf_res.overall_score if conf_res else 0.0,
                    confidence_tier=conf_res.confidence_tier if conf_res else "LOW",
                    recommendation="REJECT_OR_REPROCESS",
                    review_status=rev_rec.review_status if rev_rec else "PENDING_REVIEW",
                    validation_status=val_res.status if val_res else "FAIL",
                    error_count=1,
                    warning_count=0,
                    triggered_rules=["DUPLICATE_SOURCE_ID"]
                ))
                continue

            seen_source_ids.add(inv_id)

            record, exclusion = map_canonical_to_gstr1(
                invoice=invoice,
                validation_result=val_res,
                confidence_result=conf_res,
                review_record=rev_rec,
                invoice_id=inv_id
            )

            if exclusion:
                exclusions.append(exclusion)
            elif isinstance(record, GSTR1B2BInvoice):
                b2b_records.append(record)
            elif isinstance(record, GSTR1B2CSInvoice):
                b2c_records.append(record)

        # Run GSTR-1 statutory validation
        val_issues = validate_gstr1_batch(b2b_records, b2c_records)

        # Calculate Financial Aggregates
        tot_taxable = sum(r.taxable_value for r in b2b_records) + sum(r.taxable_value for r in b2c_records)
        tot_cgst = sum(r.cgst_amount for r in b2b_records) + sum(r.cgst_amount for r in b2c_records)
        tot_sgst = sum(r.sgst_amount for r in b2b_records) + sum(r.sgst_amount for r in b2c_records)
        tot_igst = sum(r.igst_amount for r in b2b_records) + sum(r.igst_amount for r in b2c_records)
        tot_val = sum(r.invoice_value for r in b2b_records) + sum(r.invoice_value for r in b2c_records)

        included_count = len(b2b_records) + len(b2c_records)
        excluded_count = len(exclusions)

        logger.info(
            f"GSTR-1 draft generated: {total_source_count} source invoices -> "
            f"{included_count} included ({len(b2b_records)} B2B, {len(b2c_records)} B2C), "
            f"{excluded_count} excluded, {len(val_issues)} draft validation findings"
        )

        return GSTR1Draft(
            draft_status="GSTR-1 DRAFT — NOT FILED",
            generated_at=datetime.now(timezone.utc).isoformat(),
            source_invoice_count=total_source_count,
            included_invoice_count=included_count,
            excluded_invoice_count=excluded_count,
            b2b_count=len(b2b_records),
            b2c_count=len(b2c_records),
            total_taxable_value=round(tot_taxable, 2),
            total_cgst=round(tot_cgst, 2),
            total_sgst=round(tot_sgst, 2),
            total_igst=round(tot_igst, 2),
            total_invoice_value=round(tot_val, 2),
            b2b_records=b2b_records,
            b2c_records=b2c_records,
            exclusions=exclusions,
            validation_issues=val_issues
        )
