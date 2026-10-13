"""
Service layer for Phase 5 LLM Structured Extraction.
"""
from pathlib import Path
from typing import Optional

from src.extraction.config import ExtractionConfig
from src.extraction.models import CanonicalInvoice, ExtractionMetadata
from src.extraction.llm_client import get_llm_client
from src.extraction.parser import parse_and_validate_json
from src.extraction.exceptions import ExtractionError
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ExtractionService:
    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self.client = get_llm_client(self.config)

    def extract_from_ocr_text(
        self,
        ocr_text: str,
        source_file: str = "unknown"
    ) -> CanonicalInvoice:
        """
        Takes raw OCR text, prompts the LLM provider, parses and validates
        the response, and returns a CanonicalInvoice instance.
        """
        if not ocr_text or not ocr_text.strip():
            logger.warning(f"Empty OCR text provided for {source_file}")
            return CanonicalInvoice(
                extraction_metadata=ExtractionMetadata(
                    source_file=source_file,
                    extraction_method=self.config.provider,
                    model=self.config.model,
                    confidence=0.0,
                    warnings=["Empty or blank OCR text received"]
                )
            )

        try:
            raw_response = self.client.generate_extraction(ocr_text)
            invoice = parse_and_validate_json(
                raw_response=raw_response,
                source_file=source_file,
                extraction_method=self.config.provider,
                model=self.config.model
            )
            return invoice
        except ExtractionError as e:
            logger.error(f"Extraction failed for {source_file}: {e}")
            return CanonicalInvoice(
                extraction_metadata=ExtractionMetadata(
                    source_file=source_file,
                    extraction_method=self.config.provider,
                    model=self.config.model,
                    confidence=0.0,
                    warnings=[f"Extraction error: {str(e)}"]
                )
            )
        except Exception as e:
            logger.error(f"Unexpected error during extraction for {source_file}: {e}")
            return CanonicalInvoice(
                extraction_metadata=ExtractionMetadata(
                    source_file=source_file,
                    extraction_method=self.config.provider,
                    model=self.config.model,
                    confidence=0.0,
                    warnings=[f"Unexpected error: {str(e)}"]
                )
            )

    def extract_from_ocr_file(
        self,
        file_path: str
    ) -> CanonicalInvoice:
        """Reads OCR text from a file and runs extraction."""
        path = Path(file_path)
        if not path.exists():
            return CanonicalInvoice(
                extraction_metadata=ExtractionMetadata(
                    source_file=path.name,
                    extraction_method=self.config.provider,
                    model=self.config.model,
                    confidence=0.0,
                    warnings=[f"File not found: {file_path}"]
                )
            )
        ocr_text = path.read_text(encoding="utf-8")
        return self.extract_from_ocr_text(ocr_text, source_file=path.name)

def extract_invoice_from_ocr(
    ocr_text: str,
    source_file: str = "unknown",
    config: Optional[ExtractionConfig] = None
) -> CanonicalInvoice:
    """Convenience function for invoice extraction."""
    service = ExtractionService(config)
    return service.extract_from_ocr_text(ocr_text, source_file=source_file)
