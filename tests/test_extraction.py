"""
Unit and integration tests for Phase 5 LLM Structured Extraction.
"""
import os
import pytest
import json
from unittest.mock import patch, MagicMock
from src.extraction.models import (
    CanonicalInvoice,
    Party,
    InvoiceItem,
    TaxSummary,
    ExtractionMetadata
)
from src.extraction.config import ExtractionConfig
from src.extraction.llm_client import MockLLMClient, OpenAIClient, GeminiClient, get_llm_client
from src.extraction.parser import (
    clean_json_string,
    compute_extraction_confidence,
    parse_and_validate_json
)
from src.extraction.service import ExtractionService, extract_invoice_from_ocr
from src.extraction.exceptions import (
    ExtractionError,
    LLMProviderError,
    LLMResponseParsingError,
    SchemaValidationError
)

def test_canonical_schema_defaults():
    invoice = CanonicalInvoice()
    assert invoice.invoice_number is None
    assert invoice.invoice_date is None
    assert invoice.currency == "INR"
    assert invoice.seller.gstin is None
    assert invoice.buyer.gstin is None
    assert len(invoice.items) == 0
    assert invoice.tax_summary.grand_total if hasattr(invoice.tax_summary, 'grand_total') else True
    assert invoice.grand_total is None

def test_canonical_schema_full():
    data = {
        "invoice_number": "INV-2026-001",
        "invoice_date": "2026-08-15",
        "seller": {"name": "Alpha Corp", "gstin": "29ABCDE1234F1Z5", "address": "Bengaluru"},
        "buyer": {"name": "Beta LLC", "gstin": "27XYZAB5678C1Z2", "address": "Mumbai"},
        "place_of_supply": "27",
        "items": [
            {
                "description": "Consulting Services",
                "quantity": 1.0,
                "unit_price": 5000.0,
                "taxable_value": 5000.0,
                "cgst_rate": 9.0,
                "cgst_amount": 450.0,
                "sgst_rate": 9.0,
                "sgst_amount": 450.0,
                "igst_rate": 0.0,
                "igst_amount": 0.0
            }
        ],
        "tax_summary": {
            "taxable_value": 5000.0,
            "cgst": 450.0,
            "sgst": 450.0,
            "igst": 0.0
        },
        "grand_total": 5900.0,
        "currency": "INR"
    }
    invoice = CanonicalInvoice.model_validate(data)
    assert invoice.invoice_number == "INV-2026-001"
    assert invoice.seller.gstin == "29ABCDE1234F1Z5"
    assert invoice.buyer.gstin == "27XYZAB5678C1Z2"
    assert len(invoice.items) == 1
    assert invoice.items[0].unit_price == 5000.0
    assert invoice.grand_total == 5900.0

def test_missing_optional_buyer_gstin():
    data = {
        "invoice_number": "INV-B2C-001",
        "invoice_date": "2026-08-15",
        "seller": {"name": "Retail Mart", "gstin": "29ABCDE1234F1Z5"},
        "buyer": {"name": "Consumer Individual", "gstin": None},
        "items": [
            {"description": "Item 1", "quantity": 1.0, "unit_price": 100.0, "taxable_value": 100.0}
        ],
        "tax_summary": {"taxable_value": 100.0, "cgst": 9.0, "sgst": 9.0},
        "grand_total": 118.0
    }
    invoice = CanonicalInvoice.model_validate(data)
    assert invoice.buyer.gstin is None
    assert invoice.buyer.name == "Consumer Individual"

def test_clean_json_string_fences():
    raw_markdown = "```json\n{\"invoice_number\": \"INV-001\"}\n```"
    cleaned = clean_json_string(raw_markdown)
    assert cleaned == "{\"invoice_number\": \"INV-001\"}"

def test_confidence_heuristic_complete():
    data = {
        "invoice_number": "INV-001",
        "invoice_date": "2026-08-15",
        "seller": {"name": "Seller", "gstin": "29ABCDE1234F1Z5"},
        "buyer": {"name": "Buyer", "gstin": "27XYZAB5678C1Z2"},
        "items": [{"description": "Item 0", "taxable_value": 100.0}],
        "tax_summary": {"taxable_value": 100.0},
        "grand_total": 118.0
    }
    score, warnings = compute_extraction_confidence(data)
    assert score == 1.0
    assert len(warnings) == 0

def test_confidence_heuristic_missing_critical():
    data = {
        "invoice_number": None,
        "invoice_date": "invalid-date",
        "seller": {"name": "Seller", "gstin": None},
        "items": [],
        "grand_total": None
    }
    score, warnings = compute_extraction_confidence(data)
    assert score < 0.5
    assert any("invoice_number" in w for w in warnings)
    assert any("seller GSTIN" in w for w in warnings)

def test_mock_llm_client_extraction():
    ocr_sample = """
    INVOICE NUMBER INV_CLEAN_001
    DATE 2026-08-15
    SELLERGSTIN: 29ABCDE1234F1Z5
    BUYERGSTIN: 27XYZAB5678C1Z2
    ITEM 0: 1000.0 (Taxable), CGST: 90.0, SGST: 90.0
    TOTAL TAXABLE: 1000.0
    CGST: 90.0
    SGST: 90.0
    GRAND TOTAL: 1180.0
    """
    client = MockLLMClient(ExtractionConfig(provider="mock"))
    res_json = client.generate_extraction(ocr_sample)
    invoice = parse_and_validate_json(res_json, source_file="INV_CLEAN_001.txt")
    
    assert invoice.invoice_number == "INV_CLEAN_001"
    assert invoice.invoice_date == "2026-08-15"
    assert invoice.seller.gstin == "29ABCDE1234F1Z5"
    assert invoice.buyer.gstin == "27XYZAB5678C1Z2"
    assert len(invoice.items) == 1
    assert invoice.tax_summary.taxable_value == 1000.0
    assert invoice.grand_total == 1180.0
    assert invoice.extraction_metadata.confidence == 1.0

def test_extraction_service_empty_text():
    service = ExtractionService(ExtractionConfig(provider="mock"))
    invoice = service.extract_from_ocr_text("", source_file="empty.txt")
    assert invoice.invoice_number is None
    assert invoice.extraction_metadata.confidence == 0.0
    assert "Empty or blank OCR text" in invoice.extraction_metadata.warnings[0]

def test_extraction_service_malformed_json():
    # Force parser to handle invalid json
    with pytest.raises(LLMResponseParsingError):
        parse_and_validate_json("not valid json at all", source_file="corrupt.txt")

def test_provider_factory_and_openai_missing_key():
    with patch.dict(os.environ, {}, clear=True):
        config = ExtractionConfig(provider="openai", api_key=None)
        with pytest.raises(LLMProviderError):
            OpenAIClient(config)
            
        mock_client = get_llm_client(ExtractionConfig(provider="mock"))
        assert isinstance(mock_client, MockLLMClient)

def test_openai_client_with_mocked_network():
    config = ExtractionConfig(provider="openai", api_key="sk-fake-test-key", model="gpt-4o-mini")
    client = OpenAIClient(config)
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "invoice_number": "INV-API-001",
                        "invoice_date": "2026-08-15",
                        "seller": {"name": "API Seller", "gstin": "29ABCDE1234F1Z5"},
                        "buyer": {"name": "API Buyer", "gstin": "27XYZAB5678C1Z2"},
                        "items": [],
                        "tax_summary": {"taxable_value": 500.0},
                        "grand_total": 590.0
                    })
                }
            }
        ]
    }
    
    with patch("requests.post", return_value=mock_response):
        raw_res = client.generate_extraction("sample ocr text")
        invoice = parse_and_validate_json(raw_res, source_file="test_api.txt", extraction_method="openai")
        assert invoice.invoice_number == "INV-API-001"
        assert invoice.grand_total == 590.0
        assert invoice.extraction_metadata.extraction_method == "openai"

def test_openai_client_http_error():
    config = ExtractionConfig(provider="openai", api_key="sk-fake-test-key")
    client = OpenAIClient(config)
    
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized: Invalid API key"
    
    with patch("requests.post", return_value=mock_response):
        with pytest.raises(LLMProviderError) as exc_info:
            client.generate_extraction("sample ocr text")
        assert "401" in str(exc_info.value)

def test_gemini_client_missing_key():
    with patch.dict(os.environ, {}, clear=True):
        config = ExtractionConfig(provider="gemini", api_key=None)
        with pytest.raises(LLMProviderError):
            GeminiClient(config)

def test_gemini_client_with_mocked_sdk():
    config = ExtractionConfig(provider="gemini", api_key="fake-gemini-key", model="gemini-2.5-flash")
    with patch("google.genai.Client") as mock_genai_client_cls:
        mock_instance = MagicMock()
        mock_genai_client_cls.return_value = mock_instance
        
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "invoice_number": "INV-GEMINI-001",
            "invoice_date": "2026-08-15",
            "seller": {"name": "Gemini Seller", "gstin": "29ABCDE1234F1Z5"},
            "buyer": {"name": "Gemini Buyer", "gstin": "27XYZAB5678C1Z2"},
            "items": [],
            "tax_summary": {"taxable_value": 1000.0},
            "grand_total": 1180.0
        })
        mock_instance.models.generate_content.return_value = mock_response

        client = GeminiClient(config)
        raw_res = client.generate_extraction("sample ocr text")
        invoice = parse_and_validate_json(raw_res, source_file="test_gemini.txt", extraction_method="gemini", model=config.model)
        
        assert invoice.invoice_number == "INV-GEMINI-001"
        assert invoice.grand_total == 1180.0
        assert invoice.extraction_metadata.extraction_method == "gemini"
        assert invoice.extraction_metadata.model == "gemini-2.5-flash"

def test_gemini_factory_selection():
    config = ExtractionConfig(provider="gemini", api_key="fake-key")
    with patch("google.genai.Client"):
        client = get_llm_client(config)
        assert isinstance(client, GeminiClient)
