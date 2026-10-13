"""
LLM Provider client abstraction and implementations supporting Mock, OpenAI, and Google Gemini.
"""
import os
import re
import json
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from src.extraction.config import ExtractionConfig
from src.extraction.prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_PROMPT_TEMPLATE
from src.extraction.exceptions import LLMProviderError
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BaseLLMClient(ABC):
    """Abstract Base Class for LLM extraction providers."""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        
    @abstractmethod
    def generate_extraction(self, ocr_text: str) -> str:
        """Sends OCR text to the LLM and returns the raw JSON string response."""
        pass

class GeminiClient(BaseLLMClient):
    """Google Gemini API Client supporting official google-genai SDK with JSON mode."""
    
    def __init__(self, config: ExtractionConfig):
        super().__init__(config)
        if not self.config.api_key:
            raise LLMProviderError("Gemini API key is missing. Set GEMINI_API_KEY or LLM_API_KEY.")
            
        try:
            from google import genai
            from google.genai import types
            self._genai = genai
            self._types = types
            self.client = genai.Client(api_key=self.config.api_key)
        except ImportError:
            raise LLMProviderError("google-genai SDK is not installed. Install with `pip install google-genai`.")
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize Gemini Client: {e}")

    def generate_extraction(self, ocr_text: str) -> str:
        prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(ocr_text=ocr_text)
        
        try:
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    system_instruction=EXTRACTION_SYSTEM_PROMPT,
                    temperature=self.config.temperature,
                    response_mime_type="application/json",
                    automatic_function_calling=self._types.AutomaticFunctionCallingConfig(disable=True)
                )
            )
            if not response.text:
                raise LLMProviderError("Empty text response returned by Gemini API.")
            return response.text
        except Exception as e:
            raise LLMProviderError(f"Gemini API request failed: {e}")

class OpenAIClient(BaseLLMClient):
    """OpenAI API Client supporting GPT models with JSON mode."""
    
    def __init__(self, config: ExtractionConfig):
        super().__init__(config)
        if not self.config.api_key:
            raise LLMProviderError("OpenAI API key is missing. Set OPENAI_API_KEY or LLM_API_KEY.")
            
        self.api_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")

    def generate_extraction(self, ocr_text: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": EXTRACTION_USER_PROMPT_TEMPLATE.format(ocr_text=ocr_text)}
            ],
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            
            if response.status_code != 200:
                raise LLMProviderError(
                    f"OpenAI API request failed with status {response.status_code}: {response.text}"
                )
                
            res_data = response.json()
            return res_data["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            raise LLMProviderError(f"Network error connecting to LLM provider: {e}")

class OllamaClient(BaseLLMClient):
    """
    Local Ollama LLM Client supporting Qwen and other open-source models with JSON mode.
    Provides zero API cost, locally hosted real LLM structured extraction.
    """
    
    def __init__(self, config: ExtractionConfig):
        super().__init__(config)
        self.base_url = getattr(config, "ollama_base_url", os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
        self.api_url = f"{self.base_url.rstrip('/')}/api/generate"
        self.last_usage: Dict[str, Any] = {}

    def generate_extraction(self, ocr_text: str) -> str:
        prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(ocr_text=ocr_text)
        
        payload = {
            "model": self.config.model,
            "system": EXTRACTION_SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.config.temperature
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.config.timeout
            )
            
            if response.status_code != 200:
                raise LLMProviderError(
                    f"Ollama API request failed with status {response.status_code}: {response.text}"
                )
                
            res_data = response.json()
            prompt_eval = res_data.get("prompt_eval_count", 0) or 0
            eval_cnt = res_data.get("eval_count", 0) or 0
            self.last_usage = {
                "prompt_tokens": prompt_eval,
                "completion_tokens": eval_cnt,
                "total_tokens": prompt_eval + eval_cnt,
                "total_duration_ns": res_data.get("total_duration", 0)
            }
            return res_data.get("response", "")
            
        except requests.exceptions.RequestException as e:
            raise LLMProviderError(f"Network error connecting to local Ollama server at {self.api_url}: {e}")

class MockLLMClient(BaseLLMClient):
    """
    Deterministic Mock LLM client that parses OCR text using robust pattern matching.
    Provides offline resilience for local tests, CI/CD, and evaluation workflows.
    """
    
    def generate_extraction(self, ocr_text: str) -> str:
        data: Dict[str, Any] = {
            "invoice_number": None,
            "invoice_date": None,
            "seller": {"name": "Seller", "gstin": None, "address": "Blr"},
            "buyer": {"name": "Buyer", "gstin": None, "address": "Mum"},
            "place_of_supply": "27",
            "items": [],
            "tax_summary": {
                "taxable_value": None,
                "cgst": None,
                "sgst": None,
                "igst": 0.0
            },
            "grand_total": None,
            "currency": "INR"
        }

        # 1. Invoice Number (tolerant of spaces, colons, or noise)
        inv_match = re.search(r"INVOICE\s*NUMBER[:\s]*([A-Z0-9_-]+)", ocr_text, re.IGNORECASE)
        if inv_match:
            data["invoice_number"] = inv_match.group(1).strip()

        # 2. Date (ISO YYYY-MM-DD)
        date_match = re.search(r"DATE[:\s]*(\d{4}-\d{2}-\d{2})", ocr_text, re.IGNORECASE)
        if date_match:
            data["invoice_date"] = date_match.group(1).strip()

        # 3. Seller GSTIN
        seller_match = re.search(r"SELLER\s*GSTIN[:\s]*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1}[Z]{1}[0-9A-Z]{1})", ocr_text, re.IGNORECASE)
        if not seller_match:
            seller_match = re.search(r"SELLERGSTIN[:\s]*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1}[Z]{1}[0-9A-Z]{1})", ocr_text, re.IGNORECASE)
        if seller_match:
            data["seller"]["gstin"] = seller_match.group(1).strip()
            data["place_of_supply"] = seller_match.group(1)[:2]

        # 4. Buyer GSTIN
        buyer_match = re.search(r"BUYER\s*GSTIN[:\s]*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1}[Z]{1}[0-9A-Z]{1})", ocr_text, re.IGNORECASE)
        if not buyer_match:
            buyer_match = re.search(r"BUYERGSTIN[:\s]*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1}[Z]{1}[0-9A-Z]{1})", ocr_text, re.IGNORECASE)
        if buyer_match:
            data["buyer"]["gstin"] = buyer_match.group(1).strip()
            data["place_of_supply"] = buyer_match.group(1)[:2]
        else:
            # If no buyer gstin in OCR, buyer remains B2C with null gstin
            data["buyer"]["gstin"] = None

        # 5. Line items (ITEM <j>: <taxable> (Taxable), CGST: <cgst>, SGST: <sgst>)
        item_matches = re.finditer(
            r"ITEM\s*(\d+)[:\s]*([0-9]+(?:\.[0-9]+)?)\s*\(Taxable\)[,\s]*CGST[:\s]*([0-9]+(?:\.[0-9]+)?)[,\s]*SGST[:\s]*([0-9]+(?:\.[0-9]+)?)",
            ocr_text,
            re.IGNORECASE
        )
        for m in item_matches:
            idx = int(m.group(1))
            taxable = float(m.group(2))
            cgst = float(m.group(3))
            sgst = float(m.group(4))
            data["items"].append({
                "description": f"Product {idx}",
                "quantity": 1.0,
                "unit_price": taxable,
                "taxable_value": taxable,
                "cgst_rate": 9.0,
                "cgst_amount": cgst,
                "sgst_rate": 9.0,
                "sgst_amount": sgst,
                "igst_rate": 0.0,
                "igst_amount": 0.0
            })

        # 6. Totals (Extract from summary section)
        tv_match = re.search(r"TOTAL\s*TAXABLE[:\s]*([0-9]+(?:\.[0-9]+)?)", ocr_text, re.IGNORECASE)
        if tv_match:
            data["tax_summary"]["taxable_value"] = float(tv_match.group(1))
            totals_section = ocr_text[tv_match.start():]
        else:
            totals_section = ocr_text

        cgst_match = re.search(r"(?:^|\n)\s*CGST[:\s]*([0-9]+(?:\.[0-9]+)?)", totals_section, re.IGNORECASE)
        if not cgst_match:
            cgst_match = re.search(r"CGST[:\s]*([0-9]+(?:\.[0-9]+)?)", totals_section, re.IGNORECASE)
        if cgst_match:
            data["tax_summary"]["cgst"] = float(cgst_match.group(1))

        sgst_match = re.search(r"(?:^|\n)\s*SGST[:\s]*([0-9]+(?:\.[0-9]+)?)", totals_section, re.IGNORECASE)
        if not sgst_match:
            sgst_match = re.search(r"SGST[:\s]*([0-9]+(?:\.[0-9]+)?)", totals_section, re.IGNORECASE)
        if sgst_match:
            data["tax_summary"]["sgst"] = float(sgst_match.group(1))

        gt_match = re.search(r"GRAND\s*TOTAL[:\s]*([0-9]+(?:\.[0-9]+)?)", totals_section, re.IGNORECASE)
        if gt_match:
            data["grand_total"] = float(gt_match.group(1))

        return json.dumps(data)

def get_llm_client(config: Optional[ExtractionConfig] = None) -> BaseLLMClient:
    """Factory function for instantiating the configured LLM client."""
    if config is None:
        config = ExtractionConfig()
        
    provider = config.provider.lower().strip()
    if provider in ("gemini", "google"):
        return GeminiClient(config)
    elif provider == "openai":
        return OpenAIClient(config)
    elif provider in ("ollama", "local", "qwen"):
        return OllamaClient(config)
    elif provider in ("mock", "test"):
        return MockLLMClient(config)
    else:
        logger.warning(f"Unknown LLM provider '{provider}', falling back to MockLLMClient.")
        return MockLLMClient(config)
