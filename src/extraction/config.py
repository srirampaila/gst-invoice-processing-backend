"""
Configuration for LLM Structured Extraction supporting Mock, OpenAI, and Google Gemini providers.
"""
import os
from typing import Optional

class ExtractionConfig:
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        timeout: int = 30,
        max_retries: int = 2
    ):
        # Determine provider: explicitly passed > env LLM_PROVIDER > gemini (if GEMINI_API_KEY) > mock
        if provider:
            self.provider = provider.lower().strip()
        elif os.getenv("LLM_PROVIDER"):
            self.provider = os.getenv("LLM_PROVIDER").lower().strip()
        elif os.getenv("GEMINI_API_KEY"):
            self.provider = "gemini"
        elif os.getenv("OPENAI_API_KEY"):
            self.provider = "openai"
        else:
            self.provider = "mock"

        # Resolve model
        if model:
            self.model = model
        elif self.provider == "gemini":
            self.model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        elif self.provider == "openai":
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif self.provider in ("ollama", "local"):
            self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        else:
            self.model = os.getenv("LLM_MODEL", "mock-extractor")

        # Resolve API Key / Base URL
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        if api_key:
            self.api_key = api_key
        elif self.provider == "gemini":
            self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        elif self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        else:
            self.api_key = None

        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
