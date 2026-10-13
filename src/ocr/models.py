from dataclasses import dataclass
from typing import Optional

@dataclass
class OCRResult:
    source_file: str
    raw_text: str
    processing_time: float
    average_confidence: Optional[float]
    word_count: int
    character_count: int
    success: bool
    error: Optional[str] = None
