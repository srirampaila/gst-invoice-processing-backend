import time
import cv2
import numpy as np
import math
from pathlib import Path
from src.ocr.config import OCRConfig
from src.ocr.models import OCRResult
from src.ocr.engine import TesseractEngine
from src.ocr.exceptions import OCREngineNotAvailableError, OCRProcessingError
from src.utils.logger import get_logger

logger = get_logger(__name__)

def extract_text(image_path: str, config: OCRConfig = None) -> OCRResult:
    start_time = time.time()
    if config is None:
        config = OCRConfig()
        
    path = Path(image_path)
    if not path.exists():
        return OCRResult(
            source_file=path.name, raw_text="", processing_time=0.0,
            average_confidence=None, word_count=0, character_count=0,
            success=False, error=f"File not found: {image_path}"
        )
        
    img = cv2.imread(str(path))
    if img is None:
        return OCRResult(
            source_file=path.name, raw_text="", processing_time=0.0,
            average_confidence=None, word_count=0, character_count=0,
            success=False, error="Corrupt or unreadable image format"
        )
        
    try:
        engine = TesseractEngine(config)
        
        # Get raw text
        raw_text = engine.extract_text(img)
        
        # Get detailed data for confidence
        data = engine.extract_data(img)
        # Filter out rows with confidence -1 (empty spaces)
        valid_data = data[data['conf'] != -1]
        
        if len(valid_data) > 0:
            avg_conf = valid_data['conf'].mean()
            word_count = len(valid_data)
        else:
            avg_conf = 0.0
            word_count = len(raw_text.split())
            
        char_count = len(raw_text)
        
        processing_time = time.time() - start_time
        
        return OCRResult(
            source_file=path.name,
            raw_text=raw_text,
            processing_time=processing_time,
            average_confidence=float(avg_conf) if not math.isnan(avg_conf) else 0.0,
            word_count=word_count,
            character_count=char_count,
            success=True
        )
        
    except OCREngineNotAvailableError as e:
        logger.error(str(e))
        return OCRResult(
            source_file=path.name, raw_text="", processing_time=time.time() - start_time,
            average_confidence=None, word_count=0, character_count=0,
            success=False, error=str(e)
        )
    except Exception as e:
        logger.error(f"OCR failed for {path.name}: {e}")
        return OCRResult(
            source_file=path.name, raw_text="", processing_time=time.time() - start_time,
            average_confidence=None, word_count=0, character_count=0,
            success=False, error=str(e)
        )
